"""
app/core/chunking.py
─────────────────────
Timestamp-aware text chunking.

The core challenge here:
  LangChain's RecursiveCharacterTextSplitter splits text beautifully,
  but it loses track of WHICH original segments contributed to each chunk.
  We need to know the start_time and end_time of every chunk for citations.

Our approach:
  1. Convert transcript segments into "mega-segments" by merging adjacent
     segments until we hit a target token count (~600 tokens ≈ 2400 chars).
  2. Run RecursiveCharacterTextSplitter on the merged text, but ONLY as a
     safety net to catch edge cases where a single segment is extremely long.
  3. Track which original segments contribute to each chunk so we always
     have accurate start/end timestamps.

Output: list of ChunkData dicts, each with:
  - text       : the chunk's text content
  - start_time : seconds into the video where this chunk begins
  - end_time   : seconds into the video where this chunk ends
"""

from dataclasses import dataclass
from typing import List
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Configuration ──────────────────────────────────────────────────────────────
# Target: ~600 tokens per chunk, ~100 token overlap between adjacent chunks.
# We convert to characters because RecursiveCharacterTextSplitter uses chars.
# Rough estimate: 1 token ≈ 4 characters for English text.

TARGET_CHUNK_TOKENS = 600
OVERLAP_TOKENS = 100
CHARS_PER_TOKEN = 4  # conservative estimate

TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN   # 2400
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN              # 400

# Tokenizer for accurate token counting
# cl100k_base is the tokenizer used by GPT-4 and is a good general standard
_tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkData:
    """
    Represents one chunk of transcript text with its timestamp range.
    This is what gets stored in ChromaDB and SQLite.
    """
    text: str
    start_time: float   # seconds from video start
    end_time: float     # seconds from video start
    chunk_index: int    # position in the ordered list of chunks for this video


# ── Main Chunking Function ─────────────────────────────────────────────────────

def chunk_transcript(segments: List[dict]) -> List[ChunkData]:
    """
    Convert transcript segments (from transcript.py) into chunks with timestamps.

    Args:
      segments: list of {"text": str, "start": float, "duration": float}

    Returns:
      list of ChunkData objects, ordered by start_time.
    """
    if not segments:
        return []

    # Step 1: Merge consecutive segments into groups of ~TARGET_CHUNK_TOKENS each.
    # This is our primary chunking strategy — it naturally respects speech boundaries.
    merged_groups = _merge_segments_into_groups(segments)

    # Step 2: For safety, run RecursiveCharacterTextSplitter on any group
    # that ended up way too large (e.g., a single segment with 2000+ tokens).
    all_chunks = []
    for group in merged_groups:
        token_count = _count_tokens(group["text"])
        if token_count > TARGET_CHUNK_TOKENS * 2:
            # This group is too large — split it further
            sub_chunks = _split_large_group(group)
            all_chunks.extend(sub_chunks)
        else:
            all_chunks.append(group)

    # Step 3: Add chunk_index and wrap in ChunkData objects
    chunk_data_list = []
    for i, chunk in enumerate(all_chunks):
        chunk_data_list.append(ChunkData(
            text=chunk["text"].strip(),
            start_time=chunk["start_time"],
            end_time=chunk["end_time"],
            chunk_index=i,
        ))

    print(f"[chunking] Created {len(chunk_data_list)} chunks from {len(segments)} segments.")
    _log_chunk_stats(chunk_data_list)

    return chunk_data_list


# ── Segment Merging ────────────────────────────────────────────────────────────

def _merge_segments_into_groups(segments: List[dict]) -> List[dict]:
    """
    Greedily merge consecutive transcript segments until we reach the
    target token count. When we finish a group, we start a new one.

    We also implement a simple overlap: the last N characters of the
    previous group are prepended to the next group. This prevents answers
    that span chunk boundaries from being missed by retrieval.

    Returns:
      list of dicts: {"text": str, "start_time": float, "end_time": float}
    """
    groups = []
    current_texts = []
    current_start = None
    current_end = None
    current_tokens = 0
    overlap_text = ""  # text carried over from previous chunk for overlap

    for seg in segments:
        seg_text = seg["text"]
        seg_start = seg["start"]
        seg_end = seg["start"] + seg["duration"]
        seg_tokens = _count_tokens(seg_text)

        # Initialize a new group
        if current_start is None:
            current_start = seg_start
            # Add overlap text from previous group if available
            if overlap_text:
                current_texts = [overlap_text]
                current_tokens = _count_tokens(overlap_text)
            else:
                current_texts = []
                current_tokens = 0

        # If adding this segment would exceed our target, close the current group
        if current_tokens + seg_tokens > TARGET_CHUNK_TOKENS and current_texts:
            # Save current group
            group_text = " ".join(current_texts)
            groups.append({
                "text": group_text,
                "start_time": current_start,
                "end_time": current_end,
            })

            # Prepare overlap for next chunk (take last OVERLAP_CHARS characters)
            overlap_text = group_text[-OVERLAP_CHARS:] if len(group_text) > OVERLAP_CHARS else group_text

            # Start new group
            current_start = seg_start
            current_texts = [overlap_text, seg_text] if overlap_text else [seg_text]
            current_tokens = _count_tokens(" ".join(current_texts))
            current_end = seg_end
        else:
            # Add segment to current group
            current_texts.append(seg_text)
            current_tokens += seg_tokens
            current_end = seg_end

    # Don't forget the last group
    if current_texts:
        groups.append({
            "text": " ".join(current_texts),
            "start_time": current_start,
            "end_time": current_end,
        })

    return groups


def _split_large_group(group: dict) -> List[dict]:
    """
    Use LangChain's RecursiveCharacterTextSplitter to split an oversized
    group into smaller pieces.

    We distribute timestamps proportionally across the sub-chunks since
    we can't track exact segment boundaries after merging.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHUNK_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )

    sub_texts = splitter.split_text(group["text"])
    if not sub_texts:
        return [group]

    # Distribute the timestamp range proportionally across sub-chunks
    total_chars = len(group["text"])
    time_range = group["end_time"] - group["start_time"]
    sub_chunks = []
    char_offset = 0

    for sub_text in sub_texts:
        sub_start = group["start_time"] + (char_offset / total_chars) * time_range
        sub_end = sub_start + (len(sub_text) / total_chars) * time_range
        sub_chunks.append({
            "text": sub_text,
            "start_time": round(sub_start, 2),
            "end_time": min(round(sub_end, 2), group["end_time"]),
        })
        char_offset += len(sub_text)

    return sub_chunks


# ── Utility Functions ──────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Count tokens in a string using the cl100k_base tokenizer."""
    return len(_tokenizer.encode(text))


def _log_chunk_stats(chunks: List[ChunkData]):
    """Print useful stats about the chunks for debugging."""
    if not chunks:
        return
    token_counts = [_count_tokens(c.text) for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    print(f"[chunking] Token stats — avg: {avg_tokens:.0f}, min: {min_tokens}, max: {max_tokens}")


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to a human-readable timestamp string.
    Example: 125.4 → "2:05"
    Used in the frontend and in LLM prompts.
    """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"
