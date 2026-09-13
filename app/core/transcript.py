"""
app/core/transcript.py
───────────────────────
Transcript ingestion using youtube-transcript-api 1.2.x with full
multi-language support.

Language Strategy (in order of preference):
  1. English manual transcript  -> use directly
  2. English auto-generated     -> use directly
  3. Any other language + YouTube built-in translation to English -> use
  4. Multi-layer translation engine:
     - Primary: Fast Google Translator (deep-translator) in grouped batches (0 Gemini quota used, ~10-15s total)
     - Secondary Fallback: Gemini LLM translator
     - Tertiary Fallback: Original text

This means ANY video with captions (Hindi, Spanish, Japanese, French, etc.)
can be processed and chatted with in English with accurate timestamps.
"""

import re
import os
import httpx
from typing import List
from requests import Session
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


# ── Cookie Session Builder ─────────────────────────────────────────────────────

# Path to manually exported cookies.txt file (Netscape format)
# Users can export this using browser extensions like "Get cookies.txt LOCALLY"
COOKIES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cookies.txt")


def _load_netscape_cookies(filepath: str, session: Session) -> int:
    """
    Load cookies from a Netscape-format cookies.txt file into a requests.Session.
    Returns the number of cookies loaded.
    """
    import http.cookiejar
    jar = http.cookiejar.MozillaCookieJar()
    try:
        jar.load(filepath, ignore_discard=True, ignore_expires=True)
        session.cookies.update(jar)
        return len(list(jar))
    except Exception as e:
        raise RuntimeError(f"Failed to load cookies.txt: {e}")


def _build_youtube_session() -> Session:
    """
    Build a requests.Session that bypasses YouTube IP blocks.

    Priority order:
      1. cookies.txt file in the project root (most reliable — Netscape format)
      2. Edge browser cookies (Windows-native, most likely to work without admin)
      3. Chrome browser cookies (may need Chrome to be closed)
      4. Firefox browser cookies
      5. Plain session with browser-like User-Agent (last resort)
    """
    session = Session()
    session.headers.update({
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    })

    # ── Strategy 1: cookies.txt file ──────────────────────────────────────────
    if os.path.exists(COOKIES_FILE):
        try:
            count = _load_netscape_cookies(COOKIES_FILE, session)
            print(f"[transcript] [OK] Loaded {count} cookies from cookies.txt.")
            return session
        except Exception as e:
            print(f"[transcript] cookies.txt load failed: {e}. Trying browser...")

    # ── Strategy 2–4: Auto-extract from browser ───────────────────────────────
    try:
        import browser_cookie3
        browsers_to_try = [
            ("Edge",    browser_cookie3.edge),
            ("Chrome",  browser_cookie3.chrome),
            ("Firefox", browser_cookie3.firefox),
            ("Brave",   browser_cookie3.brave),
        ]
        for name, fn in browsers_to_try:
            try:
                cookies = fn(domain_name=".youtube.com")
                session.cookies.update(cookies)
                count = sum(1 for _ in session.cookies)
                if count > 0:
                    print(f"[transcript] [OK] Loaded YouTube cookies from {name} ({count} cookies).")
                    return session
                else:
                    print(f"[transcript] {name}: 0 cookies found, trying next...")
            except Exception as be:
                print(f"[transcript] {name} cookie load failed: {be}. Trying next...")
    except ImportError:
        pass
    except Exception as e:
        print(f"[transcript] browser-cookie3 error: {e}")

    # ── Strategy 5: Plain session (may be blocked by YouTube) ────────────────
    print(
        "[transcript] [WARN] No cookies found. Using plain session. "
        "If YouTube blocks this, export cookies.txt from your browser — see README."
    )
    return session


# ── URL Parsing ────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from any common URL format.
      - https://www.youtube.com/watch?v=dQw4w9WgXcQ
      - https://youtu.be/dQw4w9WgXcQ
      - https://www.youtube.com/embed/dQw4w9WgXcQ
      - https://youtube.com/shorts/dQw4w9WgXcQ
      - dQw4w9WgXcQ  (raw ID)

    Raises ValueError if no valid video ID found.
    """
    url = url.strip()

    match = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)

    match = re.search(r"(?:youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)

    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    raise ValueError(
        f"Could not extract a YouTube video ID from: '{url}'. "
        "Please paste a standard YouTube URL (e.g. https://www.youtube.com/watch?v=...)."
    )


# ── Metadata Fetch ─────────────────────────────────────────────────────────────

def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetch video title and channel name using YouTube's oEmbed endpoint.
    No API key required. Falls back gracefully if request fails.
    """
    oembed_url = (
        f"https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )
    try:
        response = httpx.get(oembed_url, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        return {
            "title": data.get("title", f"YouTube Video ({video_id})"),
            "channel": data.get("author_name", "Unknown Channel"),
        }
    except Exception as e:
        print(f"[transcript] Warning: Could not fetch metadata for {video_id}: {e}")
        return {
            "title": f"YouTube Video ({video_id})",
            "channel": "Unknown Channel",
        }


# ── Transcript Fetch ───────────────────────────────────────────────────────────

def fetch_transcript(youtube_url: str) -> dict:
    """
    Main entry point. Takes any YouTube URL and returns structured transcript data.
    Supports ANY language — translates to English automatically.

    Returns:
      {
        "video_id"          : str,
        "title"             : str,
        "channel"           : str,
        "segments"          : [{"text": str, "start": float, "duration": float}, ...],
        "full_text"         : str,
        "detected_language" : str,
        "was_translated"    : bool,
      }

    Raises:
      ValueError  : invalid URL or no transcript available
      RuntimeError: video unavailable or unexpected error
    """
    # Step 1: Extract video ID
    video_id = extract_video_id(youtube_url)
    print(f"[transcript] Extracted video ID: {video_id}")

    # Step 2: Fetch metadata
    metadata = fetch_video_metadata(video_id)
    print(f"[transcript] Video title: {metadata['title']}")

    # Step 3: Fetch transcript with multi-language fallback
    # Use browser cookies to bypass YouTube IP blocks (looks like a real browser request)
    http_session = _build_youtube_session()
    api = YouTubeTranscriptApi(http_client=http_session)
    raw_segments = None
    detected_language = "en"
    was_translated = False


    try:
        transcript_list = api.list(video_id)

        # ── Strategy 1: Manual English transcript ──────────────────────────────
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
            raw_segments = transcript.fetch()
            detected_language = "en"
            print("[transcript] [OK] Using manually created English transcript.")

        except NoTranscriptFound:

            # ── Strategy 2: Auto-generated English transcript ──────────────────
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
                raw_segments = transcript.fetch()
                detected_language = "en"
                print("[transcript] [OK] Using auto-generated English transcript.")

            except NoTranscriptFound:

                # ── Strategy 3: Non-English transcript + YouTube translation ───
                all_transcripts = list(transcript_list)
                if not all_transcripts:
                    raise ValueError("No transcripts found for this video.")

                # Pick the best available: prefer manual over auto-generated
                source_transcript = None
                for t in all_transcripts:
                    if not t.is_generated:
                        source_transcript = t
                        break
                if source_transcript is None:
                    source_transcript = all_transcripts[0]

                orig_lang = source_transcript.language_code
                orig_lang_name = source_transcript.language
                print(f"[transcript] Found transcript in '{orig_lang_name}' ({orig_lang}).")

                # Check if YouTube supports direct translation
                if getattr(source_transcript, "is_translatable", False):
                    try:
                        translated = source_transcript.translate("en")
                        raw_segments = translated.fetch()
                        detected_language = orig_lang
                        was_translated = True
                        print(f"[transcript] [OK] YouTube translated '{orig_lang_name}' -> English ({len(raw_segments)} segments).")
                    except Exception as yt_translate_err:
                        print(f"[transcript] [INFO] YouTube translation failed ({yt_translate_err}). Translating to English...")
                        raw_orig_segments = source_transcript.fetch()
                        raw_segments = _translate_segments(raw_orig_segments, orig_lang_name)
                        detected_language = orig_lang
                        was_translated = True
                else:
                    print(f"[transcript] [INFO] YouTube translation not available for '{orig_lang_name}'. Translating to English...")
                    raw_orig_segments = source_transcript.fetch()
                    raw_segments = _translate_segments(raw_orig_segments, orig_lang_name)
                    detected_language = orig_lang
                    was_translated = True

    except TranscriptsDisabled:
        raise ValueError(
            "Transcripts are disabled for this video. "
            "The video owner has turned off subtitles/captions."
        )
    except VideoUnavailable:
        raise RuntimeError(
            f"Video '{video_id}' is unavailable. "
            "It may be private, deleted, or region-restricted."
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching transcript: {str(e)}")

    if raw_segments is None:
        raise ValueError("Failed to fetch transcript from all strategies.")

    # Step 4: Clean segments
    cleaned_segments = []
    for seg in raw_segments:
        text = seg.text if hasattr(seg, "text") else seg.get("text", "")
        start = float(seg.start if hasattr(seg, "start") else seg.get("start", 0))
        duration = float(seg.duration if hasattr(seg, "duration") else seg.get("duration", 0))
        clean_text = _clean_segment_text(text)
        if clean_text:
            cleaned_segments.append({
                "text": clean_text,
                "start": start,
                "duration": duration,
            })

    if not cleaned_segments:
        raise ValueError("Transcript was fetched but contained no usable text segments.")

    full_text = " ".join(seg["text"] for seg in cleaned_segments)

    print(
        f"[transcript] Processed {len(cleaned_segments)} segments "
        f"({len(full_text)} chars). Language: {detected_language}. "
        f"Translated: {was_translated}"
    )

    return {
        "video_id": video_id,
        "title": metadata["title"],
        "channel": metadata["channel"],
        "segments": cleaned_segments,
        "full_text": full_text,
        "detected_language": detected_language,
        "was_translated": was_translated,
    }


# ── Multi-Language Translation Engine ──────────────────────────────────────────

def _group_segments_for_translation(
    raw_segments: list,
    target_duration: float = 25.0,
    max_words: int = 60,
) -> list:
    """
    Group micro-segments (1-3s fragments) into natural speech blocks (~25s / ~60 words).
    This dramatically improves translation fluency (sentence context) and
    speeds up translation by 80-90%.
    """
    blocks = []
    curr_texts = []
    curr_start = None
    curr_end = None

    for seg in raw_segments:
        text = seg.text if hasattr(seg, "text") else seg.get("text", "")
        text = _clean_segment_text(text)
        if not text:
            continue

        start = float(seg.start if hasattr(seg, "start") else seg.get("start", 0))
        duration = float(seg.duration if hasattr(seg, "duration") else seg.get("duration", 0))
        end = start + duration

        if curr_start is None:
            curr_start = start
        curr_texts.append(text)
        curr_end = end

        combined = " ".join(curr_texts)
        if (curr_end - curr_start >= target_duration) or len(combined.split()) >= max_words:
            blocks.append({
                "text": combined,
                "start": round(curr_start, 2),
                "duration": round(curr_end - curr_start, 2),
            })
            curr_texts = []
            curr_start = None
            curr_end = None

    if curr_texts:
        blocks.append({
            "text": " ".join(curr_texts),
            "start": round(curr_start, 2),
            "duration": round(curr_end - curr_start, 2),
        })

# ── Helpers & Sanitization ───────────────────────────────────────────────────

ERROR_PATTERNS = [
    r"Error\s+500\s*\(Server\s+Error\)[^.\n]*?\.?\s*(?:That's\s+an\s+error\.)?",
    r"There\s+was\s+an\s+error\.\s*Please\s+try\s+again\s+later\.",
    r"That's\s+all\s+we\s+know\.",
    r"Error\s+500\s*\(Server\s+Error\)[^.]*?",
    r"500\.\s*That's\s+an\s+error\.",
    r"<!DOCTYPE\s+html[^>]*>",
    r"<html[^>]*>[\s\S]*?</html>",
]

def _is_error_response(text: str) -> bool:
    """
    Detect HTML error pages and scraper failures in raw text.
    Used when checking raw segment input and Google Translator output.
    Does NOT check for non-ASCII — raw Hindi/regional input is legitimately non-ASCII.
    """
    if not text or not isinstance(text, str):
        return True
    t_lower = text.lower().strip()
    if "error 500" in t_lower or "that's all we know" in t_lower:
        return True
    if "500.that's an error" in t_lower or "please try again later" in t_lower:
        return True
    if "<!doctype" in t_lower or "<html" in t_lower or "<head" in t_lower:
        return True
    if "an error occurred" in t_lower and "please try again" in t_lower:
        return True
    return False


def _is_bad_translation_output(text: str) -> bool:
    """
    Detect bad/failed translation output from the LLM or Google Translator.
    Used ONLY when validating model-generated English translations.
    Includes non-ASCII check: a valid English translation should be mostly ASCII.
    """
    if _is_error_response(text):
        return True
    if not text or not isinstance(text, str):
        return True
    t_lower = text.lower().strip()
    # LLM refusal messages
    if "i'm sorry" in t_lower and ("cannot" in t_lower or "unable" in t_lower):
        return True
    if t_lower.startswith("sorry,") and len(t_lower) < 200:
        return True
    # Mostly non-ASCII → translation failed (returned original foreign text)
    if len(text) > 10:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.5:
            return True
    return False



def _translate_segments_with_gemini(
    blocks: list,
    source_language: str,
    batch_size: int = 20,
) -> list:
    """
    High-quality educational translation using Gemini.

    Strategy:
    - Splits blocks into batches of 20 (smaller = more reliable parsing)
    - Runs batches in parallel across Key 1 and Key 2 for 2x speed
    - Retries any failed/unparsed batch once with a smaller batch size (10)
    - Falls back to original text only for truly unrecoverable batches
    - Uses TRANSLATION_MODELS (non-thinking only — no gemini-2.5-flash)
    """
    try:
        from app.core.llm import _call_gemini_with_fallback, TRANSLATION_MODELS
        from google.genai import types as genai_types
        import concurrent.futures
    except Exception as e:
        print(f"[transcript] [WARN] Cannot import LLM fallback engine: {e}")
        return blocks

    def _build_prompt(batch_blocks, lang):
        numbered = "\n".join(f"{i+1}| {b['text']}" for i, b in enumerate(batch_blocks))
        return f"""You are a professional translator. Translate the following {lang} lecture transcript lines into clear, academic English.

STRICT OUTPUT RULES — READ CAREFULLY:
- Output EXACTLY {len(batch_blocks)} lines, one per input line.
- Each output line MUST start with its number followed by a pipe: 1| 2| 3| etc.
- Use PLAIN TEXT ONLY. No markdown, no asterisks, no bold, no headers.
- Preserve ALL technical terms, library names, and code exactly as spoken.
- Do NOT add preamble, explanations, or any extra text before or after the numbered lines.
- Do NOT merge multiple lines into one.
- Do NOT output the original {lang} text.

{lang} input:
{numbered}

English output (exactly {len(batch_blocks)} numbered lines):"""

    def _call_batch(batch_blocks, key_preference, lang):
        """Translate one batch, return dict of {0-based-index: translated_text}."""
        prompt = _build_prompt(batch_blocks, lang)
        try:
            response = _call_gemini_with_fallback(
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.05,
                    max_output_tokens=4096,
                ),
                model_candidates=TRANSLATION_MODELS,
                preferred_key=key_preference,
            )
            raw = response.text.strip()
            return _parse_pipe_translations(raw, len(batch_blocks)), raw
        except Exception as e:
            return {}, str(e)

    # Split blocks into batches
    batches = []
    for i in range(0, len(blocks), batch_size):
        batches.append((i, blocks[i: i + batch_size]))

    total_batches = len(batches)
    print(f"[transcript] [Gemini] Translating {len(blocks)} speech blocks across {total_batches} batches (parallel dual-key)...")

    # Translate all batches in parallel using both API keys
    batch_results = {}  # batch_start_idx -> {0-based local idx: text}

    def _translate_batch_worker(args):
        batch_start, batch_blocks, batch_num = args
        # Alternate key assignment: even batches → primary, odd → secondary
        key_pref = "primary" if (batch_num % 2 == 0) else "secondary"
        parsed, raw = _call_batch(batch_blocks, key_pref, source_language)

        if len(parsed) < len(batch_blocks) * 0.5:
            # Less than 50% parsed — retry once with smaller sub-batches
            print(f"[transcript] [WARN] Batch {batch_num+1}/{total_batches} only {len(parsed)}/{len(batch_blocks)} parsed. Retrying in halves...")
            mid = len(batch_blocks) // 2
            parsed_a, _ = _call_batch(batch_blocks[:mid], "primary", source_language)
            parsed_b, _ = _call_batch(batch_blocks[mid:], "secondary", source_language)
            # Merge: parsed_b keys need offset
            merged = {k: v for k, v in parsed_a.items()}
            for k, v in parsed_b.items():
                merged[k + mid] = v
            parsed = merged
            print(f"[transcript] [OK] Batch {batch_num+1} retry: {len(parsed)}/{len(batch_blocks)} parsed")
        else:
            print(f"[transcript] [OK] Gemini batch {batch_num+1}/{total_batches} ({len(batch_blocks)} blocks) → {len(parsed)} parsed")

        return batch_start, parsed

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_translate_batch_worker, (start, bblocks, bnum))
            for bnum, (start, bblocks) in enumerate(batches)
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                start_idx, parsed = fut.result()
                batch_results[start_idx] = parsed
            except Exception as e:
                print(f"[transcript] [WARN] A translation batch worker raised: {e}")

    # Assemble final translated segments in original order
    translated_segments = []
    for i, b in enumerate(blocks):
        # Find which batch this block belongs to
        batch_start = (i // batch_size) * batch_size
        local_idx = i - batch_start
        parsed_batch = batch_results.get(batch_start, {})
        text_val = parsed_batch.get(local_idx, "").strip()

        if not text_val or _is_bad_translation_output(text_val):
            # Fall back to original text
            text_val = b["text"]

        clean_val = _clean_segment_text(text_val)
        translated_segments.append({
            "text": clean_val if clean_val else b["text"],
            "start": b["start"],
            "duration": b["duration"],
        })

    parsed_count = sum(1 for i, b in enumerate(blocks)
                       for bs in [(i // batch_size) * batch_size]
                       if batch_results.get(bs, {}).get(i - bs))
    print(f"[transcript] Translation complete: {parsed_count}/{len(blocks)} blocks translated.")
    return translated_segments


def _clean_segment_text(text: str) -> str:
    """Remove HTML tags, strip translation error artifacts, and normalize whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    for pat in ERROR_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text.strip()


# ── Multi-Language Translation Engine ──────────────────────────────────────────

def _group_segments_for_translation(
    raw_segments: list,
    target_duration: float = 25.0,
    max_words: int = 60,
) -> list:
    """
    Group micro-segments (1-3s fragments) into natural speech blocks (~25s / ~60 words).
    This dramatically improves translation fluency (sentence context) and
    speeds up translation by 80-90%.
    """
    blocks = []
    curr_texts = []
    curr_start = None
    curr_end = None

    for seg in raw_segments:
        text = seg.text if hasattr(seg, "text") else seg.get("text", "")
        text = _clean_segment_text(text)
        if not text or _is_error_response(text):
            continue

        start = float(seg.start if hasattr(seg, "start") else seg.get("start", 0))
        duration = float(seg.duration if hasattr(seg, "duration") else seg.get("duration", 0))
        end = start + duration

        if curr_start is None:
            curr_start = start
        curr_texts.append(text)
        curr_end = end

        combined = " ".join(curr_texts)
        if (curr_end - curr_start >= target_duration) or len(combined.split()) >= max_words:
            clean_comb = _clean_segment_text(combined)
            if clean_comb and not _is_error_response(clean_comb):
                blocks.append({
                    "text": clean_comb,
                    "start": round(curr_start, 2),
                    "duration": round(curr_end - curr_start, 2),
                })
            curr_texts = []
            curr_start = None
            curr_end = None

    if curr_texts:
        clean_comb = _clean_segment_text(" ".join(curr_texts))
        if clean_comb and not _is_error_response(clean_comb):
            blocks.append({
                "text": clean_comb,
                "start": round(curr_start, 2),
                "duration": round(curr_end - curr_start, 2),
            })

    return blocks


def _translate_segments(
    raw_segments: list,
    source_language: str,
) -> list:
    """
    Translate non-English transcript segments to English using a multi-layer strategy:
      1. Primary: Fast Google Translator (deep-translator) with strict error-response rejection.
      2. Secondary Fallback: Gemini LLM translator (fluent technical English for Hinglish/Hindi).
      3. Tertiary Fallback: Original language text (sanitized).
    """
    blocks = _group_segments_for_translation(raw_segments)
    if not blocks:
        return []

    print(f"[transcript] Pre-grouped {len(raw_segments)} snippets into {len(blocks)} speech blocks for translation.")

    # ── Strategy A: Fast Google Translator (with error validation) ───────────
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='en')
        translated_blocks = []
        group_size = 5
        google_failed = False

        for g_start in range(0, len(blocks), group_size):
            group = blocks[g_start : g_start + group_size]
            combined_text = "\n".join(b["text"] for b in group)

            try:
                translated_comb = translator.translate(combined_text)
                if _is_bad_translation_output(translated_comb):
                    print("[transcript] [WARN] GoogleTranslator returned bad output. Escalating to Gemini...")
                    google_failed = True
                    break

                trans_lines = [l.strip() for l in translated_comb.split("\n") if l.strip()]
                if len(trans_lines) < len(group) * 0.6:
                    print(f"[transcript] [WARN] GoogleTranslator line count mismatch. Escalating to Gemini...")
                    google_failed = True
                    break

                for idx, b in enumerate(group):
                    t_txt = trans_lines[idx] if idx < len(trans_lines) else b["text"]
                    if _is_bad_translation_output(t_txt):
                        google_failed = True
                        break
                    translated_blocks.append({
                        "text": _clean_segment_text(t_txt),
                        "start": b["start"],
                        "duration": b["duration"],
                    })

                if google_failed:
                    break

            except Exception as ge:
                print(f"[transcript] [WARN] GoogleTranslator group failed ({ge}). Escalating to Gemini...")
                google_failed = True
                break

        if not google_failed and len(translated_blocks) == len(blocks):
            print(f"[transcript] [OK] Successfully translated {len(translated_blocks)} speech blocks via Google Translator.")
            return translated_blocks

    except Exception as err:
        print(f"[transcript] [WARN] Google Translator unavailable ({err}). Escalating to Gemini...")

    # ── Strategy B: Gemini Educational Translation Fallback ───────────────────
    return _translate_segments_with_gemini(blocks, source_language)




def _parse_pipe_translations(text: str, expected_count: int) -> dict:
    """
    Parse Gemini's pipe-delimited output.
    Returns a dict mapping 0-based index -> translated string.
    """
    import re
    lines = text.strip().split("\n")
    result = {}
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # STRATEGY 1: If we have EXACTLY expected_count lines, 
    # we can map them sequentially and just strip any leading numbers.
    # This is 100% robust against formatting errors as long as it didn't merge lines.
    if len(non_empty_lines) == expected_count:
        for i, line in enumerate(non_empty_lines):
            clean_line = re.sub(r"^[*_#]+", "", line).strip()
            clean_line = re.sub(r"^\d+\s*[|.:)\-]\s*", "", clean_line).strip()
            clean_line = re.sub(r"[*_]+$", "", clean_line).strip()
            if not _is_bad_translation_output(clean_line):
                result[i] = clean_line
        return result
        
    # STRATEGY 2: If the count mismatched (e.g. it added a preamble or merged lines),
    # try to use regex to pick out the numbered lines.
    for line in non_empty_lines:
        clean_line = re.sub(r"^[*_#]+", "", line).strip()
        match = re.match(r"^(\d+)\s*[|.:)\-]\s*(.+)$", clean_line)
        if match:
            idx = int(match.group(1)) - 1
            res_txt = match.group(2).strip()
            res_txt = re.sub(r"[*_]+$", "", res_txt).strip()
            if not _is_bad_translation_output(res_txt):
                result[idx] = res_txt

    return result

