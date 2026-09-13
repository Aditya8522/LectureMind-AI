"""
app/core/llm.py
────────────────
Gemini API wrapper + prompt engineering for Q&A, Smart Notes, and Quiz.

Uses the NEW google-genai SDK (google.genai), v1beta API.

VERIFIED AVAILABLE MODELS (queried live from API, Sep 2026):
  Working with .text:
    gemini-3.5-flash-lite   → fast, non-thinking, strict format-following
    gemini-3.7-flash        → thinking model, high quality
    gemini-3.8-flash        → thinking model, highest quality
    gemini-3.1-flash-lite   → fast, lightweight, non-thinking
    gemini-flash-lite-latest → alias for latest lite model

  Thinking models (return None for .text — need parts extraction):
    gemini-3.5-flash        → thinking model
    gemini-3.6-flash        → thinking model (API recommends for 2.0-flash replacement)
    gemini-flash-latest     → thinking alias

  GONE (404 NOT_FOUND — do NOT use):
    gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-flash-8b, gemini-2.5-flash

KEY RULE for translation: ONLY use non-thinking models.
Thinking models output internal reasoning (gibberish) before the translation,
which completely breaks the pipe-format parser.
NON-THINKING = gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-flash-lite-latest
"""

import os
import json
import time
from types import SimpleNamespace
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from dotenv import load_dotenv

from app.core.chunking import format_timestamp


load_dotenv()

# ── Verified Model Lists — Live Tested Sep 2026, google-genai v1beta SDK ────────
#
# TEST RESULTS (queried live API and verified actual text output):
#   gemini-3.5-flash-lite   ✅ Full text output, fast, non-thinking, reliable
#   gemini-3.1-flash-lite   ✅ Full text output, fast, non-thinking, reliable
#   gemini-flash-lite-latest ✅ Alias — always latest lite (non-thinking)
#   gemini-3.5-flash        ⚠️ Thinking model — partial output only (truncated)
#   gemini-3.7-flash        ❌ Pure reasoning — zero text output
#   gemini-3.8-flash        ❌ Pure reasoning — zero text output
#   gemini-3.6-flash        ❌ Empty text output
#   gemini-2.0-flash / 1.5-flash / 1.5-flash-8b  ❌ 404 NOT_FOUND (discontinued)
#
# CONCLUSION: gemini-3.5-flash-lite is the primary model for ALL tasks.
# Non-thinking models produce reliable, complete, correctly formatted output.

# Notes: Best quality for generating comprehensive study notes
NOTES_MODELS = [
    "gemini-3.5-flash-lite",    # PRIMARY: reliable, complete output, fast
    "gemini-3.1-flash-lite",    # FALLBACK: also reliable, slightly older
    "gemini-flash-lite-latest", # ALIAS: always latest lite model
]

# Quiz: Needs clean JSON — non-thinking models are best for structured output
QUIZ_MODELS = [
    "gemini-3.5-flash-lite",    # PRIMARY: strict format-following, clean JSON
    "gemini-3.1-flash-lite",    # FALLBACK
    "gemini-flash-lite-latest", # ALIAS FALLBACK
]

# Chat: Fast Q&A responses
CHAT_MODELS = [
    "gemini-3.5-flash-lite",    # PRIMARY: instant responses, non-thinking
    "gemini-3.1-flash-lite",    # FALLBACK
    "gemini-flash-lite-latest", # ALIAS FALLBACK
]

# Translation: Non-thinking ONLY — thinking models break the numbered-pipe parser
TRANSLATION_MODELS = [
    "gemini-3.5-flash-lite",    # PRIMARY: verified working, reliable pipe format
    "gemini-3.1-flash-lite",    # FALLBACK
    "gemini-flash-lite-latest", # ALIAS FALLBACK
]

# ── Output token budgets ────────────────────────────────────────────────────────
NOTES_OUTPUT_TOKENS_SUMMARY   = 6144   # Summary: concise
NOTES_OUTPUT_TOKENS_SHORT     = 12288  # Detailed, video <= 75 min
NOTES_OUTPUT_TOKENS_PART      = 8192   # Detailed, per-part of long video
NOTES_OUTPUT_TOKENS_SYNTHESIS = 4096   # Final synthesis tables
CHAT_OUTPUT_TOKENS            = 1024   # Chat: concise grounded answers
QUIZ_OUTPUT_TOKENS            = 4096   # Quiz: JSON array


_client_primary = None    # GEMINI_API_KEY  (Key 1)
_client_secondary = None  # GEMINI_API_KEY_2 (Key 2)


def _get_clients_ordered(preferred: str = "primary") -> list:
    """Return list of (key_name, client) ordered by task preference."""
    global _client_primary, _client_secondary
    key1 = os.getenv("GEMINI_API_KEY")
    key2 = os.getenv("GEMINI_API_KEY_2")

    if not key1 and not key2:
        raise ValueError("No GEMINI_API_KEY found in .env file.")

    if _client_primary is None and key1:
        _client_primary = genai.Client(api_key=key1)
    if _client_secondary is None and key2:
        _client_secondary = genai.Client(api_key=key2)

    clients = []
    if preferred == "primary":
        if _client_primary:
            clients.append(("Primary Key", _client_primary))
        if _client_secondary:
            clients.append(("Secondary Key", _client_secondary))
    else:
        if _client_secondary:
            clients.append(("Secondary Key", _client_secondary))
        if _client_primary:
            clients.append(("Primary Key", _client_primary))

    return clients


def _has_two_keys() -> bool:
    """Return True if both GEMINI_API_KEY and GEMINI_API_KEY_2 are configured."""
    return bool(os.getenv("GEMINI_API_KEY")) and bool(os.getenv("GEMINI_API_KEY_2"))


def _extract_response_text(response) -> str:
    """
    Safely extract text from a Gemini response.

    Some models (thinking models like gemini-3.5-flash, 3.6-flash) return
    response.text = None because their output comes through 'parts' with
    separate thought and text parts. This function handles both cases.
    """
    # Try .text first (works for non-thinking models)
    if response.text:
        return response.text

    # For thinking models: iterate parts, collect only text parts (skip thought parts)
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                text_parts = []
                for part in candidate.content.parts:
                    # Skip thought parts (internal reasoning)
                    if getattr(part, 'thought', False):
                        continue
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return "".join(text_parts)

    return ""


def _call_gemini_with_fallback(
    contents,
    config=None,
    model_candidates: List[str] = None,
    preferred_key: str = "primary",
):
    """
    Call Gemini API with MODEL-FIRST fallback across all available keys.

    Tries each model on all keys before downgrading to the next model.
    Handles both thinking and non-thinking models via _extract_response_text.
    On 429 rate-limit: waits 3s before next attempt.
    """
    candidates = model_candidates or NOTES_MODELS
    client_entries = _get_clients_ordered(preferred=preferred_key)
    last_err = None

    for model_name in candidates:
        for key_label, client in client_entries:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                # response.text is read-only — wrap in SimpleNamespace so all callers
                # can uniformly use result.text regardless of model type.
                text = _extract_response_text(response)
                return SimpleNamespace(text=text)
            except Exception as e:
                err_str = str(e)
                print(f"[llm] [WARN] [{key_label}] '{model_name}' failed: {e}. Trying next...")
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(3)
                last_err = e

    raise RuntimeError(f"All Gemini models and API keys exhausted. Last error: {str(last_err)}")







# ── Prompt Building ────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a helpful study assistant for students watching YouTube lectures.
Your job is to answer questions STRICTLY based on the transcript excerpts provided below.

IMPORTANT RULES:
1. Only use information from the provided transcript context. Do NOT use outside knowledge.
2. If the answer is not in the provided context, say clearly: "I couldn't find information about this in the video transcript. Try asking about a different topic covered in the lecture."
3. When referencing specific information, mention the timestamp where it appears (e.g., "At [2:05]...").
4. Be concise but complete. Use bullet points for lists.
5. If multiple chunks contain relevant info, synthesize them into a coherent answer."""


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[dict],
    video_title: str = "the lecture",
) -> str:
    """
    Build the complete RAG prompt: system instruction + context + question.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        start_ts = format_timestamp(chunk["start_time"])
        end_ts = format_timestamp(chunk["end_time"])
        context_blocks.append(
            f"[Context {i+1}] [{start_ts} - {end_ts}]\n{chunk['text']}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_INSTRUCTION}

Video: {video_title}

--- TRANSCRIPT CONTEXT ---
{context_text}
--- END CONTEXT ---

Student Question: {question}

Answer (cite timestamps like [2:05] when referencing specific parts):"""

    return prompt


# ── LLM Call ──────────────────────────────────────────────────────────────────

def ask_gemini(
    question: str,
    retrieved_chunks: List[dict],
    video_id: str,
    video_title: str = "the lecture",
) -> dict:
    """
    Main Q&A function: retrieved chunks + question -> answer + citations.

    Returns:
      {
        "answer"           : str,
        "cited_timestamps" : list of timestamp dicts for frontend,
        "raw_chunks_used"  : list of chunk dicts passed to the LLM,
      }
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find any relevant information in the transcript. Please make sure the video has been processed.",
            "cited_timestamps": [],
            "raw_chunks_used": [],
        }

    prompt = build_rag_prompt(question, retrieved_chunks, video_title)

    try:
        response = _call_gemini_with_fallback(
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=CHAT_OUTPUT_TOKENS,
                top_p=0.8,
            ),
            model_candidates=CHAT_MODELS,
            preferred_key="secondary",
        )
        answer_text = response.text.strip()


    except Exception as e:
        raise RuntimeError(
            f"Gemini API call failed: {str(e)}. "
            "Check your API key and network connection."
        )

    cited_timestamps = _build_cited_timestamps(retrieved_chunks, video_id)

    return {
        "answer": answer_text,
        "cited_timestamps": cited_timestamps,
        "raw_chunks_used": retrieved_chunks,
    }


# ── Timestamp Formatting ───────────────────────────────────────────────────────

def _build_cited_timestamps(chunks: List[dict], video_id: str) -> List[dict]:
    """
    Convert retrieved chunks into timestamp citation dicts for the frontend.
    Each dict has: start, end, label, url.
    """
    citations = []
    for chunk in chunks:
        start = chunk["start_time"]
        end = chunk["end_time"]
        citations.append({
            "start": start,
            "end": end,
            "label": f"{format_timestamp(start)} - {format_timestamp(end)}",
            "url": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
        })
    return citations


# -- Phase 2: Notes Generation (Duration-Scaled & Multi-Part for 2+ Hour Lectures) --

# ── Phase 2: Notes Generation (ThetaWave Publication-Grade Architecture) ─────

NOTES_SUMMARY_PROMPT = """You are an expert academic note-taker. Your job is to produce a concise, high-retention Quick Summary from the lecture transcript provided below.

---
LECTURE TITLE: {title}
DURATION: {duration_str}
---

STEP 1 — UNDERSTAND THE CONTENT FIRST:
Before writing anything, internally read and understand the full transcript. Identify:
- What is the central topic or problem this lecture addresses?
- What are the 4–8 most important concepts, terms, or ideas discussed?
- What is the logical flow (beginning → middle → end)?
- Are there any comparisons, trade-offs, algorithms, formulas, or code discussed?

STEP 2 — WRITE THE SUMMARY using the rules below:

RULES (follow strictly):
✅ Base EVERY sentence strictly on what is said in the transcript. No invented facts.
✅ Capture the instructor's own words, examples, analogies, and warnings where possible.
✅ Use clear, concise language. No padding, no repetition.
✅ Include timestamps [M:SS] next to topics when the transcript provides them.
❌ Do NOT add complexity tables, formulas, or code unless the instructor explicitly discusses them.
❌ Do NOT add generic study advice or textbook content not present in the transcript.
❌ Do NOT skip any major concept discussed — even if briefly mentioned.

OUTPUT FORMAT (use this exact Markdown structure):

# 📝 {title}
> ⏱️ **Duration:** {duration_str} | **Quick Summary**

## 🎯 What This Lecture Is About
Write 2–3 clear paragraphs: What is the core topic? Why does it matter? What will the student understand after watching this?

---

## 🔑 Key Concepts & Terms
| Concept / Term | What It Means (in the context of this lecture) |
|---|---|
(List every important concept or term the instructor defines or explains. Use the instructor's own explanation, not a dictionary definition.)

---

## 🗺️ Lecture Walkthrough
List the major topics in the order the instructor covers them. For each, write 1–3 bullet points summarizing what was said.

1. **[Topic Name]** — [What the instructor explained about it]
2. **[Topic Name]** — [What the instructor explained about it]
(Continue for every major segment of the lecture)

---

(ONLY include this section if the instructor explicitly compared multiple approaches or methods:)
## ⚖️ Comparison / Trade-offs
| Approach | How It Works | Advantage | Limitation |
|---|---|---|---|
(One row per approach the instructor compared)

---

(ONLY include this section if algorithms, Big-O complexity, or formulas were discussed:)
## ⏲️ Complexity / Formulas
| Item | Detail |
|---|---|
(Capture exactly what the instructor said)

---

## 💡 Key Takeaways
(3–6 bullet points. Each one should be a complete, actionable insight a student can remember. Based only on the transcript.)
- **[Takeaway]**: ...

---

## 📚 Instructor Notes & Next Steps
(Only if mentioned: prerequisites, related lectures, homework, GitHub links, playlist pointers)

---

TRANSCRIPT:
{context}

NOW GENERATE THE QUICK SUMMARY:"""


NOTES_DETAILED_PROMPT_SINGLE = """You are an expert academic note-taker and educator. Your task is to produce a comprehensive, publication-grade Master Study Guide from the lecture transcript below. These notes will be used by students for deep study and revision.

---
LECTURE TITLE: {title}
DURATION: {duration_str}
---

STEP 1 — ANALYZE THE TRANSCRIPT FIRST:
Read the entire transcript carefully. Before writing, identify:
- What TYPE of lecture is this? (e.g., conceptual theory, coding tutorial, mathematical derivation, system design, history/story, comparison of tools)
- What are ALL the topics, subtopics, examples, analogies, demonstrations, and warnings the instructor covers?
- What is the chronological flow of the lecture?
- Does the instructor write code? Show formulas? Draw diagrams? Give real-world analogies?

STEP 2 — WRITE THE NOTES using the rules below:

RULES (follow strictly):
✅ Cover EVERY concept, subtopic, example, analogy, warning, and tip the instructor mentions — no gaps.
✅ Write in the order the instructor presents the content. Follow the lecture's natural flow.
✅ Use the instructor's own words, examples, and analogies where they are particularly clear or memorable.
✅ Write section headings that reflect the ACTUAL topic being discussed (not generic placeholders).
✅ Include approximate timestamps [M:SS] for each major section using the timestamps in the transcript.
✅ For each concept: explain what it is, why it matters, and how it works — all based on what the instructor said.
✅ Write bullet points that are complete thoughts (not single vague words).
✅ Use a Markdown table ONLY when the instructor explicitly compares multiple things side-by-side.
✅ Include a code block ONLY if the instructor actually writes or dictates code. Use the exact code discussed.
✅ Include formulas/math ONLY if the instructor explicitly states them. Write them in LaTeX ($$...$$).
✅ Include an ASCII diagram ONLY if the instructor describes a pipeline, architecture, or flow structure.
❌ Do NOT invent code, formulas, or diagrams that are not in the transcript.
❌ Do NOT add outside knowledge, textbook content, or examples the instructor did not mention.
❌ Do NOT skip any topic, even if it seems minor — a brief mention still deserves a bullet point.
❌ Do NOT use generic placeholder headings like "Topic 1" — use the real topic name.

OUTPUT FORMAT:

# 📚 {title}
> ⏱️ **Duration:** {duration_str} | **Master Study Guide**

**Overview:**
(2–3 paragraphs: What is the full scope of this lecture? What problems does it address? What will the student be able to do or understand after studying these notes? Write this based purely on the transcript.)

---

(Now write one ## section per major topic the instructor covers. Title each section using the actual topic name and approximate timestamp.)

## [Emoji] [Actual Topic Name from Transcript] [[M:SS]]

> **Core Idea:** (One sentence — the single most important thing the instructor says about this topic.)

(3–8 bullet points covering everything the instructor explains about this topic: definitions, intuition, motivation, how it works, edge cases, warnings, analogies.)

(If the instructor compares approaches — add a table here.)
(If the instructor writes code — add the exact code block here.)
(If the instructor states a formula — add LaTeX here.)
(If the instructor describes a flow/architecture — add ASCII diagram here.)

---

(Repeat the ## section block for EVERY topic the instructor discusses. Do not stop early.)

---

## 🔑 Key Concepts & Terminology Glossary
| Term | Definition & Role in This Lecture |
|---|---|
(List every important term or concept introduced, with an explanation grounded in the transcript)

---

## 💡 Important Tips, Warnings & Instructor Insights
(Capture every practical tip, common mistake warning, rule of thumb, or "pro tip" the instructor mentions)
- **[Tip/Warning]**: ...

---

## 📚 Prerequisites & What's Next
(Only if the instructor mentions them: prior knowledge required, related lectures, upcoming topics, homework, resources)

---

TRANSCRIPT:
{context}

NOW GENERATE THE COMPLETE MASTER STUDY GUIDE. Do not stop until every topic in the transcript has been covered:"""


NOTES_DETAILED_PART_PROMPT = """You are an expert academic note-taker writing Part {part_num} of {total_parts} of a Master Study Guide for the lecture: "{title}".
Duration: {duration_str} | This part covers: [{start_ts}] → [{end_ts}]

STEP 1 — READ THIS CHUNK CAREFULLY:
You are given the transcript for timestamps [{start_ts}] to [{end_ts}] only. Before writing, identify all the concepts, examples, code, comparisons, formulas, and instructor remarks in this window.

STEP 2 — WRITE NOTES for this time window using these rules:

RULES:
✅ Cover EVERY topic, subtopic, example, analogy, and warning the instructor mentions in this time window.
✅ Write section headings using the ACTUAL topic names from the transcript, not generic placeholders.
✅ Include timestamps [M:SS] for each section.
✅ For each concept: explain what it is, why it matters, and how it works — all from the transcript.
✅ Write complete, informative bullet points (not single words).
✅ Use a table ONLY if the instructor explicitly compares multiple items.
✅ Add a code block ONLY if the instructor writes or dictates actual code.
✅ Add formulas in LaTeX ($$...$$) ONLY if the instructor explicitly states them.
✅ Add an ASCII diagram ONLY if the instructor describes a pipeline or flow structure.
❌ Do NOT invent content not in this transcript chunk.
❌ Do NOT output the main document title (# heading) — this is a chapter, not the full document.
❌ Do NOT add a concluding summary — that will be handled in a separate synthesis step.

FORMAT — Write one section per topic in this time window:

## [Emoji] [Actual Topic Name] [[M:SS]]

> **Core Idea:** (One sentence — the most important thing the instructor says here.)

(3–8 bullet points covering everything the instructor explains: definitions, intuition, motivation, how it works, edge cases, analogies, warnings.)

(Conditionally: table / code block / formula / ASCII diagram — only if present in the transcript)

---

(Repeat for every topic in this time window [{start_ts}] to [{end_ts}]. Do not stop early.)

TRANSCRIPT FOR THIS PART [{start_ts}] → [{end_ts}]:
{context}

NOW GENERATE PART {part_num} NOTES. Cover every topic in this time window:"""


NOTES_DETAILED_SYNTHESIS_PROMPT = """You are an expert academic educator writing the final synthesis section of a Master Study Guide for: "{title}" (Duration: {duration_str}).

You have been given summaries of all lecture parts below. Your job is to write a concise, high-value synthesis that ties everything together.

RULES:
✅ Base everything strictly on what is in the part summaries below.
✅ Only include sections that are genuinely relevant to this specific lecture's content.
✅ Be concise and information-dense — no repetition, no padding.
❌ Do NOT repeat content already covered in the chapter notes.
❌ Do NOT include code snippets, formulas, or diagrams unless they are essential for a quick-reference summary.
❌ Do NOT invent content not present in the summaries.

WRITE ONLY THE SECTIONS THAT APPLY to this specific lecture:

## 🔑 Complete Key Concepts Glossary
| Term | What It Means in This Lecture |
|---|---|
(All important terms from the entire lecture — grounded in what the instructor said)

---

(ONLY IF code was discussed in the lecture:)
## 🧮 Code Quick-Reference
| Pattern / Approach | Key Syntax or Logic |
|---|---|

---

(ONLY IF comparisons or multiple approaches were discussed:)
## ⚖️ Full Comparison Table
| Item | Details |
|---|---|

---

## 💡 Master Takeaways
(5–8 bullet points — the most important things a student should remember from this entire lecture)
- **[Insight]**: ...

---

## 🔧 All Tips, Warnings & Instructor Insights
(Every practical tip, common mistake warning, or "pro tip" mentioned across the full lecture)
- **[Tip/Warning]**: ...

---

## 📚 Prerequisites, Roadmap & Resources
(Only if mentioned by the instructor: prior knowledge, related lectures, next topics, links, homework)

---

LECTURE PART SUMMARIES:
{context_summary}

NOW GENERATE THE SYNTHESIS SECTION:"""


def _partition_chunks_by_duration(sorted_chunks: List[dict], target_segment_duration_sec: float = 3000.0) -> List[List[dict]]:
    """
    Partition chunks into logical time segments for multi-part generation.
    - If total duration <= 75 mins (4500s) -> 1 single part.
    - If total duration > 75 mins (e.g. 2hr, 3hr+) -> 2 to 4 parts (~45-60 mins each).
    """
    if not sorted_chunks:
        return []

    total_duration = max(1.0, sorted_chunks[-1]["end_time"] - sorted_chunks[0]["start_time"])
    if total_duration <= 4500.0:  # <= 75 minutes
        return [sorted_chunks]

    num_parts = max(2, min(5, round(total_duration / target_segment_duration_sec)))
    part_duration = total_duration / num_parts

    parts = []
    current_part = []
    current_boundary = sorted_chunks[0]["start_time"] + part_duration

    for c in sorted_chunks:
        if c["start_time"] >= current_boundary and len(parts) < num_parts - 1 and current_part:
            parts.append(current_part)
            current_part = [c]
            current_boundary += part_duration
        else:
            current_part.append(c)

    if current_part:
        parts.append(current_part)

    return parts


def generate_notes(chunks: List[dict], video_title: str, mode: str = "summary") -> str:
    """
    Generate structured study notes from transcript chunks.

    Strategy:
      - Summary mode:   Always single-pass (fast, concise).
      - Detailed mode, video <= 75 min:  Single-pass with high token budget.
      - Detailed mode, video > 75 min:   Multi-part generation.
          * If 2 API keys configured → parallel (fast, uses both keys).
          * If only 1 API key        → sequential (safe, avoids 429 rate limits).
        Each part gets its own full LLM output budget.
        Final synthesis ties everything together.

    Token budgets are tuned for gemini-2.5-flash (65K out) with graceful
    fallback to 2.0-flash / 1.5-flash (8K out). The fallback models will
    truncate if the requested tokens exceed their limit, but the response
    is still valid — just shorter.
    """
    if not chunks:
        return "# No Content\nNo transcript content available to generate notes from."

    sorted_chunks = sorted(chunks, key=lambda c: c["start_time"])
    start_sec = sorted_chunks[0]["start_time"]
    end_sec = sorted_chunks[-1]["end_time"]
    total_duration_sec = max(1.0, end_sec - start_sec)
    duration_str = format_timestamp(total_duration_sec)
    duration_minutes = total_duration_sec / 60.0

    print(f"[notes] Generating '{mode}' notes for '{video_title}' ({duration_str}, {len(sorted_chunks)} chunks)")

    def _build_context(chunk_list):
        """Format chunks into a clean timestamped transcript string."""
        return "\n\n".join(
            f"[{format_timestamp(c['start_time'])}] {c['text']}"
            for c in chunk_list
        )

    # ── 1. Summary Mode ──────────────────────────────────────────────────────────
    if mode == "summary":
        context = _build_context(sorted_chunks)
        prompt = NOTES_SUMMARY_PROMPT.format(
            title=video_title,
            duration_str=duration_str,
            context=context,
        )
        try:
            response = _call_gemini_with_fallback(
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=NOTES_OUTPUT_TOKENS_SUMMARY,
                    top_p=0.9,
                ),
                model_candidates=NOTES_MODELS,
                preferred_key="primary",
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Summary notes generation failed: {str(e)}")

    # ── 2. Detailed Mode: decide single-pass vs multi-part ───────────────────────
    parts = _partition_chunks_by_duration(sorted_chunks, target_segment_duration_sec=3000.0)

    # ── 2A. Short video (<= 75 min) → Single-pass ────────────────────────────────
    if len(parts) <= 1:
        context = _build_context(sorted_chunks)
        prompt = NOTES_DETAILED_PROMPT_SINGLE.format(
            title=video_title,
            duration_str=duration_str,
            context=context,
        )
        try:
            response = _call_gemini_with_fallback(
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=NOTES_OUTPUT_TOKENS_SHORT,
                    top_p=0.9,
                ),
                model_candidates=NOTES_MODELS,
                preferred_key="primary",
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Detailed notes generation failed: {str(e)}")

    # ── 2B. Long video (> 75 min) → Multi-part generation ────────────────────────
    two_keys = _has_two_keys()
    mode_label = "Parallel (2 keys)" if two_keys else "Sequential (1 key — safe mode)"
    print(f"[notes] Long lecture ({duration_str}, {len(parts)} parts). Mode: {mode_label}")

    def _generate_part(idx, part_chunks, total_parts):
        part_num = idx + 1
        part_start_ts = format_timestamp(part_chunks[0]["start_time"])
        part_end_ts   = format_timestamp(part_chunks[-1]["end_time"])
        part_context  = _build_context(part_chunks)

        part_prompt = NOTES_DETAILED_PART_PROMPT.format(
            part_num=part_num,
            total_parts=total_parts,
            title=video_title,
            duration_str=duration_str,
            start_ts=part_start_ts,
            end_ts=part_end_ts,
            context=part_context,
        )

        # Alternate keys to spread load: even parts → primary, odd parts → secondary
        assigned_key = "primary" if (idx % 2 == 0) else "secondary"
        print(f"[notes] Part {part_num}/{total_parts} [{part_start_ts}→{part_end_ts}] via {assigned_key}...")

        resp = _call_gemini_with_fallback(
            contents=part_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=NOTES_OUTPUT_TOKENS_PART,
                top_p=0.9,
            ),
            model_candidates=NOTES_MODELS,
            preferred_key=assigned_key,
        )
        return (idx, resp.text.strip())

    generated_parts_content = []
    try:
        if two_keys:
            # Parallel: safe because each part targets a different key
            max_workers = min(len(parts), 2)  # max 2 — one per key
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_generate_part, idx, p, len(parts))
                    for idx, p in enumerate(parts)
                ]
                results = [f.result() for f in futures]
        else:
            # Sequential: single key — avoid hammering RPM limits
            results = []
            for idx, p in enumerate(parts):
                results.append(_generate_part(idx, p, len(parts)))
                # Small courtesy delay between sequential calls
                if idx < len(parts) - 1:
                    time.sleep(2)

        results.sort(key=lambda r: r[0])
        generated_parts_content = [r[1] for r in results]
        print(f"[notes] All {len(parts)} parts generated successfully.")

    except Exception as e:
        print(f"[notes] [WARN] Multi-part generation failed: {e}. Falling back to single-pass.")
        return _generate_notes_single_pass_fallback(sorted_chunks, video_title, duration_str)

    # ── 2C. Synthesis — tie all parts together ────────────────────────────────────
    # Use the FULL generated text of each part (not just 8 sample chunks).
    # This gives the synthesis model the real content to summarise from.
    synthesis_context_parts = []
    for idx, (part_chunks, part_text) in enumerate(zip(parts, generated_parts_content)):
        p_start = format_timestamp(part_chunks[0]["start_time"])
        p_end   = format_timestamp(part_chunks[-1]["end_time"])
        # Trim each part to ~1500 chars to fit synthesis context budget
        trimmed = part_text[:1500] + ("..." if len(part_text) > 1500 else "")
        synthesis_context_parts.append(f"=== Part {idx+1} [{p_start} → {p_end}] ===\n{trimmed}")

    synthesis_prompt = NOTES_DETAILED_SYNTHESIS_PROMPT.format(
        title=video_title,
        duration_str=duration_str,
        context_summary="\n\n".join(synthesis_context_parts),
    )

    try:
        print("[notes] Generating synthesis (key concepts, takeaways, tips)...")
        syn_resp = _call_gemini_with_fallback(
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=NOTES_OUTPUT_TOKENS_SYNTHESIS,
                top_p=0.9,
            ),
            model_candidates=NOTES_MODELS,
            preferred_key="primary",
        )
        synthesis_content = syn_resp.text.strip()
    except Exception as e:
        print(f"[notes] [WARN] Synthesis generation failed: {e}. Skipping synthesis.")
        synthesis_content = ""

    # ── Assemble final document ───────────────────────────────────────────────────
    header = (
        f"# 📚 {video_title}\n\n"
        f"> ⏱️ **Total Duration:** {duration_str} | "
        f"**Master Study Guide ({len(parts)} Parts)**\n\n"
        f"---\n"
    )
    parts_text   = "\n\n---\n\n".join(generated_parts_content)
    final_doc    = f"{header}\n{parts_text}"
    if synthesis_content:
        final_doc += f"\n\n---\n\n{synthesis_content}"

    print(f"[notes] Final document: {len(final_doc)} chars, ~{len(final_doc.split())} words.")
    return final_doc.strip()


def _generate_notes_single_pass_fallback(sorted_chunks: List[dict], video_title: str, duration_str: str) -> str:
    """Fallback: generate notes in a single pass when multi-part fails."""
    context = "\n\n".join(
        f"[{format_timestamp(c['start_time'])}] {c['text']}"
        for c in sorted_chunks
    )
    prompt = NOTES_DETAILED_PROMPT_SINGLE.format(
        title=video_title,
        duration_str=duration_str,
        context=context,
    )
    response = _call_gemini_with_fallback(
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=NOTES_OUTPUT_TOKENS_SHORT,
            top_p=0.9,
        ),
        model_candidates=NOTES_MODELS,
        preferred_key="primary",
    )
    return response.text.strip()



# ── Phase 2: Quiz Generation ──────────────────────────────────────────────────

QUIZ_GENERATION_PROMPT = """You are an expert educator creating a multiple-choice quiz from a lecture transcript.

Generate exactly {num_questions} multiple-choice questions based on the transcript below.

RESPOND WITH ONLY VALID JSON — NO MARKDOWN, NO EXPLANATION, NO CODE BLOCK — just raw JSON.

The JSON must be an array of objects with this EXACT structure:
[
  {{
    "id": 1,
    "question": "What is ...?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 0,
    "explanation": "The correct answer is A because... (cite timestamp like [2:05])",
    "timestamp": 125.0
  }}
]

RULES:
- correct_index is 0-based (0=A, 1=B, 2=C, 3=D)
- All 4 options must be plausible (no obviously wrong answers)
- Questions must test real understanding, not trivial recall
- Include a mix of concept, application, and definition questions
- explanation must cite the timestamp in the transcript where the answer is found
- timestamp must be the float seconds value from the transcript context

VIDEO TITLE: {title}

TRANSCRIPT CONTEXT:
{context}

OUTPUT ONLY THE JSON ARRAY:"""


def generate_quiz(chunks: List[dict], video_title: str, num_questions: int = 5) -> List[dict]:
    """
    Generate multiple-choice quiz questions from transcript chunks.

    Returns:
        list of question dicts with: id, question, options, correct_index, explanation, timestamp
    """
    if not chunks:
        return []

    sorted_chunks = sorted(chunks, key=lambda c: c["start_time"])
    context_parts = []
    for chunk in sorted_chunks:
        ts = format_timestamp(chunk["start_time"])
        context_parts.append(f"[{ts} / {chunk['start_time']:.1f}s] {chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt = QUIZ_GENERATION_PROMPT.format(
        num_questions=num_questions,
        title=video_title,
        context=context,
    )

    try:
        response = _call_gemini_with_fallback(
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=QUIZ_OUTPUT_TOKENS,
                top_p=0.9,
                response_mime_type="application/json",
            ),
            model_candidates=QUIZ_MODELS,
            preferred_key="secondary",
        )
        raw = response.text.strip()




        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        questions = json.loads(raw)

        # Validate structure
        validated = []
        for i, q in enumerate(questions):
            if all(k in q for k in ("question", "options", "correct_index")):
                q["id"] = i + 1
                q.setdefault("explanation", "See the video for details.")
                q.setdefault("timestamp", 0.0)
                validated.append(q)

        return validated

    except Exception as e:
        raise RuntimeError(f"Quiz generation failed: {str(e)}")


def grade_quiz(questions: List[dict], user_answers: dict) -> dict:
    """
    Grade a submitted quiz locally (no LLM call needed — answer is in the question dict).

    Args:
        questions:    list of question dicts (with correct_index, explanation, timestamp)
        user_answers: dict mapping str(question_id) -> selected_index (int)

    Returns:
        {
          "score": float (0.0 – 1.0),
          "correct_count": int,
          "total": int,
          "per_question": list of per-question result dicts
        }
    """
    total = len(questions)
    correct_count = 0
    per_question = []

    for q in questions:
        qid = str(q["id"])
        selected = user_answers.get(qid)
        is_correct = selected is not None and int(selected) == int(q["correct_index"])
        if is_correct:
            correct_count += 1

        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "selected_index": selected,
            "correct_index": q["correct_index"],
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
            "timestamp": q.get("timestamp", 0.0),
            "options": q["options"],
        })

    score = correct_count / total if total > 0 else 0.0
    return {
        "score": score,
        "correct_count": correct_count,
        "total": total,
        "per_question": per_question,
    }
