"""
app/core/llm.py
────────────────
Gemini API wrapper + prompt engineering for Q&A, Smart Notes, and Quiz.

Uses the NEW google-genai SDK (google.genai), not the deprecated
google-generativeai package.

Features:
  - Multi-model fallback chain (gemini-3.5-flash -> gemini-3.7-flash -> gemini-3.6-flash)
  - Automatic resilience against 503 high demand or 429 quota exhaustion
  - Grounded RAG Q&A with exact timestamp citations
  - Smart Notes generation (summary & detailed modes)
  - Interactive MCQ Quiz generation with validated JSON output & local grading
"""

import os
import json
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv

from app.core.chunking import format_timestamp

load_dotenv()

# ── Task-Specific Model Configurations ─────────────────────────────────────────
# Based on Google AI Studio rate limits and model reasoning capabilities:

# Smart Notes: Flagship reasoning model for deepest synthesis, structure, and academic clarity
NOTES_MODELS = [
    "gemini-3.7-flash",      # Flagship intelligence for rich, high-quality notes
    "gemini-3.5-flash",      # High-capability fallback
    "gemini-3.5-flash-lite", # Ultra-fast fallback
]

# Practice Quiz: Strong reasoning for plausible distractors and timestamp-grounded explanations
QUIZ_MODELS = [
    "gemini-3.5-flash",      # Fast, highly accurate structured JSON generation
    "gemini-3.7-flash",      # Deep reasoning fallback
    "gemini-3.5-flash-lite", # Lightweight fallback
]

# AI Tutor Chat: Fast Q&A with huge daily quota (500 RPD on Flash Lite) for extended study sessions
CHAT_MODELS = [
    "gemini-3.5-flash",      # High-accuracy direct answers
    "gemini-3.5-flash-lite", # 500 RPD high-throughput study chat
    "gemini-3.1-flash-lite", # Secondary high-throughput tier (500 RPD)
    "gemini-3.7-flash",      # Flagship fallback
]

_client_primary = None    # GEMINI_API_KEY (Key 1: Dedicated to Smart Notes)
_client_secondary = None  # GEMINI_API_KEY_2 (Key 2: Dedicated to Chat & Quiz)


def _get_clients_ordered(preferred: str = "primary") -> list:
    """
    Return list of (key_name, client) ordered by task preference.
      - preferred = 'primary'   -> [Key 1 (Notes), Key 2 (Secondary)]
      - preferred = 'secondary' -> [Key 2 (Chat/Quiz), Key 1 (Primary)]
    """
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


def _call_gemini_with_fallback(
    contents,
    config=None,
    model_candidates: List[str] = None,
    preferred_key: str = "primary",
):
    """
    Call Gemini API using a MODEL-FIRST fallback strategy.

    For each model (best -> good -> lite), ALL available API keys are tried
    BEFORE downgrading to the next model. This ensures the best model
    (e.g. gemini-3.7-flash) is attempted on BOTH accounts before falling
    back to a lower-capability model.

    Example for Notes (preferred_key='primary', NOTES_MODELS):
      1. Key1 + gemini-3.7-flash   ← try best model on primary account
      2. Key2 + gemini-3.7-flash   ← try SAME best model on secondary account
      3. Key1 + gemini-3.5-flash   ← downgrade only if both keys fail on 3.7
      4. Key2 + gemini-3.5-flash
      5. Key1 + gemini-3.5-flash-lite
      6. Key2 + gemini-3.5-flash-lite
    """
    candidates = model_candidates or NOTES_MODELS
    client_entries = _get_clients_ordered(preferred=preferred_key)
    last_err = None

    # MODEL-FIRST: iterate models in outer loop, keys in inner loop
    for model_name in candidates:
        for key_label, client in client_entries:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return response
            except Exception as e:
                print(f"[llm] [WARN] [{key_label}] '{model_name}' failed: {e}. Trying next...")
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
                max_output_tokens=1024,
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

NOTES_SUMMARY_PROMPT = """You are an expert academic note-taker. Generate an executive study summary from the lecture transcript below.
Total Lecture Duration: {duration_str}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS MARKDOWN STRUCTURE:
# 📝 Executive Summary: {title}
> ⏱️ **Duration:** {duration_str} | **High-Yield Overview & Concept Roadmap**

## 🎯 Core Lecture Thesis
2-3 clear, comprehensive paragraphs explaining the fundamental topic, why it matters, core objectives, and theoretical foundation.

## 🗺️ Lecture Roadmap & Milestones
- **Phase 1: Foundations** [0:00] — Brief outline of introductory topics and intuition.
- **Phase 2: Core Formulations & Derivations** — Key mathematical / algorithmic principles.
- **Phase 3: Implementation & Validation** — Practical coding and empirical results.
(Map out the entire timeline of the lecture with milestone timestamps [M:SS] or [H:MM:SS])

## 📌 Major Conceptual Pillars
- **[Concept Name]** [timestamp] — Clear, high-yield explanation. Include key LaTeX formulas ($...$ or $$...$$) where applicable.
- (Detail all major concepts covered across the full lecture duration)

## 🔑 Key Terms & Mathematical Definitions
| Term / Symbol | Definition / Formula | Significance / Use Case |
|---|---|---|
| (Include 6-12 core terms and mathematical equations formatted in LaTeX) |

## 💡 Master Takeaways
1. First critical takeaway
2. Second critical takeaway
(5-8 practical, high-value takeaways summarizing the entire lecture)

RULES:
- Maintain high academic clarity and structure.
- Format all mathematical equations in valid LaTeX ($formula$ for inline, $$formula$$ for display).
- Cover the entire timeline from the beginning to the end.

VIDEO TITLE: {title}

TRANSCRIPT CONTEXT:
{context}

GENERATE EXECUTIVE SUMMARY NOTES NOW:"""


NOTES_DETAILED_PROMPT_SINGLE = """You are an expert academic professor and technical author. Cover the ENTIRE lecture in exhaustive detail — every concept, proof, equation, algorithm, and example from minute 0 to the very last minute.

CRITICAL INSTRUCTIONS FOR DEPTH & COMPLETENESS:
1. EXHAUSTIVE DEPTH: Do NOT rush or over-summarize. For every topic, provide rich, thorough explanations (4-8 comprehensive bullet points) explaining intuition, mechanics, mathematical equations, and real-world implications.
2. MATHEMATICAL FORMULAS IN FULL LATEX:
   - Always format math variables and equations using standard LaTeX:
     - Inline variables: $y = mx + b$, $\\theta_j$, $\\alpha$, $\\bar{{x}}$
     - Standalone display equations: $$\\frac{{\\partial E}}{{\\partial b}} = -2 \\sum_{{i=1}}^{{n}} (y_i - mx_i - b) = 0$$
   - Show complete algebraic derivations step-by-step. NEVER skip intermediate proof steps.
3. CODE IMPLEMENTATIONS: When programming/code is discussed or shown in the lecture, write out the complete, well-commented code block.
4. CHAPTER HEADINGS WITH TIMESTAMPS: Every subtopic must have its own ## heading with a milestone timestamp (e.g. `## 🔍 Ordinary Least Squares (OLS) [05:45]`).

OUTPUT FORMAT (follow EXACTLY):

# 📚 {title}
> ⏱️ **Duration:** {duration_str} | **Comprehensive Master Study Guide**

Brief overview of the full lecture scope, theoretical objectives, and practical significance.

---

## [emoji] [Topic Title] [M:SS]

> **[Key Term / Concept]:** One-sentence formal definition of the concept introduced here.

- **[Point]** — detailed explanation covering intuition and mechanics.
- **[Point]** — detailed explanation.
- (Cover all important details, parameters, nuances, and edge cases from the transcript for this section)

[If explaining step-by-step logic, include numbered list:]
1. Step one
2. Step two

[If mathematical derivation is present, show complete LaTeX steps:]
$$formula$$

[If code was demonstrated, include clean Python code:]
```python
# Full code from lecture
```

[If comparing concepts, include a structured table:]
| Feature / Concept | Mechanism | Strengths | Trade-offs |
|---|---|---|---|
| value | value | value | value |

---

[REPEAT the ## section block for EVERY concept in the lecture. NEVER skip any topic.]

---

## ⚡ Comprehensive Quick Reference Table

| Concept / Technique | Mechanism & Math | Strengths | Best Used When |
|---|---|---|---|
| (Include one comprehensive row per concept/algorithm covered in the lecture) |

---

## 💡 Master Key Takeaways (Exam & Production Review)

- **[Takeaway 1]** — comprehensive explanation
- (8–12 master takeaways covering the entire lecture)

---

## 📝 Practical Implementation Notes & Nuances

- Practical tips, pitfalls to avoid, computational complexity, and best practices.

---

VIDEO TITLE: {title}

TRANSCRIPT CONTEXT:
{context}

GENERATE COMPLETE MASTER STUDY NOTES NOW:"""


NOTES_DETAILED_PART_PROMPT = """You are an expert academic professor and technical author writing Part {part_num} of {total_parts} of an exhaustive Master Study Guide for the lecture: "{title}".
Total Lecture Duration: {duration_str}
This Part covers timestamps: [{start_ts}] to [{end_ts}].

CRITICAL INSTRUCTIONS FOR UNCOMPRESSED DEPTH:
1. COMPLETE CHAPTER COVERAGE: Generate in-depth, thorough notes for EVERY concept, derivation, theorem, algorithm, parameter, and code implementation introduced between [{start_ts}] and [{end_ts}].
2. MATHEMATICAL FORMULAS IN FULL LATEX:
   - Format all math variables and equations in standard LaTeX:
     - Inline math: $formula$ (e.g. $y_i$, $\\bar{{x}}$, $\\beta$)
     - Display equations: $$\\sum_{{i=1}}^{{n}} (x_i - \\bar{{x}})^2$$
   - Provide full, step-by-step mathematical derivations without skipping algebra.
3. SECTION FORMAT:
   For every subtopic/milestone in this time window:
   ## [emoji] [Topic Title] [{start_ts}]
   > **[Key Term / Concept]:** Clear formal definition.
   - Detailed conceptual breakdown (4–8 thorough, informative bullet points detailing intuition, mechanics, formulas, and trade-offs).
   - [If mathematical derivation]: Complete step-by-step LaTeX display equations ($$..$$).
   - [If step-by-step logic]: Numbered list (1., 2., 3.)
   - [If structured comparison]: Clean Markdown table (| Col 1 | Col 2 | ...)
   - [If code was shown or explained]: Full, clean Python/relevant code block with comments.
4. DO NOT output the main document title (# Title) or concluding summary tables in this part — focus 100% on rich, deep chapter notes for [{start_ts}] to [{end_ts}].

TRANSCRIPT CONTEXT FOR THIS PART:
{context}

GENERATE PART {part_num} CHAPTER NOTES NOW:"""


NOTES_DETAILED_SYNTHESIS_PROMPT = """You are an expert academic educator finalizing a Master Study Guide for the lecture: "{title}".
Total Lecture Duration: {duration_str}

Below is the complete transcript outline of the entire lecture:
{context_summary}

Generate the final synthesis section containing:
1. An Executive Quick Reference Table summarizing all major concepts, formulas (in LaTeX), strengths, and trade-offs across the ENTIRE lecture.
2. 8–12 Master Key Takeaways for exam/interview review.
3. Practical Implementation Notes, edge cases, and computational complexity.

FORMAT:
## ⚡ Master Quick Reference Table

| Concept / Method | Mathematical Formulation / Mechanism | Strengths | Best Used When / Trade-offs |
|---|---|---|---|
| (Include 6-12 rows covering the full lecture) |

---

## 💡 Master Key Takeaways (Exam & Production Review)

- **[Takeaway 1]** — in-depth explanation
- (8-12 comprehensive takeaways)

---

## 📝 Practical Implementation Notes & Nuances

- Practical tips, pitfalls, complexity analysis, and implementation best practices.

OUTPUT ONLY THE MARKDOWN SYNTHESIS SECTIONS:"""


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
    Generate structured study notes from transcript chunks with duration-scaled depth.

    For long lectures (> 75 min / 2+ hours):
      - Uses multi-part chronological partition to give every hour dedicated LLM output budget.
      - Produces complete mathematical derivations (LaTeX), code implementations, and full chapter depth.

    Args:
        chunks: list of chunk dicts (with text, start_time, end_time)
        video_title: display title of the video
        mode: "summary" or "detailed"

    Returns:
        Markdown string of notes
    """
    if not chunks:
        return "# No Content\nNo transcript content available to generate notes from."

    sorted_chunks = sorted(chunks, key=lambda c: c["start_time"])
    start_sec = sorted_chunks[0]["start_time"]
    end_sec = sorted_chunks[-1]["end_time"]
    total_duration_sec = max(1.0, end_sec - start_sec)
    duration_str = format_timestamp(total_duration_sec)
    duration_minutes = total_duration_sec / 60.0

    print(f"[notes] Generating '{mode}' notes for '{video_title}' (Duration: {duration_str}, {len(sorted_chunks)} chunks)")

    # ── 1. Summary Mode ──────────────────────────────────────────────────────────
    if mode == "summary":
        context_parts = []
        for chunk in sorted_chunks:
            ts = format_timestamp(chunk["start_time"])
            context_parts.append(f"[{ts}] {chunk['text']}")
        context = "\n\n".join(context_parts)

        prompt = NOTES_SUMMARY_PROMPT.format(
            title=video_title,
            duration_str=duration_str,
            context=context,
        )

        try:
            response = _call_gemini_with_fallback(
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.35,
                    max_output_tokens=8192 if duration_minutes > 60 else 4096,
                    top_p=0.9,
                ),
                model_candidates=NOTES_MODELS,
                preferred_key="primary",
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Summary notes generation failed: {str(e)}")

    # ── 2. Detailed Mode: Check if multi-part partition is needed (> 75 mins / 2+ hours) ──
    parts = _partition_chunks_by_duration(sorted_chunks, target_segment_duration_sec=3000.0)

    # ── 2A. Standard Length (< 75 minutes) -> Single-pass detailed generation ─────
    if len(parts) <= 1:
        context_parts = []
        for chunk in sorted_chunks:
            ts = format_timestamp(chunk["start_time"])
            context_parts.append(f"[{ts}] {chunk['text']}")
        context = "\n\n".join(context_parts)

        prompt = NOTES_DETAILED_PROMPT_SINGLE.format(
            title=video_title,
            duration_str=duration_str,
            context=context,
        )

        try:
            response = _call_gemini_with_fallback(
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.35,
                    max_output_tokens=8192,
                    top_p=0.9,
                ),
                model_candidates=NOTES_MODELS,
                preferred_key="primary",
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Detailed notes generation failed: {str(e)}")

    # ── 2B. Long Lecture (>= 75 minutes / 2+ hours) -> Multi-Part In-Depth Generation ──
    print(f"[notes] Long lecture detected ({duration_str}). Partitioning into {len(parts)} in-depth parts.")

    generated_parts_content = []

    for idx, part_chunks in enumerate(parts):
        part_num = idx + 1
        part_start_ts = format_timestamp(part_chunks[0]["start_time"])
        part_end_ts = format_timestamp(part_chunks[-1]["end_time"])

        part_context_list = []
        for chunk in part_chunks:
            ts = format_timestamp(chunk["start_time"])
            part_context_list.append(f"[{ts}] {chunk['text']}")
        part_context = "\n\n".join(part_context_list)

        part_prompt = NOTES_DETAILED_PART_PROMPT.format(
            part_num=part_num,
            total_parts=len(parts),
            title=video_title,
            duration_str=duration_str,
            start_ts=part_start_ts,
            end_ts=part_end_ts,
            context=part_context,
        )

        try:
            print(f"[notes] Generating Part {part_num}/{len(parts)} ([{part_start_ts}] - [{part_end_ts}])...")
            part_resp = _call_gemini_with_fallback(
                contents=part_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.35,
                    max_output_tokens=8192,
                    top_p=0.9,
                ),
                model_candidates=NOTES_MODELS,
                preferred_key="primary",
            )
            part_text = part_resp.text.strip()
            generated_parts_content.append(part_text)
        except Exception as e:
            print(f"[notes] [WARN] Part {part_num} generation failed: {e}. Falling back to single-pass.")
            # Fallback to single pass if multi-part has an unexpected error
            return _generate_notes_single_pass_fallback(sorted_chunks, video_title, duration_str)

    # Generate Final Synthesis across the full lecture
    synthesis_context_summary = []
    for idx, part_chunks in enumerate(parts):
        p_start = format_timestamp(part_chunks[0]["start_time"])
        p_end = format_timestamp(part_chunks[-1]["end_time"])
        # Take key excerpt summaries from each part
        sample_texts = " ".join([c["text"] for c in part_chunks[:4] + part_chunks[-4:]])
        synthesis_context_summary.append(f"Part {idx+1} ([{p_start}] - [{p_end}]): {sample_texts[:600]}...")

    synthesis_prompt = NOTES_DETAILED_SYNTHESIS_PROMPT.format(
        title=video_title,
        duration_str=duration_str,
        context_summary="\n\n".join(synthesis_context_summary),
    )

    try:
        print("[notes] Generating final synthesis (Master Reference Table & Key Takeaways)...")
        syn_resp = _call_gemini_with_fallback(
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(
                temperature=0.35,
                max_output_tokens=4096,
                top_p=0.9,
            ),
            model_candidates=NOTES_MODELS,
            preferred_key="primary",
        )
        synthesis_content = syn_resp.text.strip()
    except Exception as e:
        print(f"[notes] [WARN] Synthesis generation failed: {e}")
        synthesis_content = ""

    # Assemble the final Master Document
    header_block = (
        f"# 📚 {video_title}\n\n"
        f"> ⏱️ **Total Lecture Duration:** {duration_str} | **Comprehensive Master Study Guide ({len(parts)} Parts)**\n\n"
        f"This master study guide provides an exhaustive, chapter-by-chapter breakdown of the complete {duration_str} lecture, "
        f"including step-by-step mathematical derivations, complete formulas, and implementation code.\n\n"
        f"---\n"
    )

    parts_combined = "\n\n---\n\n".join(generated_parts_content)

    final_document = f"{header_block}\n{parts_combined}\n\n---\n\n{synthesis_content}"
    print(f"[notes] Master Study Guide assembled ({len(final_document)} chars, ~{len(final_document.split())} words)")
    return final_document.strip()


def _generate_notes_single_pass_fallback(sorted_chunks: List[dict], video_title: str, duration_str: str) -> str:
    """Fallback generator in case multi-part synthesis encounters an issue."""
    context_parts = []
    for chunk in sorted_chunks:
        ts = format_timestamp(chunk["start_time"])
        context_parts.append(f"[{ts}] {chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt = NOTES_DETAILED_PROMPT_SINGLE.format(
        title=video_title,
        duration_str=duration_str,
        context=context,
    )
    response = _call_gemini_with_fallback(
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.35,
            max_output_tokens=8192,
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
                max_output_tokens=4096,
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
