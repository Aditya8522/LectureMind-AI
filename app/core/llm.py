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


# -- Phase 2: Notes Generation -----------------------------------------------

NOTES_SUMMARY_PROMPT = """You are an expert academic note-taker. Generate concise, high-quality study notes from the lecture transcript below.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS MARKDOWN STRUCTURE:
# 📝 Summary: {title}

## 🎯 Core Topic
2-3 clear sentences explaining what this lecture is fundamentally about and why it matters.

## 📌 Key Concepts
- **Concept Name** — Clear, high-yield explanation (cite [M:SS] for major topic milestone)
- (list all core concepts with concise explanations)

## 💡 Main Takeaways
1. First major takeaway
2. Second major takeaway
(3-5 practical, high-value takeaways)

## 🔑 Key Terms & Definitions
| Term | Definition |
|------|------------|
| Term | Clear explanation |

RULES:
- Focus on high-quality conceptual clarity and clean explanations.
- Add timestamps [M:SS] ONLY for major section / concept milestones (do NOT clutter every line with timestamps).
- Use clean Markdown tables for structured definitions. Do not include ASCII flowcharts.

VIDEO TITLE: {title}

TRANSCRIPT CONTEXT:
{context}

GENERATE CONCISE SUMMARY NOTES NOW:"""


NOTES_DETAILED_PROMPT = """You are an expert academic note-taker. Cover the ENTIRE lecture — every concept from the very first minute to the very last.

CRITICAL RULE: Be CONCISE per section. Dense bullet points, not long paragraphs. This lets you cover ALL topics within the output limit.

OUTPUT FORMAT (follow EXACTLY):

# 📚 {title}

One or two sentences about what this lecture covers and why it matters.

---

## [emoji] [Section Title] [M:SS]

> **[Key term]:** One-sentence definition of the concept introduced in this section.

- **[Point]** — concise explanation
- **[Point]** — concise explanation
- (all important details from the transcript for this section)

[Add numbered list ONLY if explaining a step-by-step workflow:]
1. Step one
2. Step two

[Add ONE compact table ONLY if the section compares items or shows structured info:]
| Column | Column | Column |
|--------|--------|--------|
| value  | value  | value  |

[Add code block ONLY if code was actually shown in the lecture:]
Example:
```python
# Actual code from the lecture
```

---

[REPEAT the ## section block for EVERY concept in the lecture. NEVER skip any topic.]

---

## ⚡ Quick Reference Table

| Concept / Type | Mechanism | Strengths | Best Used When |
|---|---|---|---|
| (one row per concept/type covered in the lecture) |

---

## 💡 Key Takeaways

- **[Takeaway]** — explanation
(5–8 key takeaways)

---

## 📝 Additional Notes

- Extra tips, caveats, or practical advice from the lecture.

---

STRICT RULES:
1. FULL COVERAGE: Go from minute 0 to the very end. If space is tight, compress earlier sections — NEVER drop later topics.
2. ONE ## PER CONCEPT: Keep each section focused. Avoid deep ### nesting.
3. BLOCKQUOTE FOR DEFINITIONS: Use `> **Term:** definition` for core terms being defined.
4. TIMESTAMPS: Only on ## headings (e.g. `## 🔍 MMR Retriever [28:10]`). Never in bullets.
5. NO FLOWCHARTS/ASCII: Tables only for structured data.
6. CODE: Only when code appeared in the lecture.
7. COMPLETENESS: A student must master the full topic from these notes alone.

VIDEO TITLE: {title}

TRANSCRIPT CONTEXT:
{context}

GENERATE COMPLETE NOTES COVERING THE FULL LECTURE FROM START TO FINISH:"""


def generate_notes(chunks: List[dict], video_title: str, mode: str = "summary") -> str:
    """
    Generate structured study notes from transcript chunks.

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
    context_parts = []
    for chunk in sorted_chunks:
        ts = format_timestamp(chunk["start_time"])
        context_parts.append(f"[{ts}] {chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt_template = NOTES_SUMMARY_PROMPT if mode == "summary" else NOTES_DETAILED_PROMPT
    prompt = prompt_template.format(title=video_title, context=context)

    try:
        response = _call_gemini_with_fallback(
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=8192 if mode == "detailed" else 2048,
                top_p=0.9,
            ),
            model_candidates=NOTES_MODELS,
            preferred_key="primary",
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Notes generation failed: {str(e)}")


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
