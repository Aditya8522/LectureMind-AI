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
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from dotenv import load_dotenv

from app.core.chunking import format_timestamp


load_dotenv()

# ── Task-Specific Model Configurations ─────────────────────────────────────────
# Optimized for ultra-fast response (1-2s) and high reliability without 503 stalls:

# Smart Notes: Blazingly fast primary engine with deep academic structure & instant fallback
NOTES_MODELS = [
    "gemini-3.5-flash",      # 1.5s latency, rock-solid reliability & academic synthesis
    "gemini-3.6-flash",      # Advanced intelligence fallback
    "gemini-3.5-flash-lite", # Ultra-fast 0.9s fallback
    "gemini-3.7-flash",      # Deep reasoning fallback
]

# Practice Quiz: Fast, highly accurate structured JSON generation
QUIZ_MODELS = [
    "gemini-3.5-flash",      # Fast, highly accurate structured JSON generation
    "gemini-3.5-flash-lite", # 0.9s ultra-fast quiz fallback
    "gemini-3.6-flash",      # Advanced reasoning fallback
    "gemini-3.7-flash",      # Deep reasoning fallback
]

# AI Tutor Chat: Fast Q&A with huge daily throughput for continuous study sessions
CHAT_MODELS = [
    "gemini-3.5-flash",      # High-accuracy direct answers
    "gemini-3.5-flash-lite", # 500 RPD high-throughput study chat
    "gemini-3.1-flash-lite", # Secondary high-throughput tier
    "gemini-3.6-flash",      # Advanced fallback
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

# ── Phase 2: Notes Generation (ThetaWave Publication-Grade Architecture) ─────

NOTES_SUMMARY_PROMPT = """You are an expert academic note-taker specializing in high-retention, publication-grade study guides in the exact style of ThetaWave AI.
Generate a concise, highly structured Executive Summary from the lecture transcript below.
Total Lecture Duration: {duration_str}

CRITICAL RULES:
1. STRICT CONTEXT FIDELITY: Only include concepts, terms, examples, code, and remarks that are ACTUALLY present in the transcript. Do NOT hallucinate outside textbook trivia, quotes, or unrelated theory.
2. HIGH-DENSITY TABLES: Use clean Markdown tables for key concepts, steps, and complexities.
3. CONCISE & SCANNABLE: Keep explanations crisp, high-yield, and organized with clear emojis and headers.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS MARKDOWN STRUCTURE:

# 📝 {title}
> ⏱️ **Duration:** {duration_str} | **Executive Study Summary**

## 🔄 Core Lecture Essence & Objective
A concise 2-3 paragraph breakdown explaining the central problem, why it matters, core theoretical insights, and the overall solution paradigm presented in the lecture.

---

## 🔑 Key Concepts & Terminology
| Concept / Term | Formal Definition & Role in Lecture |
|---|---|
| (Include 4-8 core concepts directly discussed in the transcript) |

---

## 🗺️ Lecture Flow & Procedural Roadmap
1. **[Milestone 1]** [timestamp] — Brief overview of intuition, problem setup, and foundations.
2. **[Milestone 2]** [timestamp] — Core mechanisms, formulas, and step-by-step algorithms.
3. **[Milestone 3]** [timestamp] — Practical implementation, optimization, and edge cases.

---

## ⏲️ Complexity & Approach Summary
| Approach / Paradigm | Time Complexity | Space Complexity | Notes & Trade-offs |
|---|---|---|---|
| (Include one row per approach/technique discussed in the lecture) |

---

## 💡 Master Key Takeaways
- **[Takeaway 1]** — Actionable high-yield summary point.
- **[Takeaway 2]** — Actionable high-yield summary point.
(5-8 master takeaways summarizing the key learnings for quick exam/interview revision)

---

## 📚 Series & Practical Context
- (Include specific prerequisites, playlist references, or instructor recommendations mentioned in the transcript)

VIDEO TITLE: {title}

TRANSCRIPT CONTEXT:
{context}

GENERATE EXECUTIVE SUMMARY NOTES NOW:"""


NOTES_DETAILED_PROMPT_SINGLE = """You are an expert academic tutor and technical author specializing in publication-grade master study guides in the exact style of ThetaWave AI.
Cover the ENTIRE lecture in exhaustive detail — every concept, step-by-step algorithm, table, code implementation, parameter, and instructor nuance from minute 0 to the very last minute.

CRITICAL INSTRUCTIONS FOR DEPTH, STRUCTURE & ACCURACY:
1. 100% STRICT TRANSCRIPT GROUNDING: Base every definition, proof, code block, prerequisite, and tip strictly on the transcript context. Do NOT invent external historical quotes or outside textbook trivia not discussed in the video.
2. RICH MARKDOWN TABLES EVERYWHERE:
   - For step-by-step procedures: Create a `| Step | Description / Action |` table.
   - For variable tracking / state windowing: Create a `| Variable | Meaning & Purpose |` table.
   - For complexity analysis: Create a `| Approach | Time Complexity | Space Complexity | Notes |` table.
   - For terminology: Create a `| Term | Definition |` table.
3. FULL LATEX MATHEMATICS: Format all formulas and variables in standard LaTeX ($x$, $\\theta$, $$\\sum...$$). Never omit algebraic steps.
4. COMPLETE CODE SNIPPETS: Write out complete, cleanly formatted code in the primary language demonstrated in the lecture (C++, Python, Java, etc.) with step comments.
5. RECURSION TREES / WORKFLOWS: When branching or workflows are discussed, provide a clear structured diagram or ASCII tree (e.g. `f(5) ├── f(4)...`) showing redundancy.
6. METAPHORS & TEACHER INSIGHTS: Capture the instructor's intuitive metaphors, playlist prerequisites (e.g., lecture numbers), code availability notes, and practical rules of thumb.

OUTPUT STRUCTURE (follow this EXACT modular layout):

# 📚 {title}
> ⏱️ **Total Duration:** {duration_str} | **Comprehensive Master Study Guide**

Brief overview of the full lecture scope, fundamental objectives, and learning roadmap.

---

## 🔄 [Topic 1: Introduction & Motivation] [M:SS]

> **[Core Concept]:** Clear formal definition from the lecture.

- Core intuition and theoretical breakdown (3–6 detailed bullet points).
- The motivation ("Why do we need this approach?").

[If multi-approach comparison or pipeline is introduced, include a structured table or flow list:]
| Approach | Mechanism | Primary Advantage | Limitation |
|---|---|---|---|
| value | value | value | value |

---

## 🔍 [Topic 2: Foundational Recurrence / Mechanism] [M:SS]

> **[Key Concept / Recurrence]:** Mathematical or structural definition.

Mathematical recurrence and boundary conditions:
$$formula$$

[If code was shown, include complete, well-commented code block:]
```cpp
// Full implementation from lecture
```

Complexity Analysis:
- **Time Complexity:** $O(...)$ — detailed derivation.
- **Space Complexity:** $O(...)$ — detailed call stack / memory breakdown.

[If recursion tree or branching was explained, include ASCII tree:]
```text
f(5)
├── f(4)
│   ├── f(3)
│   │   ├── f(2)
...
```

---

## 💾 [Topic 3: Intermediate Optimization / Memoization] [M:SS]

> **[Key Term]:** Formal definition.

### Steps to Implement:
| Step | Description & Code Action |
|---|---|
| Step 0 | Declare and initialize cache structure |
| Step 1 | Check if subproblem is already solved before computing |
| Step 2 | Store computed result in cache before returning |

[Include complete code implementation with step annotations:]
```cpp
// Full optimized code
```

Complexity Analysis:
- **Time Complexity:** $O(...)$
- **Space Complexity:** $O(...)$

---

## 📊 [Topic 4: Iterative State Resolution / Tabulation] [M:SS]

> **[Key Term]:** Formal definition.

### Steps to Convert:
| Step | Description & State Mapping |
|---|---|
| Step 1 | Initialize table of size $n+1$ |
| Step 2 | Set base cases explicitly |
| Step 3 | Iteratively compute states from base cases to target |

[Include complete code implementation:]
```cpp
// Full tabulation code
```

Complexity Analysis:
- **Time Complexity:** $O(...)$
- **Space Complexity:** $O(...)$

---

## ⚡ [Topic 5: Space Optimization / Variable State Windowing] [M:SS]

> **[Key Concept]:** Explanation of minimum active window.

### Variables Used:
| Variable | Meaning & Stored Subproblem State |
|---|---|
| prev2 | Stores state $i-2$ |
| prev | Stores state $i-1$ |
| curr | Current state being computed |

[Include space-optimized code:]
```cpp
// Full space-optimized code
```

---

[REPEAT the detailed section format for EVERY subsequent concept, algorithm, or demonstration in the lecture. Never skip any topic.]

---

## ⏲️ Time & Space Complexity Master Summary

| Approach | Implementation Details | Time Complexity | Space Complexity | Key Notes & Overhead |
|---|---|---|---|---|
| (Complete row per approach discussed) |

---

## 🔑 Key Concepts & Terminology

| Concept / Term | Comprehensive Explanation & Role in Lecture |
|---|---|
| (Include 6-12 core concepts and terms directly from the transcript) |

---

## 🧮 Key Code Snippet Summary

| Approach / Pattern | 1–2 Line Core Logic Snippet |
|---|---|
| (Include quick-reference 1-line syntax for each technique) |

---

## 💡 Metaphor for Deep Understanding

> **[Intuitive Analogy]:** A vivid real-world metaphor explaining the core concept as illuminated by the instructor.

- Breakdown of the metaphor mapping to technical components.

---

## 🔧 Practical Implementation Tips & Nuances

- **[Tip 1]** — Concrete implementation guideline.
- **[Pitfall to Avoid]** — Common error (e.g. array indexing, passing by value vs reference, uninitialized base cases).
- (5–8 actionable best practices from the lecture)

---

## 📚 Prerequisites, Series Roadmap & Next Steps

- **Prerequisites:** Prior lectures, playlist topics, or foundational concepts required.
- **Series Roadmap:** How this lecture connects to upcoming topics and interview preparation.
- **Instructor Notes:** Specific requests (e.g., code repository, article links, engagement) mentioned in the video.

---

VIDEO TITLE: {title}

TRANSCRIPT CONTEXT:
{context}

GENERATE THE THETAWAVE-STYLE MASTER STUDY GUIDE NOW:"""


NOTES_DETAILED_PART_PROMPT = """You are an expert academic tutor and technical author writing Part {part_num} of {total_parts} of an exhaustive, publication-grade Master Study Guide in the exact style of ThetaWave AI for the lecture: "{title}".
Total Lecture Duration: {duration_str}
This Part covers timestamps: [{start_ts}] to [{end_ts}].

CRITICAL INSTRUCTIONS FOR UNCOMPRESSED THETAWAVE DEPTH:
1. STRICT TRANSCRIPT FIDELITY: Only include concepts, algorithms, code, and remarks spoken in this timestamp window. Do not invent outside theories.
2. HIGH-DENSITY TABLES: Whenever procedures, variables, or comparisons are explained, generate clean Markdown tables (`| Step | Action |` or `| Variable | Meaning |`).
3. COMPLETE LATEX MATHEMATICS: Format all formulas, recurrences, and variables in standard LaTeX ($x$, $$\\sum...$$).
4. COMPLETE CODE BLOCKS: Provide full, clean code implementations in the primary language demonstrated.
5. SECTION FORMAT:
   For every subtopic in this time window:
   ## [emoji] [Topic Title] [{start_ts}]
   > **[Core Concept]:** One-sentence formal definition.
   - Comprehensive conceptual breakdown (4–8 thorough, scannable bullet points).
   - [If multi-step recipe]: Markdown table of steps (`| Step | Description |`).
   - [If variable windowing]: Markdown table of variables (`| Variable | Meaning |`).
   - [If mathematical proof]: Full LaTeX display equations ($$..$$).
   - [If code demonstrated]: Complete, annotated code snippet.
   - **Complexity Analysis:** Explicit breakdown of Time $O(...)$ and Space $O(...)$.
6. DO NOT output the main document title (# Title) or concluding summary tables in this part — focus 100% on rich, deep chapter notes for [{start_ts}] to [{end_ts}].

TRANSCRIPT CONTEXT FOR THIS PART:
{context}

GENERATE PART {part_num} CHAPTER NOTES NOW:"""


NOTES_DETAILED_SYNTHESIS_PROMPT = """You are an expert academic educator finalizing a publication-grade Master Study Guide in the exact style of ThetaWave AI for the lecture: "{title}".
Total Lecture Duration: {duration_str}

Below is the complete transcript outline across all lecture parts:
{context_summary}

Generate the final synthesis section containing:
1. ⏲️ **Time & Space Complexity Master Summary Table** across all approaches.
2. 🔑 **Key Concepts & Terminology Table** (6–12 core terms with formal explanations).
3. 🧮 **Key Code Snippet Summary Table** (1–2 line core logic snippet per approach).
4. 💡 **Metaphor for Deep Understanding** (Intuitive analogy reflecting the instructor's explanation).
5. 🔧 **Practical Implementation Tips & Pitfalls to Avoid**.
6. 📚 **Prerequisites, Series Roadmap & Next Steps** (capturing playlist context and instructor notes).

OUTPUT STRUCTURE:

## ⏲️ Time & Space Complexity Master Summary

| Approach | Implementation Details | Time Complexity | Space Complexity | Key Notes & Overhead |
|---|---|---|---|---|
| (Include one comprehensive row per approach discussed) |

---

## 🔑 Key Concepts & Terminology

| Concept / Term | Comprehensive Explanation & Role in Lecture |
|---|---|
| (Include 6-12 core concepts and terms directly from the transcript) |

---

## 🧮 Key Code Snippet Summary

| Approach / Pattern | 1–2 Line Core Logic Snippet |
|---|---|
| (Include quick-reference 1-line syntax for each technique) |

---

## 💡 Metaphor for Deep Understanding

> **[Intuitive Analogy]:** Clear real-world metaphor explaining the core concept.

- Breakdown of the metaphor mapping to technical components.

---

## 🔧 Practical Implementation Tips & Pitfalls to Avoid

- **[Best Practice]** — Concrete implementation guideline.
- **[Common Pitfall]** — Pitfall to avoid (e.g. base cases, recursion depth, memory allocation).
- (5–8 actionable best practices from the lecture)

---

## 📚 Prerequisites, Series Roadmap & Next Steps

- **Prerequisites:** Foundational topics and playlist prerequisites mentioned in the lecture.
- **Series Roadmap:** How this lecture connects to upcoming topics and interview preparation.
- **Instructor Notes:** Engagement requests and resource links from the video.

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

    # ── 2B. Long Lecture (>= 75 minutes / 2+ hours) -> Parallel Multi-Part Generation ──
    print(f"[notes] Long lecture detected ({duration_str}). Partitioning into {len(parts)} in-depth parts with Dual-Key Parallelism.")

    def _generate_single_part_worker(part_tuple):
        idx, part_chunks, total_parts, video_title_arg, duration_str_arg = part_tuple
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
            total_parts=total_parts,
            title=video_title_arg,
            duration_str=duration_str_arg,
            start_ts=part_start_ts,
            end_ts=part_end_ts,
            context=part_context,
        )

        # Distribute parts evenly across Key 1 (primary) and Key 2 (secondary)
        assigned_key = "primary" if (idx % 2 == 0) else "secondary"
        print(f"[notes] [Parallel] Launching Part {part_num}/{total_parts} ([{part_start_ts}] - [{part_end_ts}]) via {assigned_key}...")

        part_resp = _call_gemini_with_fallback(
            contents=part_prompt,
            config=types.GenerateContentConfig(
                temperature=0.35,
                max_output_tokens=8192,
                top_p=0.9,
            ),
            model_candidates=NOTES_MODELS,
            preferred_key=assigned_key,
        )
        return (idx, part_resp.text.strip())

    try:
        max_workers = min(len(parts), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            worker_args = [(idx, p, len(parts), video_title, duration_str) for idx, p in enumerate(parts)]
            futures = [executor.submit(_generate_single_part_worker, arg) for arg in worker_args]
            part_results = [f.result() for f in futures]

        # Sort back into chronological order
        part_results.sort(key=lambda r: r[0])
        generated_parts_content = [r[1] for r in part_results]
        print(f"[notes] [Parallel] All {len(parts)} parts generated concurrently!")

    except Exception as e:
        print(f"[notes] [WARN] Parallel multi-part generation failed: {e}. Falling back to single-pass.")
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
