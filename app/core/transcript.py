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

    return blocks


def _translate_segments(
    raw_segments: list,
    source_language: str,
) -> list:
    """
    Translate non-English transcript segments to English using a multi-layer strategy:
      1. Primary: Fast Google Translator (deep-translator) in grouped paragraph batches.
         - Translates a 20-min video in ~10-15s
         - Consumes ZERO Gemini API quota (preserves quota for Chat, Notes, Quiz)
      2. Secondary Fallback: Gemini LLM translator if deep-translator fails.
      3. Tertiary Fallback: Original language text.

    Args:
        raw_segments    : list of segment objects/dicts with .text/.start/.duration
        source_language : language name (e.g. "Hindi", "Spanish", "French")

    Returns:
        list of dicts: [{"text": translated_str, "start": float, "duration": float}]
    """
    blocks = _group_segments_for_translation(raw_segments)
    if not blocks:
        return []

    print(f"[transcript] Pre-grouped {len(raw_segments)} snippets into {len(blocks)} speech blocks for translation.")

    # ── Strategy A: Fast Google Translator ────────────────────────────────────
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='en')
        translated_blocks = []
        group_size = 5
        total_groups = (len(blocks) + group_size - 1) // group_size

        for g_start in range(0, len(blocks), group_size):
            group = blocks[g_start : g_start + group_size]
            g_num = g_start // group_size + 1
            combined_text = "\n".join(b["text"] for b in group)

            try:
                translated_comb = translator.translate(combined_text)
                trans_lines = [l.strip() for l in translated_comb.split("\n") if l.strip()]
                for idx, b in enumerate(group):
                    t_txt = trans_lines[idx] if idx < len(trans_lines) else b["text"]
                    translated_blocks.append({
                        "text": t_txt,
                        "start": b["start"],
                        "duration": b["duration"],
                    })
            except Exception as ge:
                print(f"[transcript] [WARN] Translation group {g_num}/{total_groups} failed with GoogleTranslator ({ge}). Falling back to individual.")
                for b in group:
                    try:
                        translated_blocks.append({
                            "text": translator.translate(b["text"]),
                            "start": b["start"],
                            "duration": b["duration"],
                        })
                    except Exception:
                        translated_blocks.append(b)

        print(f"[transcript] [OK] Successfully translated {len(translated_blocks)} speech blocks to English.")
        return translated_blocks

    except Exception as err:
        print(f"[transcript] [WARN] Fast Google Translator unavailable ({err}). Falling back to Gemini translator...")

    # ── Strategy B: Gemini Translation Fallback ───────────────────────────────
    return _translate_segments_with_gemini(blocks, source_language)


def _translate_segments_with_gemini(
    blocks: list,
    source_language: str,
    batch_size: int = 25,
) -> list:
    """
    Fallback translation using Gemini API.
    """
    import time
    try:
        from google import genai
        from google.genai import types as genai_types
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return blocks

        client = genai.Client(api_key=api_key)
    except Exception:
        return blocks

    translated_segments = []
    total_batches = (len(blocks) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(blocks), batch_size):
        batch = blocks[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        numbered_lines = [f"{i+1}| {b['text']}" for i, b in enumerate(batch)]
        source_text = "\n".join(numbered_lines)

        prompt = f"""You are a professional educational translator. Translate the following {source_language} lecture transcript segments into fluent, clear English.

RULES:
- Maintain the exact format: [Number]| [English translation]
- Exactly one line per numbered item (e.g. "1| Today we will learn about RAG...")
- Translate technical terms, explanations, and conversational phrasing naturally for study notes
- Do NOT skip any numbers. Output ALL {len(batch)} numbered lines.
- Output ONLY the numbered translations, nothing else.

{source_language} segments to translate:
{source_text}

English translations:"""

        translated_lines = {}
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=4000,
                    ),
                )
                raw_text = response.text.strip()
                translated_lines = _parse_pipe_translations(raw_text, len(batch))
                print(f"[transcript] [OK] Gemini translated batch {batch_num}/{total_batches} ({len(batch)} blocks)")
                break
            except Exception as e:
                print(f"[transcript] [WARN] Batch {batch_num}/{total_batches} attempt {attempt} failed: {e}")
                if attempt < 3:
                    time.sleep(15)
                else:
                    translated_lines = {i: b["text"] for i, b in enumerate(batch)}

        for i, b in enumerate(batch):
            text_val = translated_lines.get(i, "").strip() or b["text"]
            translated_segments.append({
                "text": text_val,
                "start": b["start"],
                "duration": b["duration"],
            })

    return translated_segments


def _parse_pipe_translations(text: str, expected_count: int) -> dict:
    """
    Parse Gemini's pipe-delimited output like:
      1| In this video, we will learn about RAG...
      2| Generative AI has made this possible...
    Returns a dict mapping 0-based index -> translated string.
    """
    lines = text.strip().split("\n")
    result = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\s*[|.:)]\s*(.+)$", line)
        if match:
            idx = int(match.group(1)) - 1
            result[idx] = match.group(2).strip()

    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_segment_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace from a transcript segment."""
    text = re.sub(r"<[^>]+>", "", text)
    text = " ".join(text.split())
    return text.strip()
