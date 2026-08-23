"""
app/core/embeddings.py
───────────────────────
Embedding generation using Gemini text-embedding-004 (via google-genai SDK).

Why Gemini embeddings?
  - Free tier: 1 million tokens/day
  - 768-dimensional vectors (better than MiniLM's 384)
  - Same API key as our LLM — one key for everything
  - Strong performance on semantic similarity tasks (MTEB benchmark)

This module uses the NEW google-genai SDK (google.genai), not the deprecated
google-generativeai package.
"""

import os
import time
from typing import List
from google import genai
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 50   # texts per batch call (free tier limit is 100, we use 50 safely)
BATCH_DELAY = 0.5  # seconds between batch calls

_clients = []  # list of genai.Client instances for key pooling


def _get_clients() -> List[genai.Client]:
    """Initialize and return Gemini clients for all configured API keys."""
    global _clients
    if not _clients:
        # Load keys: prefer GEMINI_API_KEY_2 for embeddings to keep Key 1 100% fresh for Notes
        key2 = os.getenv("GEMINI_API_KEY_2")
        key1 = os.getenv("GEMINI_API_KEY")

        keys = []
        if key2:
            keys.append(key2)
        if key1 and key1 not in keys:
            keys.append(key1)

        if not keys:
            raise ValueError(
                "No GEMINI_API_KEY set. "
                "Please add GEMINI_API_KEY to your .env file."
            )
        _clients = [genai.Client(api_key=k) for k in keys]
    return _clients



def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings (e.g., transcript chunks).
    Uses task_type RETRIEVAL_DOCUMENT — optimized for stored documents.
    """
    if not texts:
        return []

    clients = _get_clients()
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        print(f"[embeddings] Embedding batch {batch_num + 1}/{total_batches} "
              f"({len(batch)} texts)...")

        batch_embedded = False
        last_err = None
        for c_idx, client in enumerate(clients):
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config={"task_type": "RETRIEVAL_DOCUMENT"},
                )
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
                batch_embedded = True
                break
            except Exception as e:
                print(f"[embeddings] [WARN] Embedding client #{c_idx+1} failed: {e}. Trying fallback...")
                last_err = e

        if not batch_embedded:
            raise RuntimeError(
                f"Failed to embed batch {batch_num + 1}: {str(last_err)}. "
                "Check your GEMINI_API_KEY and internet connection."
            )

        if i + BATCH_SIZE < len(texts):
            time.sleep(BATCH_DELAY)

    print(f"[embeddings] Successfully embedded {len(all_embeddings)} texts.")
    return all_embeddings


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string (user's question).
    Uses task_type RETRIEVAL_QUERY — optimized for query-side embedding.
    """
    clients = _get_clients()
    last_err = None

    for c_idx, client in enumerate(clients):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=query,
                config={"task_type": "RETRIEVAL_QUERY"},
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"[embeddings] [WARN] Query embedding client #{c_idx+1} failed: {e}. Trying fallback...")
            last_err = e

    raise RuntimeError(
        f"Failed to embed query: {str(last_err)}. "
        "Check your GEMINI_API_KEY and internet connection."
    )

