"""
app/core/retrieval.py
──────────────────────
Retrieval pipeline: question → top-k relevant chunks.

Phase 1: Pure dense vector retrieval (ChromaDB cosine similarity).
Phase 2: Hybrid BM25 + Dense with Reciprocal Rank Fusion (RRF) fusion.

BM25 is excellent for:
  - Exact keyword matches (e.g. technical terms, proper names)
  - Short, specific queries ("what is LSTM?")

Dense retrieval is excellent for:
  - Semantic similarity ("explain how memory works" → retrieves LSTM chunk)
  - Paraphrased questions

Combining both with RRF gives the best of both worlds.
"""

from typing import List

from app.core.embeddings import embed_query
from app.core.vectorstore import query_collection, get_all_chunks

# Number of chunks to retrieve per question.
TOP_K = 5
# RRF constant (typical value is 60 — prevents top-ranked docs from dominating)
RRF_K = 60


def retrieve_chunks(video_id: str, question: str, top_k: int = TOP_K) -> List[dict]:
    """
    Phase 1 dense retrieval — kept for backward compat and as a fallback.
    Embeds the question and queries ChromaDB for top-k similar chunks.
    """
    print(f"[retrieval] Dense retrieval: top-{top_k} for video '{video_id}'")

    try:
        query_embedding = embed_query(question)
    except Exception as e:
        raise RuntimeError(f"Failed to embed question: {str(e)}")

    try:
        chunks = query_collection(
            video_id=video_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )
    except ValueError:
        raise

    if chunks:
        print(f"[retrieval] Dense: retrieved {len(chunks)} chunks. "
              f"Top distance: {chunks[0]['distance']:.4f}")

    return chunks


def retrieve_chunks_hybrid(video_id: str, question: str, top_k: int = TOP_K) -> List[dict]:
    """
    Phase 2 Hybrid retrieval: BM25 keyword search + Dense vector search,
    fused with Reciprocal Rank Fusion (RRF).

    Falls back gracefully to dense-only if rank_bm25 is not installed.
    """
    print(f"[retrieval] Hybrid retrieval: top-{top_k} for video '{video_id}'")

    try:
        all_chunks = get_all_chunks(video_id)
    except ValueError:
        raise

    if not all_chunks:
        return []

    # ── BM25 keyword ranking ───────────────────────────────────────────────────
    bm25_ranked = []
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = question.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        indexed_bm25 = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )
        bm25_ranked = [
            (all_chunks[i], rank + 1)
            for rank, (i, _) in enumerate(indexed_bm25[:top_k * 2])
        ]
        print(f"[retrieval] BM25: ranked {len(bm25_ranked)} candidates")

    except ImportError:
        print("[retrieval] rank_bm25 not installed — using dense retrieval only")
        return retrieve_chunks(video_id, question, top_k)

    # ── Dense embedding ranking ────────────────────────────────────────────────
    try:
        query_embedding = embed_query(question)
        dense_results = query_collection(
            video_id=video_id,
            query_embedding=query_embedding,
            top_k=top_k * 2,
        )
        dense_ranked = [(chunk, rank + 1) for rank, chunk in enumerate(dense_results)]
        print(f"[retrieval] Dense: ranked {len(dense_ranked)} candidates")

    except Exception as e:
        print(f"[retrieval] Dense failed: {e}. Returning BM25-only results.")
        return [chunk for chunk, _ in bm25_ranked[:top_k]]

    # ── Reciprocal Rank Fusion ─────────────────────────────────────────────────
    rrf_scores: dict = {}
    chunk_map: dict = {}

    def chunk_key(chunk: dict) -> str:
        return f"{chunk['start_time']:.3f}"

    for chunk, rank in bm25_ranked:
        key = chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunk_map[key] = chunk

    for chunk, rank in dense_ranked:
        key = chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunk_map[key] = chunk

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_chunks = []
    for key, score in fused[:top_k]:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = score
        chunk.setdefault("distance", 1.0 - score)
        final_chunks.append(chunk)

    print(f"[retrieval] Hybrid RRF: returning {len(final_chunks)} fused chunks")
    return final_chunks


def retrieve_all_chunks_for_video(video_id: str) -> List[dict]:
    """
    Return ALL chunks for a video ordered by timestamp.
    Used by notes/quiz generation endpoints for full transcript context.
    """
    return get_all_chunks(video_id)

