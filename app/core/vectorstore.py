"""
app/core/vectorstore.py
────────────────────────
ChromaDB wrapper for storing and retrieving transcript chunks.

Design decisions:
  - One ChromaDB collection per video, named after the YouTube video ID.
    This keeps videos completely isolated — no cross-contamination.
  - We use a persistent ChromaDB client so data survives app restarts.
    The database files live in chroma_db/ at the project root.
  - Embeddings are computed externally (in embeddings.py) and passed in.
    This lets us swap the embedding model later without touching this file.
  - Metadata stored per chunk: video_id, start_time, end_time, chunk_index.
    This is what enables timestamp citations in answers.
"""

import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings

from app.core.chunking import ChunkData


# ── ChromaDB Client Setup ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# We keep a single client instance (module-level) to avoid re-initializing
# ChromaDB on every request. This is safe in a single-process FastAPI app.
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Return the global ChromaDB persistent client, creating it if needed.
    The client writes to chroma_db/ directory — all data survives restarts.
    """
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        print(f"[vectorstore] ChromaDB initialized at: {CHROMA_DB_PATH}")
    return _chroma_client


def get_collection_name(video_id: str) -> str:
    """
    Generate a safe ChromaDB collection name from a YouTube video ID.
    ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens.
    YouTube video IDs can contain underscores — we replace them with hyphens.
    """
    # Prefix with "vid-" in case the video ID starts with a number (ChromaDB restriction)
    safe_name = f"vid-{video_id}".replace("_", "-").lower()
    return safe_name


# ── Store Chunks ───────────────────────────────────────────────────────────────

def store_chunks(
    video_id: str,
    chunks: List[ChunkData],
    embeddings: List[List[float]],
) -> List[str]:
    """
    Store transcript chunks and their embeddings in ChromaDB.

    Args:
      video_id   : YouTube video ID (used as collection name)
      chunks     : list of ChunkData objects (text + timestamp metadata)
      embeddings : list of embedding vectors (one per chunk, same order)

    Returns:
      list of ChromaDB document IDs (one per chunk)

    Raises:
      ValueError : if chunks and embeddings have different lengths
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
        )

    client = get_chroma_client()
    collection_name = get_collection_name(video_id)

    # Create or get the collection for this video
    # get_or_create_collection is idempotent — safe to call multiple times
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    # Build IDs, documents, metadatas for ChromaDB
    doc_ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # Unique ID: video_id + chunk_index ensures no collisions
        doc_id = f"{video_id}_chunk_{chunk.chunk_index}"
        doc_ids.append(doc_id)
        documents.append(chunk.text)
        metadatas.append({
            "video_id": video_id,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "chunk_index": chunk.chunk_index,
        })

    # Store everything in one batch call
    collection.add(
        ids=doc_ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[vectorstore] Stored {len(chunks)} chunks in collection '{collection_name}'.")
    return doc_ids


# ── Query Collection ───────────────────────────────────────────────────────────

def query_collection(
    video_id: str,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[dict]:
    """
    Retrieve the top-k most similar chunks for a given query embedding.

    Args:
      video_id        : YouTube video ID (identifies which collection to search)
      query_embedding : embedding vector of the user's question
      top_k           : number of chunks to retrieve (default: 5)

    Returns:
      list of result dicts, each containing:
        {
          "text"        : str,    # chunk text
          "start_time"  : float,  # seconds into video
          "end_time"    : float,  # seconds into video
          "chunk_index" : int,    # position in video
          "distance"    : float,  # cosine distance (lower = more similar)
        }

    Raises:
      ValueError: if the collection for this video doesn't exist
    """
    client = get_chroma_client()
    collection_name = get_collection_name(video_id)

    # Check collection exists
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        raise ValueError(
            f"No ChromaDB collection found for video '{video_id}'. "
            "Make sure the video has been processed first."
        )

    # Run similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # can't request more than exists
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested list format
    # results["documents"][0] is the list of texts for our single query
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "start_time": meta.get("start_time", 0.0),
            "end_time": meta.get("end_time", 0.0),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
        })

    return retrieved


# ── Utility ────────────────────────────────────────────────────────────────────

def collection_exists(video_id: str) -> bool:
    """
    Check whether a ChromaDB collection exists for the given video ID.
    Used to skip re-processing videos that have already been embedded.
    """
    client = get_chroma_client()
    collection_name = get_collection_name(video_id)
    existing_collections = [col.name for col in client.list_collections()]
    return collection_name in existing_collections


def get_collection_chunk_count(video_id: str) -> int:
    """
    Return the number of chunks stored in ChromaDB for this video.
    Returns 0 if the collection does not exist.
    Used during ingestion to detect when embeddings can be reused.
    """
    client = get_chroma_client()
    collection_name = get_collection_name(video_id)
    try:
        collection = client.get_collection(name=collection_name)
        return collection.count()
    except Exception:
        return 0


def delete_collection(video_id: str) -> bool:
    """
    Delete the ChromaDB collection for a video.
    Returns True if deleted, False if it didn't exist.
    Useful for re-processing a video from scratch.
    """
    client = get_chroma_client()
    collection_name = get_collection_name(video_id)
    try:
        client.delete_collection(name=collection_name)
        print(f"[vectorstore] Deleted collection '{collection_name}'.")
        return True
    except Exception:
        return False


def get_all_chunks(video_id: str) -> List[dict]:
    """
    Retrieve ALL chunks for a video (used for notes & quiz generation
    where we want the full transcript context, not just top-k).

    Returns list of chunk dicts with text, start_time, end_time, chunk_index.
    """
    client = get_chroma_client()
    collection_name = get_collection_name(video_id)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        raise ValueError(
            f"No ChromaDB collection found for video '{video_id}'. "
            "Make sure the video has been processed first."
        )

    total = collection.count()
    if total == 0:
        return []

    results = collection.get(
        include=["documents", "metadatas"],
        limit=total,
    )

    chunks = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({
            "text": doc,
            "start_time": meta.get("start_time", 0.0),
            "end_time": meta.get("end_time", 0.0),
            "chunk_index": meta.get("chunk_index", 0),
        })

    # Sort by time order
    chunks.sort(key=lambda c: c["start_time"])
    print(f"[vectorstore] Retrieved all {len(chunks)} chunks for video '{video_id}'.")
    return chunks

