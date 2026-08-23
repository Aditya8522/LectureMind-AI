"""
app/api/videos.py
──────────────────
API endpoints for video ingestion and status.

Endpoints:
  POST /api/videos/process   → ingest a YouTube URL end-to-end
  GET  /api/videos/{video_id} → get video metadata + status

The /process endpoint runs the full pipeline synchronously (MVP):
  URL → extract ID → fetch transcript → chunk → embed → store in ChromaDB + SQLite

This is intentionally synchronous for Phase 1 simplicity.
For production, this would be a background task (Celery / FastAPI BackgroundTasks)
with a polling endpoint — the code structure already supports this because
we store "status" in the DB.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.models.db import get_db, Video, User, UserVideo
from app.core.auth import get_current_user, get_current_user_optional
from app.core.transcript import fetch_transcript
from app.core.chunking import chunk_transcript
from app.core.embeddings import embed_texts
from app.core.vectorstore import store_chunks, collection_exists, get_collection_chunk_count, get_all_chunks

router = APIRouter(prefix="/api/videos", tags=["videos"])



# ── Request / Response Schemas ─────────────────────────────────────────────────

class ProcessVideoRequest(BaseModel):
    youtube_url: str


class ProcessVideoResponse(BaseModel):
    video_id: str           # internal SQLite row ID
    youtube_video_id: str   # the 11-char YouTube ID
    title: str
    status: str
    message: str
    chunk_count: int
    from_cache: bool = False   # True when embeddings were NOT re-computed


class VideoStatusResponse(BaseModel):
    video_id: str
    youtube_video_id: str
    title: str
    channel: str
    status: str
    chunk_count: int
    error_message: str | None


class VideoLibraryItem(BaseModel):
    video_id: str
    youtube_video_id: str
    title: str
    channel: str
    chunk_count: int
    processed_at: Optional[str] = None



def _link_user_video(db: Session, user_id: int, video_id: int):
    """Ensure the video is recorded in the user's personal lecture library."""
    uv = db.query(UserVideo).filter(
        UserVideo.user_id == user_id,
        UserVideo.video_id == video_id,
    ).first()
    if not uv:
        uv = UserVideo(user_id=user_id, video_id=video_id)
        db.add(uv)
    else:
        uv.last_accessed_at = datetime.utcnow()
    db.commit()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("", response_model=List[VideoLibraryItem])
def list_videos(
    user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Return all successfully processed videos for the authenticated user.
    If unauthenticated, returns an empty list.
    """
    if not user:
        return []

    user_video_links = (
        db.query(UserVideo)
        .filter(UserVideo.user_id == user.id)
        .order_by(UserVideo.last_accessed_at.desc())
        .all()
    )

    items = []
    for uv in user_video_links:
        v = uv.video
        if v and v.status == "ready":
            items.append(
                VideoLibraryItem(
                    video_id=str(v.id),
                    youtube_video_id=v.youtube_video_id,
                    title=v.title or "Untitled",
                    channel=v.channel or "",
                    chunk_count=len(v.chunks),
                    processed_at=uv.last_accessed_at.isoformat() if uv.last_accessed_at else None,
                )
            )
    return items


@router.post("/process", response_model=ProcessVideoResponse)
def process_video(
    request: ProcessVideoRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Full ingestion pipeline for a YouTube URL (Requires authenticated user).

    Smart 3-layer sync check:
      Layer 1 — SQLite ready + ChromaDB has data   → return from cache & link to user (0 API calls)
      Layer 2 — ChromaDB has data but SQLite gone  → rebuild SQLite from ChromaDB & link to user (0 embed calls)
      Layer 3 — Fresh video                        → full pipeline (fetch + embed + store) & link to user
    """
    youtube_url = request.youtube_url.strip()

    # ── Step 1: Extract video ID first (cheap — just URL parsing + one HTTP call) ──
    try:
        transcript_data = fetch_transcript(youtube_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    youtube_video_id = transcript_data["video_id"]

    # ── Layer 1: Both SQLite AND ChromaDB are ready → pure cache hit ──────────
    existing = db.query(Video).filter(
        Video.youtube_video_id == youtube_video_id
    ).first()

    chroma_count = get_collection_chunk_count(youtube_video_id)

    if existing and existing.status == "ready" and chroma_count > 0:
        chunk_count = len(existing.chunks)
        _link_user_video(db, user.id, existing.id)
        print(f"[videos] Cache hit: '{youtube_video_id}' ({chunk_count} chunks, {chroma_count} in ChromaDB) linked to user {user.email}. Skipping API calls.")
        return ProcessVideoResponse(
            video_id=str(existing.id),
            youtube_video_id=youtube_video_id,
            title=existing.title or "",
            status="ready",
            message=f"⚡ Loaded from cache — {chunk_count} chunks ready, 0 embedding API calls used.",
            chunk_count=chunk_count,
            from_cache=True,
        )


    # ── Layer 2: ChromaDB has data but SQLite record is missing/failed ─────────
    # This happens if someone deleted the SQLite DB or the process crashed after embedding.
    # We rebuild SQLite metadata from ChromaDB — no re-embedding needed.
    if chroma_count > 0 and (existing is None or existing.status != "ready"):
        print(f"[videos] ChromaDB has {chroma_count} chunks for '{youtube_video_id}' but SQLite is stale. Rebuilding SQLite metadata...")
        try:
            chroma_chunks = get_all_chunks(youtube_video_id)  # fetch from ChromaDB

            if existing is None:
                video_record = Video(
                    youtube_video_id=youtube_video_id,
                    title=transcript_data["title"],
                    channel=transcript_data["channel"],
                    status="processing",
                    raw_transcript=transcript_data["full_text"],
                )
                db.add(video_record)
            else:
                video_record = existing
                video_record.status = "processing"
                video_record.error_message = None

            db.commit()
            db.refresh(video_record)

            # Insert chunk metadata rows into SQLite
            from app.models.db import Chunk as ChunkModel
            # Clear any stale chunk records first
            db.query(ChunkModel).filter(ChunkModel.video_id == video_record.id).delete()
            for chunk in chroma_chunks:
                chroma_id = f"{youtube_video_id}_chunk_{chunk['chunk_index']}"
                db.add(ChunkModel(
                    video_id=video_record.id,
                    text=chunk["text"],
                    start_time=chunk["start_time"],
                    end_time=chunk["end_time"],
                    chunk_index=chunk["chunk_index"],
                    chroma_id=chroma_id,
                ))

            video_record.status = "ready"
            db.commit()
            db.refresh(video_record)
            _link_user_video(db, user.id, video_record.id)

            print(f"[videos] SQLite rebuilt from ChromaDB: {len(chroma_chunks)} chunks for '{youtube_video_id}'.")
            return ProcessVideoResponse(
                video_id=str(video_record.id),
                youtube_video_id=youtube_video_id,
                title=video_record.title or "",
                status="ready",
                message=f"⚡ Restored from vector cache — {len(chroma_chunks)} chunks, 0 embedding API calls used.",
                chunk_count=len(chroma_chunks),
                from_cache=True,
            )


        except Exception as rebuild_err:
            print(f"[videos] SQLite rebuild failed: {rebuild_err}. Falling through to full ingestion.")
            # Fall through to Layer 3

    # ── Layer 3: Full fresh ingestion ──────────────────────────────────────────
    print(f"[videos] Fresh ingestion for '{youtube_video_id}'...")

    if existing:
        video_record = existing
        video_record.status = "processing"
        video_record.error_message = None
    else:
        video_record = Video(
            youtube_video_id=youtube_video_id,
            title=transcript_data["title"],
            channel=transcript_data["channel"],
            status="processing",
            raw_transcript=transcript_data["full_text"],
        )
        db.add(video_record)

    db.commit()
    db.refresh(video_record)

    try:
        # Chunk transcript
        chunks = chunk_transcript(transcript_data["segments"])
        if not chunks:
            raise ValueError("Chunking produced no output. The transcript may be empty.")

        # Embed chunks with Gemini (this costs API quota)
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_texts(chunk_texts)

        # Store embeddings in ChromaDB (permanent on disk)
        chroma_ids = store_chunks(
            video_id=youtube_video_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        # Store chunk metadata in SQLite
        from app.models.db import Chunk as ChunkModel
        db.query(ChunkModel).filter(ChunkModel.video_id == video_record.id).delete()
        for chunk, chroma_id in zip(chunks, chroma_ids):
            db.add(ChunkModel(
                video_id=video_record.id,
                text=chunk.text,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                chunk_index=chunk.chunk_index,
                chroma_id=chroma_id,
            ))

        video_record.status = "ready"
        db.commit()
        db.refresh(video_record)
        _link_user_video(db, user.id, video_record.id)

        was_translated = transcript_data.get("was_translated", False)
        detected_lang = transcript_data.get("detected_language", "en")

        translation_note = (
            f" (auto-translated from '{detected_lang}')" if was_translated else ""
        )

        return ProcessVideoResponse(
            video_id=str(video_record.id),
            youtube_video_id=youtube_video_id,
            title=video_record.title or "",
            status="ready",
            message=f"Successfully processed {len(chunks)} chunks{translation_note}.",
            chunk_count=len(chunks),
            from_cache=False,
        )

    except Exception as e:
        video_record.status = "failed"
        video_record.error_message = str(e)
        db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )



@router.get("/{video_id}", response_model=VideoStatusResponse)
def get_video_status(video_id: str, db: Session = Depends(get_db)):
    """
    Get the current status and metadata for a processed video.
    video_id here is the internal SQLite row ID (integer as string).
    """
    try:
        vid_id_int = int(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="video_id must be an integer.")

    video = db.query(Video).filter(Video.id == vid_id_int).first()
    if not video:
        raise HTTPException(status_code=404, detail=f"Video with id '{video_id}' not found.")

    return VideoStatusResponse(
        video_id=str(video.id),
        youtube_video_id=video.youtube_video_id,
        title=video.title or "",
        channel=video.channel or "",
        status=video.status,
        chunk_count=len(video.chunks),
        error_message=video.error_message,
    )


@router.get("/{video_id}/chunks")
def get_video_chunks(video_id: str, db: Session = Depends(get_db)):
    """
    Get all text chunks with timestamps for a video.
    Used by the frontend to render the live interactive transcript explorer.
    """
    try:
        vid_id_int = int(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="video_id must be an integer.")

    video = db.query(Video).filter(Video.id == vid_id_int).first()
    if not video:
        raise HTTPException(status_code=404, detail=f"Video with id '{video_id}' not found.")

    chunks = (
        db.query(from_import_Chunk if False else Video)  # type hint
    )
    from app.models.db import Chunk as ChunkModel
    chunk_records = (
        db.query(ChunkModel)
        .filter(ChunkModel.video_id == vid_id_int)
        .order_by(ChunkModel.start_time.asc())
        .all()
    )

    return {
        "video_id": str(video.id),
        "youtube_video_id": video.youtube_video_id,
        "title": video.title,
        "chunks": [
            {
                "id": c.id,
                "text": c.text,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "chunk_index": c.chunk_index,
            }
            for c in chunk_records
        ],
        "raw_transcript": video.raw_transcript,
    }

