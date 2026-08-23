"""
app/api/chat.py
────────────────
Chat API endpoint — the core Q&A interaction.

Endpoint:
  POST /api/chat  →  ask a question about a processed video

Flow:
  question → retrieve relevant chunks → build RAG prompt → Gemini → answer + timestamps
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.db import get_db, Video, ChatHistory
from app.core.retrieval import retrieve_chunks_hybrid
from app.core.llm import ask_gemini


router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Request / Response Schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    video_id: str       # internal SQLite video ID
    question: str


class TimestampCitation(BaseModel):
    start: float
    end: float
    label: str          # e.g. "2:05 - 3:20"
    url: str            # clickable YouTube link with timestamp


class ChatResponse(BaseModel):
    answer: str
    cited_timestamps: List[TimestampCitation]
    video_id: str
    question: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Answer a question about a processed YouTube video.

    Steps:
      1. Validate the video exists and is in "ready" state
      2. Retrieve the most relevant transcript chunks
      3. Call Gemini with the RAG prompt
      4. Log the Q&A to chat_history
      5. Return the answer + timestamp citations
    """
    # ── Step 1: Validate video ─────────────────────────────────────────────────
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        vid_id_int = int(request.video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="video_id must be an integer.")

    video = db.query(Video).filter(Video.id == vid_id_int).first()
    if not video:
        raise HTTPException(
            status_code=404,
            detail=f"Video '{request.video_id}' not found. Please process it first."
        )
    if video.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Video is not ready yet (status: {video.status}). "
                   "Wait for processing to complete."
        )

    # ── Step 2: Retrieve relevant chunks (Hybrid BM25 + Dense) ───────────────
    try:
        chunks = retrieve_chunks_hybrid(
            video_id=video.youtube_video_id,
            question=question,
            top_k=5,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


    # ── Step 3: Call Gemini ────────────────────────────────────────────────────
    try:
        result = ask_gemini(
            question=question,
            retrieved_chunks=chunks,
            video_id=video.youtube_video_id,
            video_title=video.title or "the lecture",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Step 4: Log to chat_history ────────────────────────────────────────────
    chat_record = ChatHistory(
        video_id=vid_id_int,
        question=question,
        answer=result["answer"],
        cited_timestamps=result["cited_timestamps"],
    )
    db.add(chat_record)
    db.commit()

    # ── Step 5: Return response ────────────────────────────────────────────────
    return ChatResponse(
        answer=result["answer"],
        cited_timestamps=[TimestampCitation(**ts) for ts in result["cited_timestamps"]],
        video_id=request.video_id,
        question=question,
    )


@router.get("/history/{video_id}")
def get_chat_history(video_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the full chat history for a video.
    Useful for displaying previous questions when the user returns to a video.
    """
    try:
        vid_id_int = int(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="video_id must be an integer.")

    history = (
        db.query(ChatHistory)
        .filter(ChatHistory.video_id == vid_id_int)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )

    return {
        "video_id": video_id,
        "history": [
            {
                "id": h.id,
                "question": h.question,
                "answer": h.answer,
                "cited_timestamps": h.cited_timestamps,
                "created_at": h.created_at.isoformat(),
            }
            for h in history
        ],
    }
