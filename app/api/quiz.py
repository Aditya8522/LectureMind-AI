"""
app/api/quiz.py
────────────────
Quiz generation and grading API endpoints.

Endpoints:
  POST /api/quiz          → generate MCQ quiz from video transcript
  POST /api/quiz/submit   → grade a submitted quiz attempt
  GET  /api/quiz/history/{video_id} → get past quiz attempts for a video
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from sqlalchemy.orm import Session

from app.models.db import get_db, Video, QuizAttempt
from app.core.retrieval import retrieve_all_chunks_for_video
from app.core.llm import generate_quiz, grade_quiz

router = APIRouter(prefix="/api/quiz", tags=["quiz"])


# ── Request / Response Schemas ─────────────────────────────────────────────────

class QuizGenerateRequest(BaseModel):
    video_id: str
    num_questions: int = 5      # 3, 5, or 10


class QuizOption(BaseModel):
    index: int
    text: str


class QuizQuestion(BaseModel):
    id: int
    question: str
    options: List[str]


class QuizGenerateResponse(BaseModel):
    quiz_id: int                # SQLite QuizAttempt ID
    video_id: str
    questions: List[QuizQuestion]
    total: int


class QuizSubmitRequest(BaseModel):
    quiz_id: int
    video_id: str
    answers: Dict[str, int]     # {"1": 2, "2": 0, ...} — question_id → selected_index


class PerQuestionResult(BaseModel):
    id: int
    question: str
    options: List[str]
    selected_index: Optional[int]
    correct_index: int
    is_correct: bool
    explanation: str
    timestamp: float


class QuizSubmitResponse(BaseModel):
    quiz_id: int
    video_id: str
    score: float                # 0.0 – 1.0
    correct_count: int
    total: int
    percentage: int             # 0 – 100
    per_question: List[PerQuestionResult]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=QuizGenerateResponse)
def create_quiz(request: QuizGenerateRequest, db: Session = Depends(get_db)):
    """
    Generate a multiple-choice quiz from the lecture transcript.

    Steps:
      1. Validate video and cap num_questions (3-10)
      2. Fetch ALL chunks for full context
      3. Call Gemini to generate MCQ questions as JSON
      4. Save quiz to SQLite (answers + score = null until submitted)
      5. Return questions WITHOUT correct answers (prevent cheating in UI)
    """
    # Validate num_questions
    num_q = max(3, min(10, request.num_questions))

    # Validate video
    try:
        vid_id_int = int(request.video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="video_id must be an integer.")

    video = db.query(Video).filter(Video.id == vid_id_int).first()
    if not video:
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id}' not found.")
    if video.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Video is not ready yet (status: {video.status})."
        )

    # Fetch all chunks
    try:
        chunks = retrieve_all_chunks_for_video(video.youtube_video_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=404, detail="No transcript chunks found for this video.")

    # Generate quiz with Gemini
    try:
        questions = generate_quiz(
            chunks=chunks,
            video_title=video.title or "Lecture",
            num_questions=num_q,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not questions:
        raise HTTPException(status_code=503, detail="Quiz generation returned no questions. Try again.")

    # Persist to SQLite (answers_json = null, score = null until submitted)
    attempt = QuizAttempt(
        video_id=vid_id_int,
        questions_json=questions,
        answers_json=None,
        score=None,
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    print(f"[quiz] Generated {len(questions)}-question quiz (attempt #{attempt.id}) for video '{video.youtube_video_id}'")

    # Strip correct_index from response — user shouldn't see answers yet
    safe_questions = [
        QuizQuestion(id=q["id"], question=q["question"], options=q["options"])
        for q in questions
    ]

    return QuizGenerateResponse(
        quiz_id=attempt.id,
        video_id=request.video_id,
        questions=safe_questions,
        total=len(safe_questions),
    )


@router.post("/submit", response_model=QuizSubmitResponse)
def submit_quiz(request: QuizSubmitRequest, db: Session = Depends(get_db)):
    """
    Grade a submitted quiz attempt.

    Steps:
      1. Load the quiz attempt from SQLite (has the correct answers)
      2. Call grade_quiz() to compare user answers vs correct answers
      3. Update the attempt record with answers + score
      4. Return detailed results per question
    """
    # Load the quiz attempt
    attempt = db.query(QuizAttempt).filter(QuizAttempt.id == request.quiz_id).first()
    if not attempt:
        raise HTTPException(
            status_code=404,
            detail=f"Quiz attempt #{request.quiz_id} not found."
        )

    # Check video match
    try:
        vid_id_int = int(request.video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="video_id must be an integer.")

    if attempt.video_id != vid_id_int:
        raise HTTPException(
            status_code=400,
            detail="quiz_id does not belong to the specified video."
        )

    # Grade the quiz
    result = grade_quiz(
        questions=attempt.questions_json,
        user_answers=request.answers,
    )

    # Persist answers + score
    attempt.answers_json = request.answers
    attempt.score = result["score"]
    db.commit()

    print(f"[quiz] Graded attempt #{attempt.id}: {result['correct_count']}/{result['total']} correct ({result['score']*100:.0f}%)")

    return QuizSubmitResponse(
        quiz_id=attempt.id,
        video_id=request.video_id,
        score=result["score"],
        correct_count=result["correct_count"],
        total=result["total"],
        percentage=round(result["score"] * 100),
        per_question=[PerQuestionResult(**pq) for pq in result["per_question"]],
    )


@router.get("/history/{video_id}")
def get_quiz_history(video_id: str, db: Session = Depends(get_db)):
    """
    Return all past quiz attempts for a video, most recent first.
    """
    try:
        vid_id_int = int(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="video_id must be an integer.")

    attempts = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.video_id == vid_id_int)
        .order_by(QuizAttempt.created_at.desc())
        .all()
    )

    return {
        "video_id": video_id,
        "attempts": [
            {
                "id": a.id,
                "num_questions": len(a.questions_json),
                "score": a.score,
                "percentage": round(a.score * 100) if a.score is not None else None,
                "submitted": a.answers_json is not None,
                "created_at": a.created_at.isoformat(),
            }
            for a in attempts
        ],
    }
