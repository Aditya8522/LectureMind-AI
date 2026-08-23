"""
app/models/db.py
─────────────────
Database layer: SQLite via SQLAlchemy.

Tables:
  - videos       : one row per processed YouTube video
  - chunks       : one row per text chunk (with timestamp metadata)
  - chat_history : one row per Q&A exchange

We use SQLAlchemy's declarative ORM so the schema is defined in Python
classes — easy to read, easy to extend.

The database file lives at data/youtube_rag.db (created automatically).
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Path setup ─────────────────────────────────────────────────────────────────
# Put the DB file in the data/ directory at the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "youtube_rag.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ── Engine & Session ───────────────────────────────────────────────────────────
# check_same_thread=False is required for FastAPI because it serves requests
# on multiple threads while SQLite defaults to single-thread mode.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # Set to True during development to see SQL queries in console
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ── ORM Models ─────────────────────────────────────────────────────────────────

class Video(Base):
    """
    One row per YouTube video that has been submitted for processing.

    status values:
      "processing" → pipeline is running
      "ready"      → fully processed, ready for chat
      "failed"     → something went wrong (check error_message)
    """
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    youtube_video_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=True)              # fetched from YouTube
    channel = Column(String, nullable=True)            # channel name if available
    duration_seconds = Column(Float, nullable=True)    # total video length
    status = Column(String, default="processing")      # processing | ready | failed
    error_message = Column(Text, nullable=True)        # set on failure
    raw_transcript = Column(Text, nullable=True)       # full transcript as plain text
    processed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships let us do video.chunks and video.chat_history easily
    chunks = relationship("Chunk", back_populates="video", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="video", cascade="all, delete-orphan")
    notes = relationship("Note", back_populates="video", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="video", cascade="all, delete-orphan")
    user_videos = relationship("UserVideo", back_populates="video", cascade="all, delete-orphan")


class User(Base):
    """
    User accounts for authentication, personalized lecture history, and stats.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    avatar_color = Column(String, default="#6366F1")
    is_guest = Column(Integer, default=0)              # 1 for temporary guest, 0 for standard
    created_at = Column(DateTime, default=datetime.utcnow)

    user_videos = relationship("UserVideo", back_populates="user", cascade="all, delete-orphan")


class UserVideo(Base):
    """
    Junction record linking a User to their saved / accessed Video lectures.
    Allows multiple users to study the same video without re-embedding.
    """
    __tablename__ = "user_videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    last_accessed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="user_videos")
    video = relationship("Video", back_populates="user_videos")




class Chunk(Base):
    """
    One row per text chunk generated from a video transcript.

    Each chunk knows its start/end time in the video → this is what
    powers timestamp citations in chat answers.

    chroma_id is the ID used inside ChromaDB so we can cross-reference
    if needed.
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)                # the actual chunk text
    start_time = Column(Float, nullable=False)         # seconds from video start
    end_time = Column(Float, nullable=False)           # seconds from video start
    chunk_index = Column(Integer, nullable=False)      # order within the video
    chroma_id = Column(String, nullable=True)          # ID inside ChromaDB collection

    video = relationship("Video", back_populates="chunks")


class ChatHistory(Base):
    """
    One row per question asked in the chat interface.

    cited_timestamps is stored as JSON — a list of dicts like:
      [{"start": 120.0, "end": 145.0, "url": "https://youtu.be/VIDEO_ID?t=120"}]
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    cited_timestamps = Column(JSON, nullable=True)     # list of timestamp dicts
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="chat_history")


class Note(Base):
    """
    Generated study notes for a video.

    mode values:
      "summary"  → concise bullet-point overview
      "detailed" → full structured Markdown with headings & timestamps
    """
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    mode = Column(String, nullable=False)              # "summary" | "detailed"
    content = Column(Text, nullable=False)             # Markdown string
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="notes")


class QuizAttempt(Base):
    """
    One row per quiz attempt (generated + optionally submitted).

    questions_json: list of question dicts generated by LLM
    answers_json:   user's selected answers (null until submitted)
    score:          fraction correct (null until submitted)
    """
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    questions_json = Column(JSON, nullable=False)      # list of MCQ question dicts
    answers_json = Column(JSON, nullable=True)         # user's answers
    score = Column(Float, nullable=True)               # 0.0 – 1.0
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="quiz_attempts")


# ── Helpers ────────────────────────────────────────────────────────────────────

def init_db():
    """
    Create all tables if they do not exist yet.
    Called once at application startup from main.py.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency that yields a database session and closes it
    automatically when the request is done.

    Usage in a route:
        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

