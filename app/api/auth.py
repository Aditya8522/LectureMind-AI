"""
app/api/auth.py
───────────────
API endpoints for user registration, authentication, demo guest login, and user profile.

Endpoints:
  POST /api/auth/signup  → Register a new user account
  POST /api/auth/login   → Authenticate with email & password
  POST /api/auth/guest   → 1-click Demo Student login
  GET  /api/auth/me      → Get current user profile and study stats
  POST /api/auth/logout  → Log out
"""

import re
import secrets
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session


from app.models.db import get_db, User, UserVideo, Video, ChatHistory
from app.core.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

AVATAR_COLORS = [
    "#6366F1",  # Indigo
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
    "#10B981",  # Emerald
    "#3B82F6",  # Blue
    "#F59E0B",  # Amber
    "#06B6D4",  # Cyan
]


# ── Schemas ────────────────────────────────────────────────────────────────────

EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-\.]+$")

class SignUpRequest(BaseModel):
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=4)
    name: str = Field(..., min_length=2)


class LoginRequest(BaseModel):
    email: str
    password: str



class UserProfileResponse(BaseModel):
    id: int
    email: str
    name: str
    avatar_color: str
    is_guest: bool
    video_count: int
    chat_count: int


class AuthResponse(BaseModel):
    token: str
    user: UserProfileResponse
    message: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_user_response(user: User, db: Session) -> UserProfileResponse:
    video_count = db.query(UserVideo).filter(UserVideo.user_id == user.id).count()
    # Count chats associated with videos the user accessed
    user_video_ids = [uv.video_id for uv in user.user_videos]
    chat_count = (
        db.query(ChatHistory).filter(ChatHistory.video_id.in_(user_video_ids)).count()
        if user_video_ids else 0
    )
    return UserProfileResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        avatar_color=user.avatar_color or "#6366F1",
        is_guest=bool(user.is_guest),
        video_count=video_count,
        chat_count=chat_count,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/signup", response_model=AuthResponse)
def signup(request: SignUpRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    email_clean = request.email.strip().lower()
    name_clean = request.name.strip()

    existing = db.query(User).filter(User.email == email_clean).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists. Please sign in.",
        )

    # Pick a distinct color based on user name length
    color = AVATAR_COLORS[len(name_clean) % len(AVATAR_COLORS)]

    user = User(
        email=email_clean,
        name=name_clean,
        password_hash=hash_password(request.password),
        avatar_color=color,
        is_guest=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email, user.name)
    user_profile = _build_user_response(user, db)

    return AuthResponse(
        token=token,
        user=user_profile,
        message="Account created successfully!",
    )


@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with email and password."""
    email_clean = request.email.strip().lower()
    user = db.query(User).filter(User.email == email_clean).first()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token(user.id, user.email, user.name)
    user_profile = _build_user_response(user, db)

    return AuthResponse(
        token=token,
        user=user_profile,
        message=f"Welcome back, {user.name}!",
    )


@router.post("/guest", response_model=AuthResponse)
def guest_login(db: Session = Depends(get_db)):
    """
    Instant 1-click Demo Student login.
    Creates or reuses the demo account and links all pre-existing videos to it.
    """
    guest_email = "demo.student@lecturemind.ai"
    user = db.query(User).filter(User.email == guest_email).first()

    if not user:
        user = User(
            email=guest_email,
            name="Demo Student",
            password_hash=hash_password("demopassword2026"),
            avatar_color="#6366F1",
            is_guest=1,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Link all existing ready videos to this demo user if not linked yet
    ready_videos = db.query(Video).filter(Video.status == "ready").all()
    for v in ready_videos:
        link_exists = db.query(UserVideo).filter(
            UserVideo.user_id == user.id,
            UserVideo.video_id == v.id,
        ).first()
        if not link_exists:
            db.add(UserVideo(user_id=user.id, video_id=v.id))
    db.commit()

    token = create_access_token(user.id, user.email, user.name)
    user_profile = _build_user_response(user, db)

    return AuthResponse(
        token=token,
        user=user_profile,
        message="Signed in as Demo Student!",
    )


@router.get("/me", response_model=UserProfileResponse)
def get_me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return the profile and stats for the authenticated user."""
    return _build_user_response(user, db)


@router.post("/logout")
def logout():
    """Client-side token removal endpoint."""
    return {"message": "Logged out successfully."}
