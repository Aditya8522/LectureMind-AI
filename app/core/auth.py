"""
app/core/auth.py
────────────────
Authentication and security layer.

Features:
  - PBKDF2-HMAC-SHA256 password hashing with unique random salts
  - Signed HMAC session tokens (30-day validity)
  - FastAPI dependencies for current user resolution (optional & required)
"""

import os
import time
import json
import base64
import hmac
import hashlib
import secrets
from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Depends, status
from sqlalchemy.orm import Session

from app.models.db import get_db, User

# Secret key for HMAC token signing
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "lecturemind_super_secret_jwt_hmac_key_2026_thetawave")
TOKEN_EXPIRY_SECONDS = 30 * 24 * 60 * 60  # 30 days


# ── Password Hashing (PBKDF2-HMAC-SHA256) ──────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2 with 100,000 iterations and a random 16-byte salt."""
    salt = secrets.token_hex(16)
    iterations = 100_000
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against the stored PBKDF2 hash."""
    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = parts[2]
        expected_key = parts[3]
        computed_key = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        ).hex()
        return hmac.compare_digest(expected_key, computed_key)
    except Exception:
        return False


# ── Token Generation & Verification ───────────────────────────────────────────

def create_access_token(user_id: int, email: str, name: str) -> str:
    """Generate a signed HMAC token containing user metadata and expiration."""
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": int(time.time()) + TOKEN_EXPIRY_SECONDS,
        "iat": int(time.time()),
    }
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")
    
    signature = hmac.new(
        AUTH_SECRET_KEY.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    
    return f"{payload_b64}.{signature}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify signature and expiration of an access token, returning the payload if valid."""
    if not token or "." not in token:
        return None
    try:
        payload_b64, signature = token.split(".", 1)
        expected_sig = hmac.new(
            AUTH_SECRET_KEY.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, signature):
            return None

        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += "=" * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


# ── FastAPI User Dependencies ──────────────────────────────────────────────────

def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Extract and resolve the current User from the Authorization header if present."""
    if not authorization:
        return None

    token = authorization
    if token.startswith("Bearer "):
        token = token[7:].strip()

    payload = decode_access_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    user = db.query(User).filter(User.id == user_id).first()
    return user


def get_current_user(
    user: Optional[User] = Depends(get_current_user_optional),
) -> User:
    """Require an authenticated user. Raises 401 Unauthorized if missing/invalid."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
