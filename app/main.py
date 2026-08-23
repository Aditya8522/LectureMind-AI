"""
app/main.py
────────────
FastAPI application entrypoint.

Responsibilities:
  - Create the FastAPI app instance
  - Register API routers
  - Serve the Jinja2 frontend
  - Run DB initialization on startup
  - Configure CORS (needed for fetch() calls from browser)

Run with:
  uvicorn app.main:app --reload --port 8000
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from app.models.db import init_db
from app.api import auth, videos, chat, notes, quiz


# Load environment variables from .env file
load_dotenv()

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="YouTube Lecture RAG Chatbot",
    description="Chat with YouTube lecture videos, generate notes, and take quizzes.",
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Allow all origins during development. Restrict to your domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── No-Cache Static Files ──────────────────────────────────────────────────────
# Override StaticFiles so browsers never cache JS/CSS between deploys.
from starlette.staticfiles import StaticFiles as _StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class NoCacheStaticFiles(_StaticFiles):
    """Serve static assets with no-store cache headers so browsers always fetch fresh files."""
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        # Apply no-cache only to JS and CSS (not images/fonts which are safe to cache)
        if path.endswith((".js", ".css")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.mount(
    "/static",
    NoCacheStaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ── API Routers ────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(chat.router)
app.include_router(notes.router)
app.include_router(quiz.router)


# ── Startup Event ──────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    """
    Called once when the server starts.
    Initializes the SQLite database (creates tables if they don't exist).
    """
    init_db()
    print("[main] Database initialized.")
    print("[main] YouTube RAG Chatbot is running!")


# ── Frontend Routes ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the main single-page application."""
    return templates.TemplateResponse(request, "index.html")


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Simple health check endpoint for Render and monitoring."""
    return {"status": "ok", "service": "youtube-rag-chatbot"}
