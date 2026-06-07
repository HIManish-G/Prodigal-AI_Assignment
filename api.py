"""
FastAPI server for the Payment Collection Agent.

Endpoints:
    POST   /sessions                    → create a new session
    POST   /chat/{session_id}           → send a message
    GET    /sessions/{session_id}       → get session state
    DELETE /sessions/{session_id}       → end session
    GET    /health                      → health check

Sessions expire after 30 minutes of inactivity.
Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_require_confirmation

from agent import Agent
from state import Stage

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SESSION_TTL_SECONDS = 30 * 60  
CLEANUP_INTERVAL_SECONDS = 60  

# ── Session Store ─────────────────────────────────────────────────────────────

class SessionEntry:
    def __init__(self) -> None:
        self.agent = Agent()
        self.created_at = time.monotonic()
        self.last_active = time.monotonic()

    def touch(self) -> None:
        self.last_active = time.monotonic()

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_active

    @property
    def is_expired(self) -> bool:
        return self.idle_seconds > SESSION_TTL_SECONDS


_sessions: Dict[str, SessionEntry] = {}
_sessions_lock = asyncio.Lock()


# ── Background Cleanup Task ───────────────────────────────────────────────────

async def _cleanup_loop() -> None:
    """Periodically remove expired sessions."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        async with _sessions_lock:
            expired = [sid for sid, s in _sessions.items() if s.is_expired]
            for sid in expired:
                del _sessions[sid]
                log.info("Session expired and removed: %s", sid)
        if expired:
            log.info("Cleanup removed %d expired session(s).", len(expired))


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    log.info("Session cleanup task started.")
    yield
    task.cancel()
    log.info("Session cleanup task stopped.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Payment Collection Agent API",
    version="1.0.0",
    description="Conversational AI agent for end-to-end payment collection.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateSessionResponse(BaseModel):
    session_id: str
    message: str                    # agent's opening message


class ChatRequest(BaseModel):
    message: str


class CardStateResponse(BaseModel):
    has_number: bool
    has_expiry: bool
    has_cvv: bool
    has_cardholder_name: bool
    complete: bool


class ChatResponse(BaseModel):
    message: str
    stage: str
    verified: bool
    balance: Optional[float]
    payment_amount: Optional[float]
    card: CardStateResponse
    terminal: bool                 


class SessionStateResponse(BaseModel):
    session_id: str
    stage: str
    verified: bool
    account_id: Optional[str]
    balance: Optional[float]
    payment_amount: Optional[float]
    card: CardStateResponse
    last_txn_id: Optional[str]
    verify_attempts: int
    idle_seconds: float
    terminal: bool


class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    payment_confirmation_required: bool
 


# ── Helpers ───────────────────────────────────────────────────────────────────

def _card_state(session: SessionEntry) -> CardStateResponse:
    card = session.agent.state.card
    return CardStateResponse(
        has_number=bool(card.number),
        has_expiry=bool(card.expiry_month and card.expiry_year),
        has_cvv=bool(card.cvv),
        has_cardholder_name=bool(card.cardholder_name),
        complete=card.complete,
    )


def _is_terminal(stage: Stage) -> bool:
    return stage in (Stage.DONE, Stage.LOCKED_OUT, Stage.ERROR)


def _get_session_or_404(session_id: str) -> SessionEntry:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    if session.is_expired:
        del _sessions[session_id]
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return session


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session() -> CreateSessionResponse:
    """Create a new agent session. Returns session ID and the agent's opening message."""
    session_id = str(uuid.uuid4())
    entry = SessionEntry()

    # Trigger the agent's greeting
    result = await asyncio.get_event_loop().run_in_executor(
        None, entry.agent.next, ""
    )

    async with _sessions_lock:
        _sessions[session_id] = entry

    log.info("Session created: %s", session_id)
    return CreateSessionResponse(
        session_id=session_id,
        message=result["message"],
    )


@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest) -> ChatResponse:
    """Send a message to the agent and receive a response."""
    async with _sessions_lock:
        session = _get_session_or_404(session_id)
        session.touch()

    # Run the blocking agent.next() in a thread pool to avoid blocking the event loop
    result = await asyncio.get_event_loop().run_in_executor(
        None, session.agent.next, body.message
    )

    state = session.agent.state
    return ChatResponse(
        message=result["message"],
        stage=state.stage.value,
        verified=state.verified,
        balance=state.balance if state.verified else None, 
        payment_amount=state.payment_amount,
        card=_card_state(session),
        terminal=_is_terminal(state.stage),
    )


@app.get("/sessions/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str) -> SessionStateResponse:
    """Get the current state of a session."""
    async with _sessions_lock:
        session = _get_session_or_404(session_id)

    state = session.agent.state
    return SessionStateResponse(
        session_id=session_id,
        stage=state.stage.value,
        verified=state.verified,
        account_id=state.account_id,
        balance=state.balance,
        payment_amount=state.payment_amount,
        card=_card_state(session),
        last_txn_id=state.last_txn_id,
        verify_attempts=state.verify_attempts,
        idle_seconds=round(session.idle_seconds, 1),
        terminal=_is_terminal(state.stage),
    )


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    """Explicitly end a session."""
    async with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found or expired.")
        del _sessions[session_id]
    log.info("Session deleted: %s", session_id)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    async with _sessions_lock:
        active = len(_sessions)
    return HealthResponse(
        status="ok",
        active_sessions=active,
        payment_confirmation_required=get_require_confirmation()
    )  