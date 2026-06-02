from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ChatTurn:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class SessionState:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    history: list[ChatTurn] = field(default_factory=list)
    last_result: dict[str, Any] | None = None

    def add_user_message(self, content: str, **metadata: Any) -> None:
        self.history.append(ChatTurn(role="user", content=content, metadata=metadata))

    def add_assistant_message(self, content: str, **metadata: Any) -> None:
        self.history.append(ChatTurn(role="assistant", content=content, metadata=metadata))

    def clear(self) -> None:
        self.history.clear()
        self.last_result = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "history": [turn.to_dict() for turn in self.history],
            "last_result": self.last_result,
        }


class StateManager:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str | None = None) -> SessionState:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        state = SessionState(session_id=session_id or uuid4().hex)
        self.sessions[state.session_id] = state
        return state

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
