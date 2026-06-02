from __future__ import annotations

from typing import Any, Iterable

import streamlit as st


def render_chat_turn(role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
    with st.chat_message(role):
        st.write(content)
        if metadata:
            with st.expander("Metadata"):
                st.json(metadata)


def render_chat_history(history: Iterable[dict[str, Any]]) -> None:
    for turn in history:
        render_chat_turn(
            role=turn.get("role", "assistant"),
            content=turn.get("content", ""),
            metadata=turn.get("metadata") or None,
        )
