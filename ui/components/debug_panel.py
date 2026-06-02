from __future__ import annotations

from typing import Any

import streamlit as st


def render_debug_panel(result: dict[str, Any]) -> None:
    with st.expander("Workflow"):
        st.json(result.get("plan", {}))

    with st.expander("Sub-answers"):
        st.json(result.get("sub_answers", []))

    with st.expander("Retrieved passages"):
        rows = []
        for sub_answer in result.get("sub_answers", []):
            for context in sub_answer.get("contexts", []):
                document = context.get("document", {})
                rows.append(
                    {
                        "query": sub_answer.get("query"),
                        "doc_id": document.get("id"),
                        "score": context.get("score"),
                        "dense_score": context.get("dense_score"),
                        "sparse_score": context.get("sparse_score"),
                        "text": document.get("text"),
                    }
                )
        st.dataframe(rows, use_container_width=True)
