from __future__ import annotations

from pathlib import Path

import streamlit as st


DOCUMENT_DIR = Path("data/documents")


def main() -> None:
    st.set_page_config(page_title="Document Manager", layout="wide")
    st.title("Document Manager")
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)

    uploaded = st.file_uploader(
        "Upload tai lieu offline",
        type=["txt", "md", "markdown", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Luu tai lieu", type="primary"):
        for file in uploaded:
            target = DOCUMENT_DIR / file.name
            target.write_bytes(file.getbuffer())
        st.success(f"Da luu {len(uploaded)} file vao {DOCUMENT_DIR}")

    st.subheader("Danh sach tai lieu")
    files = sorted(path for path in DOCUMENT_DIR.glob("*") if path.is_file())
    if not files:
        st.info("Chua co tai lieu trong data/documents.")
        return

    for path in files:
        cols = st.columns([4, 1, 1])
        cols[0].write(path.name)
        cols[1].write(f"{path.stat().st_size / 1024:.1f} KB")
        if cols[2].button("Xoa", key=f"delete_{path.name}"):
            path.unlink(missing_ok=True)
            st.rerun()


if __name__ == "__main__":
    main()
