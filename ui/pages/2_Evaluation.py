from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from tools.evaluation import evaluate_qa_pairs


def main() -> None:
    st.set_page_config(page_title="Evaluation", layout="wide")
    st.title("Evaluation")
    st.caption("Nap file JSON gom cac field prediction va answer de tinh EM/F1 nhanh.")

    default_path = st.text_input("Duong dan file ket qua", "data/metadata/evaluation_results.json")
    if st.button("Chay danh gia"):
        path = Path(default_path)
        if not path.exists():
            st.error(f"Khong tim thay file: {path}")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            st.error("File danh gia phai la list JSON.")
            return
        result = evaluate_qa_pairs(payload)
        st.json(result.to_dict())


if __name__ == "__main__":
    main()
