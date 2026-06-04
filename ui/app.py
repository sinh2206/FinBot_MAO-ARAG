from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.reranker_agent import RerankerAgent
from agents.retriever_agent import RetrieverAgent
from core.orchestrator import Orchestrator, OrchestratorConfig
from rag_engine.embedder import EmbedderConfig, SentenceTransformerEmbedder
from rag_engine.retriever import HybridRetriever, HybridRetrieverConfig
from rag_engine.schema import Document


load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def load_documents(path: str) -> list[Document]:
    source = Path(path)
    if not source.exists():
        return demo_documents()

    if source.suffix.lower() == ".jsonl":
        documents: list[Document] = []
        for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            payload = json.loads(line)
            documents.append(Document.from_any(payload, index=index))
        return documents or demo_documents()

    if source.suffix.lower() == ".json":
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [Document.from_any(item, index=i) for i, item in enumerate(payload)]
        return [Document.from_any(payload, index=0)]

    text = source.read_text(encoding="utf-8", errors="ignore")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return [Document(id=f"chunk_{i + 1}", text=chunk) for i, chunk in enumerate(chunks)] or demo_documents()


def demo_documents() -> list[Document]:
    return [
        Document(id="demo_1", text="VNINDEX dong cua o 1.280 diem, tang 12 diem so voi phien truoc."),
        Document(id="demo_2", text="Thanh khoan thi truong dat 18.000 ty dong, cao hon muc trung binh 20 phien."),
        Document(id="demo_3", text="Nhom ngan hang dan dat da tang voi nhieu co phieu vuot tham chieu."),
    ]


def load_retriever(documents: list[Document]) -> HybridRetriever:
    index_path = Path(os.getenv("RAG_INDEX_PATH", "data/index"))
    if (index_path / "retriever.pkl").exists():
        try:
            return HybridRetriever.load(index_path)
        except Exception as exc:
            st.warning(f"Khong load duoc index da luu, se dung corpus truc tiep: {exc}")

    mode = os.getenv("RAG_RETRIEVAL_MODE", "sparse_only")
    embedder = SentenceTransformerEmbedder(
        EmbedderConfig(model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    )
    return HybridRetriever(
        documents=documents,
        embedder=embedder,
        config=HybridRetrieverConfig(
            mode=mode,
            dense_weight=env_float("RAG_DENSE_WEIGHT", 0.65),
            sparse_weight=env_float("RAG_SPARSE_WEIGHT", 0.35),
            output_top_k=env_int("RAG_TOP_K", 10),
        ),
    )


@st.cache_resource(show_spinner="Dang khoi tao RAG engine...")
def build_orchestrator() -> Orchestrator:
    documents = load_documents(os.getenv("RAG_DOCUMENT_PATH", "data/chunks/chunks.json"))
    retriever = load_retriever(documents)

    return Orchestrator(
        planner_agent=PlannerAgent(
            model_name=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            enable_llm=env_bool("ENABLE_LOCAL_PLANNER", False),
            local_files_only=env_bool("LOCAL_FILES_ONLY", True),
            load_in_4bit=env_bool("LOAD_IN_4BIT", False),
        ),
        retriever_agent=RetrieverAgent(retriever=retriever),
        reranker_agent=RerankerAgent(
            model_name=os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            enable_model=env_bool("ENABLE_RERANKER", False),
        ),
        executor_agent=ExecutorAgent(
            model_name=os.getenv(
                "EXECUTOR_MODEL_NAME",
                os.getenv("MINIMAX_MODEL_NAME", "LiquidAI/LFM2-1.2B-RAG"),
            ),
            enable_model=env_bool("ENABLE_LOCAL_EXECUTOR", False),
            local_files_only=env_bool("LOCAL_FILES_ONLY", True),
            load_in_4bit=env_bool("LOAD_IN_4BIT", False),
        ),
        config=OrchestratorConfig(
            retriever_top_k=env_int("RAG_TOP_K", 10),
            executor_top_k=env_int("RAG_EXECUTOR_TOP_K", 5),
        ),
        documents=documents,
    )


def render_debug(result: dict[str, Any]) -> None:
    with st.expander("Chi tiet workflow"):
        st.json(result["plan"])

    with st.expander("Sub-queries va cau tra loi con"):
        st.json(result["sub_answers"])

    with st.expander("Doan van duoc chon va diem so"):
        rows = []
        for sub_answer in result["sub_answers"]:
            for context in sub_answer.get("contexts", []):
                document = context.get("document", {})
                rows.append(
                    {
                        "sub_query": sub_answer.get("query"),
                        "doc_id": document.get("id"),
                        "score": context.get("score"),
                        "dense_score": context.get("dense_score"),
                        "sparse_score": context.get("sparse_score"),
                        "text": document.get("text"),
                    }
                )
        st.dataframe(rows, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="MAO-RAG Offline", layout="wide")
    st.title("MAO-RAG Offline")
    st.caption("Streamlit UI cho luong planner, retriever, reranker, executor va aggregator chay local.")

    with st.sidebar:
        st.subheader("Cau hinh")
        st.write(f"Retrieval mode: `{os.getenv('RAG_RETRIEVAL_MODE', 'sparse_only')}`")
        st.write(f"Index path: `{os.getenv('RAG_INDEX_PATH', 'data/index')}`")
        st.write(f"Document path: `{os.getenv('RAG_DOCUMENT_PATH', 'data/chunks/chunks.json')}`")
        show_debug = st.checkbox("Hien thi chi tiet", value=True)

    orchestrator = build_orchestrator()
    question = st.text_area("Nhap cau hoi", value="VNINDEX dong cua bao nhieu diem?", height=100)

    if st.button("Gui", type="primary") and question.strip():
        with st.spinner("Dang xu ly..."):
            result = orchestrator.run(question)

        st.subheader("Cau tra loi")
        st.write(result["answer"])

        if show_debug:
            render_debug(result)
