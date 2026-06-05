from __future__ import annotations

from rag_engine.schema import Answer, Document, RetrievalResult, SubQuery, WorkflowPlan


__all__ = [
    "Answer",
    "Document",
    "DocumentProcessor",
    "DocumentProcessorConfig",
    "FaissVectorStore",
    "HybridRetriever",
    "IndexBuildConfig",
    "IndexManager",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "SubQuery",
    "WorkflowPlan",
]


def __getattr__(name: str) -> object:
    if name in {"DocumentProcessor", "DocumentProcessorConfig"}:
        from rag_engine.document_processor import DocumentProcessor, DocumentProcessorConfig

        return {
            "DocumentProcessor": DocumentProcessor,
            "DocumentProcessorConfig": DocumentProcessorConfig,
        }[name]
    if name == "SentenceTransformerEmbedder":
        from rag_engine.embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder
    if name in {"IndexBuildConfig", "IndexManager"}:
        from rag_engine.index_manager import IndexBuildConfig, IndexManager

        return {"IndexBuildConfig": IndexBuildConfig, "IndexManager": IndexManager}[name]
    if name == "HybridRetriever":
        from rag_engine.retriever import HybridRetriever

        return HybridRetriever
    if name == "FaissVectorStore":
        from rag_engine.vector_store import FaissVectorStore

        return FaissVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
