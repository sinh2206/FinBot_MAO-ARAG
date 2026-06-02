from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rag_engine.retriever import HybridRetriever
from rag_engine.schema import Document, RetrievalResult


@dataclass(slots=True)
class RetrieverAgentConfig:
    top_k: int = 10
    dense_top_k: int = 50
    sparse_top_k: int = 50


class RetrieverAgent:
    """Retriever agent without LLM: FAISS + BM25 through HybridRetriever."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        documents: Iterable[Document | str | dict] | None = None,
        config: RetrieverAgentConfig | None = None,
    ) -> None:
        self.config = config or RetrieverAgentConfig()
        self.retriever = retriever or HybridRetriever(documents=documents)

    def index_documents(self, documents: Iterable[Document | str | dict]) -> None:
        self.retriever.index_documents(documents)

    def retrieve(self, sub_query: str, top_k: int | None = None) -> list[RetrievalResult]:
        return self.retriever.search(
            sub_query,
            top_k=top_k or self.config.top_k,
            dense_top_k=self.config.dense_top_k,
            sparse_top_k=self.config.sparse_top_k,
        )
