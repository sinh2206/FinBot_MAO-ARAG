from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from rag_engine.embedder import SentenceTransformerEmbedder
from rag_engine.schema import Document, RetrievalResult
from rag_engine.vector_store import FaissVectorStore


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, documents: Iterable[Document] = (), k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        docs = list(documents)
        if docs:
            self.fit(docs)

    def fit(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        self.doc_tokens = [tokenize(doc.text) for doc in self.documents]
        self.doc_freq = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

        total_docs = max(len(self.documents), 1)
        self.avgdl = sum(len(tokens) for tokens in self.doc_tokens) / total_docs
        self.idf = {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in self.doc_freq.items()
        }

    def scores(self, query: str) -> np.ndarray:
        if not self.documents:
            return np.empty((0,), dtype=np.float32)
        query_terms = tokenize(query)
        values = np.zeros(len(self.documents), dtype=np.float32)
        if not query_terms:
            return values

        for index, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue
            frequencies = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                freq = frequencies.get(term, 0)
                if freq == 0:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-12))
                score += self.idf.get(term, 0.0) * numerator / denominator
            values[index] = score
        return values

    def search(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        scores = self.scores(query)
        if scores.size == 0 or top_k <= 0:
            return []
        top_indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        return [
            RetrievalResult(
                document=self.documents[int(index)],
                score=float(scores[index]),
                sparse_score=float(scores[index]),
                metadata={"bm25_index": int(index)},
            )
            for index in top_indices
            if scores[index] > 0
        ]


@dataclass(slots=True)
class HybridRetrieverConfig:
    mode: str = "hybrid"
    dense_weight: float = 0.65
    sparse_weight: float = 0.35
    dense_top_k: int = 50
    sparse_top_k: int = 50
    output_top_k: int = 10


class HybridRetriever:
    """Combines dense FAISS search and sparse BM25 retrieval."""

    def __init__(
        self,
        documents: Iterable[Document | str | dict] | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
        vector_store: FaissVectorStore | None = None,
        config: HybridRetrieverConfig | None = None,
        **kwargs: object,
    ) -> None:
        self.config = config or HybridRetrieverConfig(**kwargs)
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.vector_store = vector_store or FaissVectorStore(dimension=self.embedder.dimension)
        self.documents: list[Document] = []
        self.bm25 = BM25Index()
        if documents:
            self.index_documents(documents)

    def index_documents(self, documents: Iterable[Document | str | dict]) -> None:
        self.documents = [Document.from_any(item, index=i) for i, item in enumerate(documents)]
        self.bm25.fit(self.documents)
        if self.config.mode in {"hybrid", "dense_only"} and self.documents:
            vectors = self.embedder.encode([doc.text for doc in self.documents])
            self.vector_store = FaissVectorStore(dimension=vectors.shape[1])
            self.vector_store.add_vectors(vectors, self.documents)

    def _dense_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.config.mode == "sparse_only" or not self.documents:
            return []
        query_vector = self.embedder.encode(query)
        return self.vector_store.search(query_vector, top_k=top_k)

    def _sparse_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.config.mode == "dense_only":
            return []
        return self.bm25.search(query, top_k=top_k)

    @staticmethod
    def _normalize(results: list[RetrievalResult], attr: str) -> dict[str, float]:
        if not results:
            return {}
        raw_scores = [float(getattr(item, attr) or item.score) for item in results]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        denominator = max(max_score - min_score, 1e-12)
        return {
            item.document.id: (float(getattr(item, attr) or item.score) - min_score) / denominator
            if max_score != min_score
            else 1.0
            for item in results
        }

    def search(
        self,
        query: str,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        output_top_k = top_k or self.config.output_top_k
        dense_results = self._dense_search(query, dense_top_k or self.config.dense_top_k)
        sparse_results = self._sparse_search(query, sparse_top_k or self.config.sparse_top_k)

        if self.config.mode == "dense_only":
            return dense_results[:output_top_k]
        if self.config.mode == "sparse_only":
            return sparse_results[:output_top_k]

        dense_norm = self._normalize(dense_results, "dense_score")
        sparse_norm = self._normalize(sparse_results, "sparse_score")
        by_id: dict[str, RetrievalResult] = {}
        for item in [*dense_results, *sparse_results]:
            current = by_id.get(item.document.id)
            if current is None:
                by_id[item.document.id] = item
                continue
            current.dense_score = current.dense_score if current.dense_score is not None else item.dense_score
            current.sparse_score = current.sparse_score if current.sparse_score is not None else item.sparse_score
            current.metadata.update(item.metadata)

        combined: list[RetrievalResult] = []
        for doc_id, item in by_id.items():
            score = (
                self.config.dense_weight * dense_norm.get(doc_id, 0.0)
                + self.config.sparse_weight * sparse_norm.get(doc_id, 0.0)
            )
            combined.append(
                RetrievalResult(
                    document=item.document,
                    score=float(score),
                    dense_score=item.dense_score,
                    sparse_score=item.sparse_score,
                    metadata={**item.metadata, "retrieval_mode": "hybrid"},
                )
            )

        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:output_top_k]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if self.config.mode in {"hybrid", "dense_only"}:
            self.vector_store.save_index(target / "faiss")
        with (target / "retriever.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "config": self.config,
                    "documents": self.documents,
                    "bm25": self.bm25,
                },
                fh,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: SentenceTransformerEmbedder | None = None,
    ) -> "HybridRetriever":
        target = Path(path)
        with (target / "retriever.pkl").open("rb") as fh:
            payload = pickle.load(fh)
        vector_store = None
        if payload["config"].mode in {"hybrid", "dense_only"}:
            vector_store = FaissVectorStore.load_index(target / "faiss")
        instance = cls(embedder=embedder, vector_store=vector_store, config=payload["config"])
        instance.documents = payload["documents"]
        instance.bm25 = payload["bm25"]
        return instance
