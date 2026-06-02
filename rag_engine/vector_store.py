from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from rag_engine.schema import Document, RetrievalResult


class FaissVectorStore:
    """FAISS index wrapper with document metadata persistence."""

    def __init__(self, dimension: int = 384, metric: str = "cosine") -> None:
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.documents: list[Document] = []

    def _faiss(self):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss is required for dense vector search. Install faiss-cpu/faiss-gpu "
                "or use HybridRetriever(mode='sparse_only')."
            ) from exc
        return faiss

    def _ensure_index(self) -> None:
        if self.index is not None:
            return
        faiss = self._faiss()
        if self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors with dimension {self.dimension}, got {values.shape[1]}")
        if self.metric == "cosine":
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = values / np.maximum(norms, 1e-12)
        return np.ascontiguousarray(values, dtype=np.float32)

    def add_vectors(self, vectors: np.ndarray, documents: Iterable[Document | str | dict]) -> None:
        docs = [Document.from_any(item, index=len(self.documents) + i) for i, item in enumerate(documents)]
        if not docs:
            return
        prepared = self._prepare_vectors(vectors)
        if len(docs) != prepared.shape[0]:
            raise ValueError("Number of vectors must match number of documents")
        self._ensure_index()
        self.index.add(prepared)
        self.documents.extend(docs)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0 or not self.documents:
            return []
        self._ensure_index()
        prepared = self._prepare_vectors(query_vector)
        top_k = min(top_k, len(self.documents))
        scores, indices = self.index.search(prepared[:1], top_k)

        results: list[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(
                RetrievalResult(
                    document=self.documents[int(index)],
                    score=float(score),
                    dense_score=float(score),
                    metadata={"vector_index": int(index)},
                )
            )
        return results

    def save_index(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty FAISS index")
        faiss = self._faiss()
        target = Path(path)
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            index_path = target
            metadata_path = target.with_suffix(target.suffix + ".pkl")
        else:
            target.mkdir(parents=True, exist_ok=True)
            index_path = target / "index.faiss"
            metadata_path = target / "metadata.pkl"

        faiss.write_index(self.index, str(index_path))
        with metadata_path.open("wb") as fh:
            pickle.dump(
                {
                    "dimension": self.dimension,
                    "metric": self.metric,
                    "documents": self.documents,
                },
                fh,
            )

    @classmethod
    def load_index(cls, path: str | Path) -> "FaissVectorStore":
        target = Path(path)
        if target.suffix:
            index_path = target
            metadata_path = target.with_suffix(target.suffix + ".pkl")
        else:
            index_path = target / "index.faiss"
            metadata_path = target / "metadata.pkl"

        with metadata_path.open("rb") as fh:
            payload = pickle.load(fh)

        store = cls(dimension=int(payload["dimension"]), metric=str(payload["metric"]))
        faiss = store._faiss()
        store.index = faiss.read_index(str(index_path))
        store.documents = list(payload["documents"])
        return store
