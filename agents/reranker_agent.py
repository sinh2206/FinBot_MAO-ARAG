from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from rag_engine.schema import RetrievalResult


@dataclass(slots=True)
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str | None = None
    batch_size: int = 16
    local_files_only: bool = False
    enable_model: bool = True
    fallback_to_input_order: bool = True


class RerankerAgent:
    """Cross-encoder reranker. No LLM is used here."""

    def __init__(self, config: RerankerConfig | None = None, **kwargs: Any) -> None:
        self.config = config or RerankerConfig(**kwargs)
        self._model = None

    def rerank(
        self,
        query: str,
        passages: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not passages:
            return []
        if not self.config.enable_model:
            return passages[:top_k] if top_k else passages

        try:
            model = self._load_model()
            pairs = [(query, item.document.text) for item in passages]
            scores = model.predict(pairs, batch_size=self.config.batch_size)
            reranked = [
                replace(
                    item,
                    score=float(score),
                    metadata={**item.metadata, "retrieval_score": item.score, "rerank_score": float(score)},
                )
                for item, score in zip(passages, scores)
            ]
            reranked.sort(key=lambda item: item.score, reverse=True)
            return reranked[:top_k] if top_k else reranked
        except Exception:
            if not self.config.fallback_to_input_order:
                raise
            return passages[:top_k] if top_k else passages

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for reranker_agent") from exc

        self._model = CrossEncoder(
            self.config.model_name,
            device=self.config.device,
            tokenizer_args={"local_files_only": self.config.local_files_only},
            automodel_args={"local_files_only": self.config.local_files_only},
        )
        return self._model
