from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class EmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None
    normalize_embeddings: bool = True
    local_files_only: bool = False


class SentenceTransformerEmbedder:
    """Thin wrapper around sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(self, config: EmbedderConfig | None = None, **kwargs: object) -> None:
        self.config = config or EmbedderConfig(**kwargs)
        self._model = None

    @property
    def dimension(self) -> int:
        return 384

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for dense embeddings. "
                "Install it or use HybridRetriever(mode='sparse_only')."
            ) from exc

        try:
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                local_files_only=self.config.local_files_only,
            )
        except TypeError:
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model

    def encode(self, texts: str | Iterable[str], batch_size: int = 32) -> np.ndarray:
        is_single = isinstance(texts, str)
        values = [texts] if is_single else list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._load_model()
        vectors = model.encode(
            values,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors
