from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw_data"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed_data"
    documents_dir: Path = PROJECT_ROOT / "data" / "processed_data"
    chunks_dir: Path = PROJECT_ROOT / "data" / "chunks"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    embeddings_dir: Path = PROJECT_ROOT / "data" / "embeddings"
    metadata_dir: Path = PROJECT_ROOT / "data" / "metadata"
    models_dir: Path = PROJECT_ROOT / "models"
    config_dir: Path = PROJECT_ROOT / "config"


@dataclass(slots=True)
class RetrievalSettings:
    mode: str = "sparse_only"
    top_k: int = 10
    executor_top_k: int = 5
    dense_top_k: int = 50
    sparse_top_k: int = 50
    dense_weight: float = 0.65
    sparse_weight: float = 0.35
    chunk_size: int = 384
    chunk_overlap_ratio: float = 0.2


@dataclass(slots=True)
class ModelSettings:
    qwen_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    executor_model_name: str = "LiquidAI/LFM2-1.2B-RAG"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    local_files_only: bool = True
    device_map: str | None = "auto"


@dataclass(slots=True)
class AppConfig:
    paths: PathConfig
    retrieval: RetrievalSettings
    models: ModelSettings

    @classmethod
    def from_env(cls) -> "AppConfig":
        paths = PathConfig()
        return cls(
            paths=paths,
            retrieval=RetrievalSettings(
                mode=os.getenv("RAG_RETRIEVAL_MODE", "sparse_only"),
                top_k=_env_int("RAG_TOP_K", 10),
                executor_top_k=_env_int("RAG_EXECUTOR_TOP_K", 5),
                dense_weight=_env_float("RAG_DENSE_WEIGHT", 0.65),
                sparse_weight=_env_float("RAG_SPARSE_WEIGHT", 0.35),
                chunk_size=_env_int("RAG_CHUNK_SIZE", 384),
                chunk_overlap_ratio=_env_float("RAG_CHUNK_OVERLAP_RATIO", 0.2),
            ),
            models=ModelSettings(
                qwen_model_name=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
                executor_model_name=os.getenv(
                    "EXECUTOR_MODEL_NAME",
                    os.getenv("MINIMAX_MODEL_NAME", "LiquidAI/LFM2-1.2B-RAG"),
                ),
                embedding_model_name=os.getenv(
                    "EMBEDDING_MODEL_NAME",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ),
                reranker_model_name=os.getenv(
                    "RERANKER_MODEL_NAME",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ),
                local_files_only=_env_bool("LOCAL_FILES_ONLY", True),
            ),
        )

    def ensure_directories(self) -> None:
        for path in [
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.documents_dir,
            self.paths.chunks_dir,
            self.paths.index_dir,
            self.paths.embeddings_dir,
            self.paths.metadata_dir,
            self.paths.models_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load YAML config files") from exc
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {source}")
    return payload


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
