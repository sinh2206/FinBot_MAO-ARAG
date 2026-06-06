from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "co", "có"}:
            return True
        if normalized in {"0", "false", "no", "off", "khong", "không"}:
            return False
    return bool(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None and value.strip() != "":
        return value.strip()
    return default


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass(slots=True)
class BackendSettings:
    project_root: Path
    frontend_dir: Path
    processed_dir: Path
    chunks_file: Path
    index_dir: Path
    metadata_dir: Path
    model_config_path: Path
    agent_config_path: Path
    retrieval_config_path: Path
    planner_model_name: str
    executor_model_name: str
    fallback_model_name: str
    fallback_api_key: str | None
    enable_local_planner: bool
    enable_local_executor: bool
    enable_gemini_fallback: bool
    local_files_only: bool
    load_in_4bit: bool
    component_top_k: int
    context_window_chars: int
    max_context_chars: int
    cors_origins: list[str]
    model_config: dict[str, Any]
    agent_config: dict[str, Any]
    retrieval_config: dict[str, Any]
    index_summary_path: Path

    @classmethod
    def from_env(cls) -> "BackendSettings":
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env", override=False)
        model_config = _load_yaml(project_root / "config" / "model_config.yaml")
        agent_config = _load_yaml(project_root / "config" / "agent_config.yaml")
        retrieval_config = _load_yaml(project_root / "config" / "retrieval_config.yaml")

        model_models = model_config.get("models", {}) if isinstance(model_config.get("models"), dict) else {}
        agent_planner = agent_config.get("planner", {}) if isinstance(agent_config.get("planner"), dict) else {}
        agent_executor = agent_config.get("executor", {}) if isinstance(agent_config.get("executor"), dict) else {}
        agent_retriever = agent_config.get("retriever", {}) if isinstance(agent_config.get("retriever"), dict) else {}
        retrieval_section = retrieval_config.get("retrieval", {}) if isinstance(retrieval_config.get("retrieval"), dict) else {}
        paths_section = retrieval_config.get("paths", {}) if isinstance(retrieval_config.get("paths"), dict) else {}

        planner_model_name = _env(
            "PLANNER_MODEL_NAME",
            str(
                agent_planner.get("model_name")
                or model_models.get("planner", {}).get("local_dir")
                or "models/phi"
            ),
        )
        executor_model_name = _env(
            "EXECUTOR_MODEL_NAME",
            str(
                agent_executor.get("primary", {}).get("model_name")
                or model_models.get("executor_primary", {}).get("local_dir")
                or "models/qwen"
            ),
        )
        fallback_model_name = _env(
            "GEMINI_MODEL_NAME",
            str(
                agent_executor.get("fallback", {}).get("model_name")
                or model_models.get("executor_fallback", {}).get("model_name")
                or "gemini-2.5-flash"
            ),
        )

        processed_dir = _resolve_path(project_root, str(_env("PROCESSED_DIR", paths_section.get("processed_data_dir", "data/processed_data"))))
        chunks_override = _env("CHUNKS_FILE")
        if chunks_override:
            chunks_path = Path(chunks_override)
            chunks_file = chunks_path if chunks_path.suffix.lower() == ".json" else chunks_path / "chunks.json"
        else:
            chunks_file = _resolve_path(project_root, str(paths_section.get("chunks_dir", "data/chunks"))) / "chunks.json"
        index_dir = _resolve_path(project_root, str(_env("INDEX_DIR", paths_section.get("index_dir", "data/index"))))
        metadata_dir = _resolve_path(project_root, str(_env("METADATA_DIR", paths_section.get("metadata_dir", "data/metadata"))))

        cors_raw = _env("CORS_ALLOW_ORIGINS", "*") or "*"
        cors_origins = [item.strip() for item in cors_raw.split(",") if item.strip()] or ["*"]

        return cls(
            project_root=project_root,
            frontend_dir=project_root / "frontend",
            processed_dir=processed_dir,
            chunks_file=chunks_file,
            index_dir=index_dir,
            metadata_dir=metadata_dir,
            model_config_path=project_root / "config" / "model_config.yaml",
            agent_config_path=project_root / "config" / "agent_config.yaml",
            retrieval_config_path=project_root / "config" / "retrieval_config.yaml",
            planner_model_name=planner_model_name,
            executor_model_name=executor_model_name,
            fallback_model_name=fallback_model_name,
            fallback_api_key=_env("GEMINI_API_KEY", _env("GEMINI_FALLBACK_API_KEY")),
            enable_local_planner=_as_bool(_env("ENABLE_LOCAL_PLANNER", str(agent_planner.get("provider", "local") == "local")), True),
            enable_local_executor=_as_bool(_env("ENABLE_LOCAL_EXECUTOR", str(agent_executor.get("primary", {}).get("provider", "local") == "local")), True),
            enable_gemini_fallback=_as_bool(
                _env(
                    "ENABLE_GEMINI_FALLBACK_EXECUTOR",
                    _env("ENABLE_GEMINI_API", str(agent_executor.get("fallback", {}).get("provider", "gemini") == "gemini")),
                ),
                True,
            ),
            local_files_only=_as_bool(_env("LOCAL_FILES_ONLY", str(agent_planner.get("local_files_only", True))), True),
            load_in_4bit=_as_bool(_env("LOAD_IN_4BIT", str(agent_executor.get("primary", {}).get("quantize", "") == "4bit")), True),
            component_top_k=int(_env("RAG_EXECUTOR_TOP_K", str(retrieval_section.get("executor_top_k", 5))) or 5),
            context_window_chars=int(_env("CONTEXT_WINDOW_CHARS", "480") or "480"),
            max_context_chars=int(_env("MAX_CONTEXT_CHARS", "1800") or "1800"),
            cors_origins=cors_origins,
            model_config=model_config,
            agent_config=agent_config,
            retrieval_config=retrieval_config,
            index_summary_path=project_root / "data" / "metadata" / "index_summary.json",
        )
