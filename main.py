#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import mimetypes
import os
import pickle
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from rag_engine.embedder import EmbedderConfig, SentenceTransformerEmbedder
from rag_engine.retriever import HybridRetriever
from scripts import evaluate as system_eval
from scripts import evaluate_executor as executor_eval
from scripts import evaluate_planner as planner_eval


LOGGER = logging.getLogger("vn_stock_rag")


@dataclass(slots=True)
class RuntimeConfig:
    host: str
    port: int
    frontend_dir: Path
    processed_dir: Path
    chunks_file: Path
    index_dir: Path
    index_summary_path: Path
    local_files_only: bool
    load_in_4bit: bool
    device_map: str
    gpu_memory_limit: str | None
    planner_model_name_or_path: str
    planner_adapter_path: str
    planner_max_new_tokens: int
    planner_prompt_max_length: int
    planner_max_sources: int
    executor_model_name_or_path: str
    executor_adapter_path: str
    executor_max_new_tokens: int
    component_top_k: int
    context_window_chars: int
    max_context_chars: int
    candidate_top_k: int
    enable_gemini_fallback: bool
    fallback_model_name: str
    fallback_api_key: str | None
    fallback_temperature: float
    fallback_max_output_tokens: int
    fallback_thinking_budget: int
    fallback_context_chars: int
    fallback_max_snippets: int
    fallback_max_snippet_chars: int
    embedding_model_name: str

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        def env(name: str, default: str | None = None) -> str | None:
            return executor_eval.load_dotenv_value(name, default)

        def env_bool(name: str, default: bool) -> bool:
            raw = env(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in {"1", "true", "yes", "on", "có", "co"}

        def env_int(name: str, default: int) -> int:
            raw = env(name)
            return int(raw) if raw not in {None, ""} else default

        def env_float(name: str, default: float) -> float:
            raw = env(name)
            return float(raw) if raw not in {None, ""} else default

        return cls(
            host=env("APP_HOST", "0.0.0.0") or "0.0.0.0",
            port=env_int("APP_PORT", 8000),
            frontend_dir=PROJECT_ROOT / "frontend",
            processed_dir=PROJECT_ROOT / "data/processed_data",
            chunks_file=PROJECT_ROOT / (env("RAG_DOCUMENT_PATH", "data/chunks/chunks.json") or "data/chunks/chunks.json"),
            index_dir=PROJECT_ROOT / (env("RAG_INDEX_PATH", "data/index") or "data/index"),
            index_summary_path=PROJECT_ROOT / "data/metadata/index_summary.json",
            local_files_only=env_bool("LOCAL_FILES_ONLY", True),
            load_in_4bit=env_bool("LOAD_IN_4BIT", True),
            device_map=env("DEVICE_MAP", "auto") or "auto",
            gpu_memory_limit=env("GPU_MEMORY_LIMIT", None),
            planner_model_name_or_path=env("PHI_MODEL_NAME", env("PLANNER_MODEL_NAME", "models/phi")) or "models/phi",
            planner_adapter_path=env("PLANNER_ADAPTER_PATH", "models/phi_planner_lora") or "models/phi_planner_lora",
            planner_max_new_tokens=env_int("PLANNER_MAX_NEW_TOKENS", 256),
            planner_prompt_max_length=env_int("PLANNER_PROMPT_MAX_LENGTH", 1024),
            planner_max_sources=env_int("PLANNER_MAX_SOURCES", 12),
            executor_model_name_or_path=env("EXECUTOR_MODEL_NAME", "models/qwen") or "models/qwen",
            executor_adapter_path=env("EXECUTOR_ADAPTER_PATH", "models/qwen_executor_lora") or "models/qwen_executor_lora",
            executor_max_new_tokens=env_int("EXECUTOR_MAX_NEW_TOKENS", 128),
            component_top_k=env_int("RAG_EXECUTOR_TOP_K", 5),
            context_window_chars=env_int("RAG_CONTEXT_WINDOW_CHARS", 520),
            max_context_chars=env_int("RAG_MAX_CONTEXT_CHARS", 2200),
            candidate_top_k=env_int("RAG_TOP_K", 10),
            enable_gemini_fallback=env_bool("ENABLE_GEMINI_FALLBACK_EXECUTOR", True),
            fallback_model_name=env("GEMINI_MODEL_NAME", "gemini-2.5-flash") or "gemini-2.5-flash",
            fallback_api_key=env("GEMINI_API_KEY", None),
            fallback_temperature=env_float("GEMINI_FALLBACK_TEMPERATURE", 0.0),
            fallback_max_output_tokens=env_int("GEMINI_FALLBACK_MAX_OUTPUT_TOKENS", 64),
            fallback_thinking_budget=env_int("GEMINI_FALLBACK_THINKING_BUDGET", 0),
            fallback_context_chars=env_int("GEMINI_FALLBACK_CONTEXT_CHARS", 900),
            fallback_max_snippets=env_int("GEMINI_FALLBACK_MAX_SNIPPETS", 2),
            fallback_max_snippet_chars=env_int("GEMINI_FALLBACK_MAX_SNIPPET_CHARS", 220),
            embedding_model_name=env("EMBEDDING_MODEL_NAME", "models/embedder") or "models/embedder",
        )


class ChatRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.start_time = time.time()
        self.available_sources = planner_eval.list_processed_sources(
            self.config.processed_dir,
            max_sources=max(self.config.planner_max_sources, 200),
        )
        self.known_tickers = self._build_known_tickers(self.available_sources)
        self.chunk_map = executor_eval.load_chunk_map(self.config.chunks_file)
        self.index_summary = self._build_index_summary()
        self.text_cache: dict[Path, str] = {}
        self._retriever: HybridRetriever | None = None
        self._planner_runner: planner_eval.PhiPlannerRunner | None = None
        self._executor_runner: executor_eval.QwenExecutorRunner | None = None
        self._fallback_runner: executor_eval.GeminiFallbackRunner | None = None
        self._retriever_lock = threading.Lock()
        self._planner_lock = threading.Lock()
        self._executor_lock = threading.Lock()
        self._fallback_lock = threading.Lock()

    def get_health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "paths": {
                "processed_dir": str(self.config.processed_dir),
                "chunks_file": str(self.config.chunks_file),
                "index_dir": str(self.config.index_dir),
                "frontend_dir": str(self.config.frontend_dir),
            },
            "resources": {
                "processed_source_count": len(self.available_sources),
                "chunk_source_count": len(self.chunk_map),
                "index_ready": (self.config.index_dir / "retriever.pkl").exists(),
                "chunks_ready": self.config.chunks_file.exists(),
                "planner_loaded": self._planner_runner is not None,
                "executor_loaded": self._executor_runner is not None,
                "gemini_ready": bool(self.config.enable_gemini_fallback and self.config.fallback_api_key),
                "gemini_loaded": self._fallback_runner is not None,
                "retriever_loaded": self._retriever is not None,
            },
            "index_summary": self.index_summary,
        }

    def get_client_config(self) -> dict[str, Any]:
        return {
            "title": "VN Stock MAO ARAG",
            "planner": self.config.planner_model_name_or_path,
            "executor_primary": self.config.executor_model_name_or_path,
            "executor_fallback": self.config.fallback_model_name if self.config.enable_gemini_fallback else None,
            "fallback_enabled": bool(self.config.enable_gemini_fallback and self.config.fallback_api_key),
            "retrieval": {
                "index_dir": str(self.config.index_dir),
                "chunks_file": str(self.config.chunks_file),
                "component_top_k": self.config.component_top_k,
                "candidate_top_k": self.config.candidate_top_k,
                "summary": self.index_summary,
            },
            "index_files": self.build_index_manifest(),
        }

    def chat(self, message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        query = re.sub(r"\s+", " ", message).strip()
        if not query:
            raise ValueError("Câu hỏi đang trống.")
        ticker = self.infer_ticker(query)
        qa_type = self.infer_qa_type(query)
        candidate_hits = self.retrieve_candidate_sources(query, top_k=self.config.candidate_top_k)
        candidate_sources = [item["source_file"] for item in candidate_hits]
        source_hint = candidate_sources[0] if candidate_sources else self.guess_source_from_ticker(ticker)
        planner_item = {
            "id": "runtime-chat",
            "query": query,
            "qa_type": qa_type,
            "ticker": ticker,
            "source_file": source_hint or "",
            "available_sources": candidate_sources[: self.config.planner_max_sources] or self.available_sources[: self.config.planner_max_sources],
        }

        raw_plan = ""
        predicted_plan: dict[str, Any] | None = None
        planner_error = None
        try:
            raw_plan = self.get_planner().generate(planner_eval.format_prompt(planner_item))
            predicted_plan = planner_eval.extract_json(raw_plan)
        except Exception as exc:
            planner_error = str(exc)
            LOGGER.exception("Planner generation failed")

        selected_sources = self.select_sources(
            query=query,
            ticker=ticker,
            qa_type=qa_type,
            predicted_plan=predicted_plan,
            candidate_sources=candidate_sources,
        )
        source_docs, source_text = self.load_selected_sources(selected_sources)
        reference_component_queries = executor_eval.build_component_queries(query=query, qa_type=qa_type, ticker=ticker)
        planned_component_queries = system_eval.build_plan_component_queries(
            predicted_plan,
            planner_item,
            reference_component_queries=reference_component_queries,
        )
        component_results = executor_eval.evaluate_component_queries(
            component_queries=planned_component_queries,
            source_docs=source_docs,
            source_text=source_text,
            top_k=self.config.component_top_k,
            window_chars=self.config.context_window_chars,
        )
        context = executor_eval.build_context_from_components(
            component_results,
            max_context_chars=self.config.max_context_chars,
        )
        component_count_ok = system_eval.planned_component_count_ok(qa_type, component_results)
        component_support_rate, all_components_supported = executor_eval.component_support_metrics(component_results)
        executor_row = {
            "source_file": ", ".join(selected_sources) if selected_sources else "KHÔNG CHỌN ĐƯỢC NGUỒN",
            "ticker": ticker,
            "qa_type": qa_type,
            "component_results": component_results,
            "context": context,
            "query": query,
        }

        qwen_answer = self.get_executor().answer(executor_row)
        final_answer = qwen_answer
        executor_used = "qwen"
        routing_reason = "qwen_primary"
        fallback_answer = None
        fallback_used = False

        fallback_runner = self.get_fallback_runner()
        if fallback_runner is not None and self.should_use_live_fallback(
            query=query,
            qa_type=qa_type,
            qwen_answer=qwen_answer,
            component_results=component_results,
            component_count_ok=component_count_ok,
        ):
            fallback_answer = fallback_runner.answer(executor_row)
            final_answer = fallback_answer or final_answer
            executor_used = "gemini"
            fallback_used = True
            routing_reason = "gemini_fallback_after_qwen"

        return {
            "answer": final_answer,
            "query": query,
            "history_length": len(history or []),
            "qa_type": qa_type,
            "ticker": ticker,
            "executor_used": executor_used,
            "fallback_used": fallback_used,
            "routing_reason": routing_reason,
            "selected_sources": selected_sources,
            "candidate_sources": candidate_hits,
            "planner": {
                "raw_prediction": raw_plan,
                "predicted_plan": predicted_plan,
                "error": planner_error,
            },
            "executor": {
                "qwen_answer": qwen_answer,
                "gemini_answer": fallback_answer,
            },
            "retrieval": {
                "component_count_ok": component_count_ok,
                "component_support_rate": component_support_rate,
                "all_components_supported": all_components_supported,
                "component_results": self._compact_component_results(component_results),
                "context_preview": context[:800],
            },
        }

    def build_index_manifest(self) -> list[dict[str, Any]]:
        manifest = []
        if not self.config.index_dir.exists():
            return manifest
        for path in sorted(self.config.index_dir.rglob("*")):
            if not path.is_file():
                continue
            manifest.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "size_bytes": path.stat().st_size,
                }
            )
        return manifest

    def _build_index_summary(self) -> dict[str, Any]:
        summary = self._load_json(self.config.index_summary_path)
        retriever_path = self.config.index_dir / "retriever.pkl"
        if not retriever_path.exists():
            return summary
        try:
            with retriever_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            LOGGER.exception("Could not inspect %s", retriever_path)
            return summary

        config = payload.get("config")
        documents = payload.get("documents") or []
        retriever_summary = {
            "retrieval_mode": getattr(config, "mode", None),
            "dense_weight": getattr(config, "dense_weight", None),
            "sparse_weight": getattr(config, "sparse_weight", None),
            "dense_top_k": getattr(config, "dense_top_k", None),
            "sparse_top_k": getattr(config, "sparse_top_k", None),
            "output_top_k": getattr(config, "output_top_k", None),
            "chunk_count": len(documents),
            "index_dir": str(self.config.index_dir),
        }
        merged = {**summary, **{key: value for key, value in retriever_summary.items() if value is not None}}
        return merged

    def infer_ticker(self, query: str) -> str:
        matches = re.findall(r"\b[A-Z]{2,5}\b", query)
        for token in matches:
            if token in self.known_tickers:
                return token
        normalized = executor_eval.normalize_answer(query)
        for ticker in sorted(self.known_tickers):
            if executor_eval.normalize_answer(ticker) in normalized:
                return ticker
        return matches[0] if matches else ""

    def infer_qa_type(self, query: str) -> str:
        normalized = executor_eval.normalize_answer(query)
        multi_markers = (
            "chiem",
            "phan tram",
            "tang bao nhieu",
            "giam bao nhieu",
            "chenh lech",
            "so voi",
            "bao nhieu diem",
            "hai mang",
            "hai khoan",
        )
        return "multi_hop" if any(marker in normalized for marker in multi_markers) else "single_hop"

    def guess_source_from_ticker(self, ticker: str) -> str:
        if not ticker:
            return ""
        prefix = f"{ticker.upper()}-"
        for source in self.available_sources:
            if source.upper().startswith(prefix):
                return source
        return ""

    def retrieve_candidate_sources(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        retriever = self.get_retriever()
        hits: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        if retriever is None:
            return hits
        for result in retriever.search(query, top_k=top_k):
            source_file = executor_eval.canonical_source_name(
                str(
                    result.document.metadata.get("source_file")
                    or result.document.metadata.get("file_name")
                    or result.document.metadata.get("relative_path")
                    or result.document.id
                )
            )
            if not source_file or source_file in seen_sources:
                continue
            seen_sources.add(source_file)
            hits.append(
                {
                    "source_file": source_file,
                    "score": round(float(result.score), 6),
                    "snippet": self._compact_text(result.document.text, 220),
                }
            )
        return hits

    def select_sources(
        self,
        *,
        query: str,
        ticker: str,
        qa_type: str,
        predicted_plan: dict[str, Any] | None,
        candidate_sources: list[str],
    ) -> list[str]:
        selected_sources = system_eval.normalize_selected_sources(predicted_plan, self.available_sources)
        if selected_sources:
            return selected_sources
        if candidate_sources:
            limit = 2 if qa_type == "multi_hop" else 1
            return candidate_sources[:limit]
        guessed = self.guess_source_from_ticker(ticker)
        if guessed:
            return [guessed]
        fallback_hits = [item["source_file"] for item in self.retrieve_candidate_sources(query, top_k=2)]
        if fallback_hits:
            return fallback_hits
        return self.available_sources[:1]

    def load_selected_sources(self, selected_sources: list[str]) -> tuple[list[Any], str]:
        if not selected_sources:
            return [], ""
        source_docs, source_text = system_eval.load_selected_sources(
            selected_sources=selected_sources,
            processed_dir=self.config.processed_dir,
            chunk_map=self.chunk_map,
            text_cache=self.text_cache,
        )
        return source_docs, source_text

    def should_use_live_fallback(
        self,
        *,
        query: str,
        qa_type: str,
        qwen_answer: str,
        component_results: list[dict[str, Any]],
        component_count_ok: bool,
    ) -> bool:
        if executor_eval.is_not_found(qwen_answer):
            return True
        if qa_type != "multi_hop":
            return False
        if not component_count_ok:
            return False
        if not self.answer_shape_looks_valid(query, qwen_answer):
            retrieval_items = [item for item in component_results if str(item.get("type")) in {"retrieval", "retrieval_qa"}]
            return bool(retrieval_items) and all(item.get("supported") for item in retrieval_items[:2])
        return False

    def answer_shape_looks_valid(self, query: str, answer: str) -> bool:
        normalized_query = executor_eval.normalize_answer(query)
        normalized_answer = executor_eval.normalize_answer(answer)
        answer_numbers = executor_eval.extract_numbers(answer)
        if any(marker in normalized_query for marker in ("chiem", "phan tram")):
            return "%" in answer or bool(answer_numbers)
        if any(marker in normalized_query for marker in ("tang bao nhieu", "giam bao nhieu", "chenh lech", "so voi", "bao nhieu diem")):
            return bool(answer_numbers)
        if " va " in normalized_query or " hai " in normalized_query:
            return len(answer_numbers) >= 2 or " và " in answer.lower()
        return bool(normalized_answer)

    def get_retriever(self) -> HybridRetriever | None:
        if self._retriever is not None:
            return self._retriever
        if not (self.config.index_dir / "retriever.pkl").exists():
            return None
        with self._retriever_lock:
            if self._retriever is not None:
                return self._retriever
            embedder = SentenceTransformerEmbedder(
                EmbedderConfig(
                    model_name=str(PROJECT_ROOT / self.config.embedding_model_name)
                    if not Path(self.config.embedding_model_name).is_absolute()
                    else self.config.embedding_model_name,
                    local_files_only=self.config.local_files_only,
                )
            )
            self._retriever = HybridRetriever.load(self.config.index_dir, embedder=embedder)
            LOGGER.info("Loaded retriever from %s", self.config.index_dir)
        return self._retriever

    def get_planner(self) -> planner_eval.PhiPlannerRunner:
        if self._planner_runner is not None:
            return self._planner_runner
        with self._planner_lock:
            if self._planner_runner is None:
                args = self._build_model_args(
                    model_name_or_path=self.config.planner_model_name_or_path,
                    adapter_path=self.config.planner_adapter_path,
                    max_new_tokens=self.config.planner_max_new_tokens,
                    prompt_max_length=self.config.planner_prompt_max_length,
                )
                self._planner_runner = planner_eval.PhiPlannerRunner(args)
                LOGGER.info("Loaded planner model %s", self.config.planner_model_name_or_path)
        return self._planner_runner

    def get_executor(self) -> executor_eval.QwenExecutorRunner:
        if self._executor_runner is not None:
            return self._executor_runner
        with self._executor_lock:
            if self._executor_runner is None:
                args = self._build_model_args(
                    model_name_or_path=self.config.executor_model_name_or_path,
                    adapter_path=self.config.executor_adapter_path,
                    max_new_tokens=self.config.executor_max_new_tokens,
                )
                self._executor_runner = executor_eval.QwenExecutorRunner(args)
                LOGGER.info("Loaded executor model %s", self.config.executor_model_name_or_path)
        return self._executor_runner

    def get_fallback_runner(self) -> executor_eval.GeminiFallbackRunner | None:
        if not (self.config.enable_gemini_fallback and self.config.fallback_api_key):
            return None
        if self._fallback_runner is not None:
            return self._fallback_runner
        with self._fallback_lock:
            if self._fallback_runner is None:
                args = type(
                    "GeminiArgs",
                    (),
                    {
                        "fallback_api_key": self.config.fallback_api_key,
                        "fallback_model_name": self.config.fallback_model_name,
                        "fallback_temperature": self.config.fallback_temperature,
                        "fallback_max_output_tokens": self.config.fallback_max_output_tokens,
                        "fallback_thinking_budget": self.config.fallback_thinking_budget,
                        "fallback_context_chars": self.config.fallback_context_chars,
                        "fallback_max_snippets": self.config.fallback_max_snippets,
                        "fallback_max_snippet_chars": self.config.fallback_max_snippet_chars,
                    },
                )()
                self._fallback_runner = executor_eval.GeminiFallbackRunner(args)
                LOGGER.info("Loaded Gemini fallback %s", self.config.fallback_model_name)
        return self._fallback_runner

    def _build_model_args(
        self,
        *,
        model_name_or_path: str,
        adapter_path: str,
        max_new_tokens: int,
        prompt_max_length: int | None = None,
    ) -> Any:
        payload = {
            "model_name_or_path": model_name_or_path,
            "adapter_path": adapter_path,
            "local_files_only": self.config.local_files_only,
            "load_in_4bit": self.config.load_in_4bit,
            "device_map": self.config.device_map,
            "gpu_memory_limit": self.config.gpu_memory_limit,
            "max_new_tokens": max_new_tokens,
        }
        if prompt_max_length is not None:
            payload["prompt_max_length"] = prompt_max_length
        return type("Args", (), payload)()

    def _compact_component_results(self, component_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compacted = []
        for item in component_results:
            compacted.append(
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "query": item.get("query"),
                    "supported": item.get("supported"),
                    "top_score": item.get("top_score"),
                    "evidence_snippets": [
                        {
                            "source": evidence.get("source"),
                            "mode": evidence.get("mode"),
                            "score": evidence.get("score"),
                            "snippet": self._compact_text(str(evidence.get("snippet") or ""), 260),
                        }
                        for evidence in (item.get("evidence_snippets") or [])[:2]
                    ],
                }
            )
        return compacted

    @staticmethod
    def _compact_text(text: str, max_chars: int) -> str:
        value = re.sub(r"\s+", " ", text).strip()
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "..."

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _build_known_tickers(source_names: list[str]) -> set[str]:
        tickers = set()
        for source_name in source_names:
            match = re.match(r"([A-Z0-9]{2,6})[-_]", Path(source_name).name)
            if match:
                tickers.add(match.group(1))
        return tickers


class AppHandler(BaseHTTPRequestHandler):
    server_version = "VNStockRAGHTTP/1.0"

    @property
    def runtime(self) -> ChatRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    @property
    def config(self) -> RuntimeConfig:
        return self.server.config  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json(HTTPStatus.OK, self.runtime.get_health())
            return
        if parsed.path == "/api/config":
            self._send_json(HTTPStatus.OK, self.runtime.get_client_config())
            return
        self._serve_frontend(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy endpoint."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "JSON không hợp lệ."})
            return

        message = str(payload.get("message") or "").strip()
        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        if not message:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Thiếu trường message."})
            return

        try:
            response = self.runtime.chat(message, history)
        except Exception as exc:
            LOGGER.exception("Chat request failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": str(exc),
                    "answer": "Không xử lý được câu hỏi. Kiểm tra log backend để biết chi tiết.",
                },
            )
            return

        self._send_json(HTTPStatus.OK, response)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def _serve_frontend(self, request_path: str) -> None:
        requested = Path(unquote(request_path.lstrip("/")))
        if request_path in {"", "/"}:
            requested = Path("index.html")
        candidate = (self.config.frontend_dir / requested).resolve()
        frontend_root = self.config.frontend_dir.resolve()
        if frontend_root not in candidate.parents and candidate != frontend_root / "index.html":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy file frontend."})
            return
        if not candidate.exists() or not candidate.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy file frontend."})
            return
        content_type, _ = mimetypes.guess_type(candidate.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(candidate.read_bytes())

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.client_address[0], format % args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = RuntimeConfig.from_env()
    runtime = ChatRuntime(config)
    server = ThreadingHTTPServer((config.host, config.port), AppHandler)
    server.runtime = runtime  # type: ignore[attr-defined]
    server.config = config  # type: ignore[attr-defined]
    LOGGER.info("Serving frontend from %s", config.frontend_dir)
    LOGGER.info("API listening on http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
