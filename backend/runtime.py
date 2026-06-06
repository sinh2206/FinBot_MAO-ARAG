from __future__ import annotations

import json
import logging
import re
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rag_engine import DocumentProcessor, DocumentProcessorConfig, IndexManager
from rag_engine.retriever import BM25Index
from rag_engine.schema import Document
from scripts.evaluate_executor import (
    GeminiFallbackRunner,
    QwenExecutorRunner,
    build_component_queries,
    build_context_from_components,
    clean_prediction,
    component_support_metrics,
    evaluate_component_queries,
    is_not_found,
    load_chunk_map,
    load_dotenv_value,
    compact_text,
)
from scripts.evaluate_planner import PhiPlannerRunner, extract_json, format_prompt, score_plan

from backend.settings import BackendSettings


logger = logging.getLogger(__name__)


class ChatRuntime:
    def __init__(self, settings: BackendSettings) -> None:
        self.settings = settings
        self.model_config = settings.model_config
        self.agent_config = settings.agent_config
        self.retrieval_config = settings.retrieval_config
        self._documents = self._load_documents()
        self._bm25 = BM25Index(self._documents) if self._documents else None
        self._chunk_map = self._load_chunk_map()
        self._executor_runner: QwenExecutorRunner | None = None
        self._planner_runner: PhiPlannerRunner | None = None
        self._fallback_runner: GeminiFallbackRunner | None = None
        self._executor_error: str | None = None
        self._planner_error: str | None = None
        self._fallback_error: str | None = None

    def _load_documents(self) -> list[Document]:
        if self.settings.chunks_file.exists():
            try:
                return IndexManager.load_chunks(self.settings.chunks_file)
            except Exception as exc:
                logger.warning("Failed to load chunks file %s: %s", self.settings.chunks_file, exc)

        if self.settings.processed_dir.exists():
            try:
                processor = DocumentProcessor(
                    DocumentProcessorConfig(
                        chunk_size=int(self.retrieval_config.get("chunking", {}).get("chunk_size", 384)),
                        chunk_overlap_ratio=float(self.retrieval_config.get("chunking", {}).get("chunk_overlap_ratio", 0.2)),
                    )
                )
                return processor.process(self.settings.processed_dir)
            except Exception as exc:
                logger.warning("Failed to build fallback chunks from %s: %s", self.settings.processed_dir, exc)

        return []

    def _load_chunk_map(self) -> dict[str, list[Document]]:
        if self.settings.chunks_file.exists():
            try:
                return load_chunk_map(self.settings.chunks_file)
            except Exception as exc:
                logger.warning("Failed to read chunk map from %s: %s", self.settings.chunks_file, exc)
        grouped: dict[str, list[Document]] = {}
        for document in self._documents:
            source = self._source_name(document)
            if not source:
                continue
            grouped.setdefault(source, []).append(document)
        return grouped

    @cached_property
    def source_names(self) -> list[str]:
        return sorted(self._chunk_map.keys())

    @cached_property
    def corpus_preview_text(self) -> str:
        if not self._documents:
            return ""
        return "\n\n".join(doc.text for doc in self._documents[:25])

    def health_payload(self) -> dict[str, Any]:
        index_summary = self._read_json(self.settings.index_summary_path)
        if not index_summary:
            index_summary = {
                "retrieval_mode": self.retrieval_mode,
                "chunk_count": len(self._documents),
                "source_count": len(self.source_names),
            }
        return {
            "status": "ok",
            "retrieval_mode": self.retrieval_mode,
            "document_count": len(self._documents),
            "source_count": len(self.source_names),
            "index_summary": index_summary,
        }

    def config_payload(self) -> dict[str, Any]:
        fallback_enabled = bool(self.settings.enable_gemini_fallback and self.settings.fallback_api_key)
        return {
            "planner": self._friendly_name(self.settings.planner_model_name, default="Phi"),
            "executor_primary": self._friendly_name(self.settings.executor_model_name, default="Qwen"),
            "fallback_enabled": fallback_enabled,
            "executor_fallback": self._friendly_name(self.settings.fallback_model_name, default="Gemini"),
            "retrieval_mode": self.retrieval_mode,
        }

    @property
    def retrieval_mode(self) -> str:
        retrieval_cfg = self.retrieval_config.get("retrieval", {})
        if isinstance(retrieval_cfg, dict):
            return str(retrieval_cfg.get("mode") or "hybrid")
        return "hybrid"

    def chat(self, message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        text = message.strip()
        if not text:
            raise ValueError("message is required")

        qa_type = self._infer_qa_type(text)
        ticker = self._extract_ticker(text)
        component_queries = build_component_queries(query=text, qa_type=qa_type, ticker=ticker)
        component_results = evaluate_component_queries(
            component_queries=component_queries,
            source_docs=self._documents,
            source_text=self.corpus_preview_text,
            top_k=self.settings.component_top_k,
            window_chars=self.settings.context_window_chars,
        )
        context = build_context_from_components(component_results, max_context_chars=self.settings.max_context_chars)
        selected_sources = self._selected_sources(component_results)
        planner_payload, planner_error = self._build_planner_payload(
            query=text,
            qa_type=qa_type,
            ticker=ticker,
            selected_sources=selected_sources,
        )
        row = {
            "source_file": ", ".join(selected_sources) if selected_sources else "KHÔNG CHỌN ĐƯỢC NGUỒN",
            "ticker": ticker,
            "qa_type": qa_type,
            "component_results": component_results,
            "context": context,
            "query": text,
        }
        answer, executor_used, executor_payload, routing_reason, fallback_used, executor_error = self._answer(row)
        if not answer:
            answer = self._heuristic_answer(text, component_results, context)
            executor_used = executor_used or "rules"
            routing_reason = routing_reason or "heuristic_fallback"

        support_rate, all_supported = component_support_metrics(component_results)
        retrieval_items = [item for item in component_results if str(item.get("type") or "") in {"retrieval", "retrieval_qa"}]
        component_count_ok = bool(retrieval_items) and all(bool(item.get("evidence_snippets")) for item in retrieval_items)

        retrieval_payload = {
            "component_results": component_results,
            "context_preview": compact_text(context, min(self.settings.max_context_chars, 1200)),
            "component_support_rate": support_rate,
            "component_count_ok": component_count_ok,
        }

        executor_details = {
            "qwen_answer": executor_payload.get("qwen_answer"),
            "gemini_answer": executor_payload.get("gemini_answer"),
        }
        if executor_error:
            executor_details["error"] = executor_error

        response = {
            "answer": answer,
            "executor_used": executor_used,
            "qa_type": qa_type,
            "ticker": ticker or None,
            "selected_sources": selected_sources,
            "routing_reason": routing_reason,
            "fallback_used": fallback_used,
            "planner": planner_payload,
            "retrieval": retrieval_payload,
            "executor": executor_details,
        }
        if planner_error:
            response["planner"]["error"] = planner_error
        if history:
            response["history_items"] = len(history)
        if qa_type == "multi_hop":
            response["retrieval"]["all_components_supported"] = all_supported
        return response

    def _build_planner_payload(
        self,
        *,
        query: str,
        qa_type: str,
        ticker: str,
        selected_sources: list[str],
    ) -> tuple[dict[str, Any], str | None]:
        if not self.settings.enable_local_planner:
            return self._heuristic_plan(query=query, qa_type=qa_type, ticker=ticker, selected_sources=selected_sources), None

        runner = self._planner_runner_cached()
        if runner is None:
            return self._heuristic_plan(query=query, qa_type=qa_type, ticker=ticker, selected_sources=selected_sources), self._planner_error

        available_sources = self.source_names or selected_sources or ["data/chunks/chunks.json"]
        item = {
            "query": query,
            "qa_type": qa_type,
            "ticker": ticker,
            "source_file": selected_sources[0] if selected_sources else available_sources[0],
            "available_sources": available_sources,
        }
        try:
            raw_plan = runner.generate(format_prompt(item))
            predicted_plan = extract_json(raw_plan)
            if not predicted_plan:
                return self._heuristic_plan(query=query, qa_type=qa_type, ticker=ticker, selected_sources=selected_sources), "planner returned invalid JSON"
            predicted_plan.setdefault("selected_sources", selected_sources or item["available_sources"][:1])
            metrics = score_plan(predicted_plan, item)
            return {
                "raw_prediction": raw_plan,
                "predicted_plan": predicted_plan,
                "metrics": metrics,
            }, None
        except Exception as exc:
            self._planner_error = str(exc)
            logger.warning("Planner load/generation failed: %s", exc)
            return self._heuristic_plan(query=query, qa_type=qa_type, ticker=ticker, selected_sources=selected_sources), str(exc)

    def _answer(
        self,
        row: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any], str, bool, str | None]:
        executor_payload: dict[str, Any] = {}
        executor_error: str | None = None
        if self.settings.enable_local_executor:
            runner = self._executor_runner_cached()
            if runner is not None:
                try:
                    qwen_answer = runner.answer(row)
                    executor_payload["qwen_answer"] = qwen_answer
                    if qwen_answer and not is_not_found(qwen_answer):
                        return qwen_answer, "qwen", executor_payload, "local_qwen", False, None
                except Exception as exc:
                    executor_error = str(exc)
                    self._executor_error = executor_error
                    logger.warning("Executor generation failed: %s", exc)

        if self.settings.enable_gemini_fallback and self.settings.fallback_api_key:
            fallback_runner = self._fallback_runner_cached()
            if fallback_runner is not None:
                try:
                    gemini_answer = fallback_runner.answer(row)
                    executor_payload["gemini_answer"] = gemini_answer
                    if gemini_answer:
                        return gemini_answer, "gemini", executor_payload, "gemini_fallback", True, executor_error
                except Exception as exc:
                    executor_error = str(exc)
                    self._fallback_error = executor_error
                    logger.warning("Gemini fallback failed: %s", exc)

        return "", "rules", executor_payload, "heuristic_fallback", False, executor_error

    def _executor_runner_cached(self) -> QwenExecutorRunner | None:
        if self._executor_runner is not None or self._executor_error is not None:
            return self._executor_runner
        if not self._looks_like_local_model(self.settings.executor_model_name):
            self._executor_error = f"executor model folder not found: {self.settings.executor_model_name}"
            return None
        try:
            args = SimpleNamespace(
                model_name_or_path=self.settings.executor_model_name,
                adapter_path=str(self.settings.project_root / "models" / "qwen_executor_lora"),
                local_files_only=self.settings.local_files_only,
                load_in_4bit=self.settings.load_in_4bit,
                device_map="auto",
                gpu_memory_limit=None,
                max_new_tokens=128,
            )
            self._executor_runner = QwenExecutorRunner(args)
        except Exception as exc:
            self._executor_error = str(exc)
            logger.warning("Failed to initialize executor runner: %s", exc)
        return self._executor_runner

    def _planner_runner_cached(self) -> PhiPlannerRunner | None:
        if self._planner_runner is not None or self._planner_error is not None:
            return self._planner_runner
        if not self._looks_like_local_model(self.settings.planner_model_name):
            self._planner_error = f"planner model folder not found: {self.settings.planner_model_name}"
            return None
        try:
            args = SimpleNamespace(
                model_name_or_path=self.settings.planner_model_name,
                adapter_path=str(self.settings.project_root / "models" / "phi_planner_lora"),
                local_files_only=self.settings.local_files_only,
                load_in_4bit=self.settings.load_in_4bit and self._maybe_has_cuda(),
                device_map="auto",
                gpu_memory_limit=None,
                max_new_tokens=256,
                prompt_max_length=1024,
            )
            self._planner_runner = PhiPlannerRunner(args)
        except Exception as exc:
            self._planner_error = str(exc)
            logger.warning("Failed to initialize planner runner: %s", exc)
        return self._planner_runner

    def _fallback_runner_cached(self) -> GeminiFallbackRunner | None:
        if self._fallback_runner is not None or self._fallback_error is not None:
            return self._fallback_runner
        try:
            args = SimpleNamespace(
                fallback_api_key=self.settings.fallback_api_key,
                fallback_model_name=self.settings.fallback_model_name,
                fallback_temperature=0.0,
                fallback_max_output_tokens=int(load_dotenv_value("GEMINI_FALLBACK_MAX_OUTPUT_TOKENS", "64") or "64"),
                fallback_thinking_budget=int(load_dotenv_value("GEMINI_FALLBACK_THINKING_BUDGET", "0") or "0"),
                fallback_context_chars=int(load_dotenv_value("GEMINI_FALLBACK_CONTEXT_CHARS", "900") or "900"),
                fallback_max_snippets=int(load_dotenv_value("GEMINI_FALLBACK_MAX_SNIPPETS", "2") or "2"),
                fallback_max_snippet_chars=int(load_dotenv_value("GEMINI_FALLBACK_MAX_SNIPPET_CHARS", "220") or "220"),
            )
            self._fallback_runner = GeminiFallbackRunner(args)
        except Exception as exc:
            self._fallback_error = str(exc)
            logger.warning("Failed to initialize Gemini fallback runner: %s", exc)
        return self._fallback_runner

    def _heuristic_plan(
        self,
        *,
        query: str,
        qa_type: str,
        ticker: str,
        selected_sources: list[str],
    ) -> dict[str, Any]:
        component_queries = build_component_queries(query=query, qa_type=qa_type, ticker=ticker)
        return {
            "raw_prediction": None,
            "predicted_plan": {
                "strategy": "sequential",
                "qa_type": qa_type,
                "ticker": ticker,
                "selected_sources": selected_sources or self.source_names[:1],
                "sub_queries": component_queries,
                "executor_instruction": "Dựa trên context đã truy xuất để trả lời ngắn gọn, đúng số liệu và đơn vị.",
            },
            "metrics": {
                "json_valid": False,
                "source_accuracy": None,
                "qa_type_accuracy": None,
                "ticker_accuracy": None,
                "subquery_present": float(bool(component_queries)),
                "planner_score": 0.0,
            },
        }

    def _heuristic_answer(self, query: str, component_results: list[dict[str, Any]], context: str) -> str:
        evidence_lines = []
        for item in component_results:
            for evidence in item.get("evidence_snippets") or []:
                snippet = str(evidence.get("snippet") or "").strip()
                if snippet:
                    evidence_lines.append(snippet)
        if evidence_lines:
            candidate = evidence_lines[0]
            if len(candidate) > 420:
                candidate = candidate[:420].rstrip() + "..."
            return candidate
        if context.strip():
            candidate = compact_text(context, 420)
            return candidate or "Mình chưa tìm thấy đủ bằng chứng trong tài liệu."
        return "Mình chưa tìm thấy đủ bằng chứng trong tài liệu."

    def _selected_sources(self, component_results: list[dict[str, Any]]) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()
        for item in component_results:
            for evidence in item.get("evidence_snippets") or []:
                source = self._source_name(str(evidence.get("source") or ""))
                if not source or source in seen:
                    continue
                seen.add(source)
                sources.append(source)
        return sources[:5]

    def _infer_qa_type(self, query: str) -> str:
        normalized = query.lower()
        multi_hop_markers = [
            "so sánh",
            "so sanh",
            "chênh lệch",
            "chenh lech",
            "tỷ lệ",
            "ty le",
            "chiếm",
            "tăng bao nhiêu",
            "giam bao nhieu",
            "giảm bao nhiêu",
            "bao nhiêu điểm",
            "bao nhieu diem",
            "giữa hai kỳ",
            "giua hai ky",
            "so với",
            "so voi",
        ]
        return "multi_hop" if any(marker in normalized for marker in multi_hop_markers) else "single_hop"

    def _extract_ticker(self, query: str) -> str:
        known_sources = {self._source_name(name).upper() for name in self.source_names}
        for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,4}\b", query):
            candidate = match.group(0).upper()
            if candidate in known_sources or len(candidate) <= 5:
                return candidate
        return ""

    def _source_name(self, value: str) -> str:
        return Path(value).name if value else ""

    def _friendly_name(self, value: str, *, default: str) -> str:
        if not value:
            return default
        name = Path(value).name.replace("_", " ").replace("-", " ").title()
        return name or default

    def _looks_like_local_model(self, model_name: str) -> bool:
        return Path(model_name).exists()

    def _maybe_has_cuda(self) -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _read_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
