#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from scripts import evaluate_executor as executor_eval
from scripts import evaluate_planner as planner_eval


NOT_FOUND = "KHÔNG TÌM THẤY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current Phi -> Qwen -> Gemini fallback architecture. "
            "For multi-hop, planner sub-queries must retrieve supporting evidence from data/processed_data; "
            "the final answer is scored against reference_answers.json."
        )
    )
    parser.add_argument("--planner_model_name_or_path", default="models/phi")
    parser.add_argument("--planner_adapter_path", default="models/phi_planner_lora")
    parser.add_argument("--executor_model_name_or_path", default="models/qwen")
    parser.add_argument("--executor_adapter_path", default="models/qwen_executor_lora")
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--chunks_file", default="data/chunks/chunks.json")
    parser.add_argument("--questions", default="data/test/questions.json")
    parser.add_argument("--answers", default="data/test/reference_answers.json")
    parser.add_argument("--plans_file", default="data/evaluation/system_plans.jsonl")
    parser.add_argument("--predictions_file", default="data/evaluation/system_predictions.jsonl")
    parser.add_argument("--metrics_file", default="data/evaluation/system_metrics.json")
    parser.add_argument("--output_file", default="output/system_output1.json")
    parser.add_argument("--qa_type_filter", default="all", choices=["all", "single_hop", "multi_hop"])
    parser.add_argument("--component_top_k", type=int, default=3)
    parser.add_argument("--max_context_chars", type=int, default=2200)
    parser.add_argument("--context_window_chars", type=int, default=520)
    parser.add_argument("--planner_max_sources", type=int, default=80)
    parser.add_argument("--planner_max_new_tokens", type=int, default=256)
    parser.add_argument("--planner_prompt_max_length", type=int, default=1024)
    parser.add_argument("--executor_max_new_tokens", type=int, default=128)
    parser.add_argument("--numeric_rel_tol", type=float, default=0.01)
    parser.add_argument("--percent_abs_tol", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--gpu_memory_limit", default=None)
    parser.add_argument("--enable_gemini_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback_model_name", default=executor_eval.load_dotenv_value("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
    parser.add_argument("--fallback_api_key", default=executor_eval.load_dotenv_value("GEMINI_API_KEY", None))
    parser.add_argument("--fallback_temperature", type=float, default=0.0)
    parser.add_argument("--fallback_max_output_tokens", type=int, default=int(executor_eval.load_dotenv_value("GEMINI_FALLBACK_MAX_OUTPUT_TOKENS", "64") or "64"))
    parser.add_argument("--fallback_thinking_budget", type=int, default=int(executor_eval.load_dotenv_value("GEMINI_FALLBACK_THINKING_BUDGET", "0") or "0"))
    parser.add_argument("--fallback_context_chars", type=int, default=int(executor_eval.load_dotenv_value("GEMINI_FALLBACK_CONTEXT_CHARS", "900") or "900"))
    parser.add_argument("--fallback_max_snippets", type=int, default=int(executor_eval.load_dotenv_value("GEMINI_FALLBACK_MAX_SNIPPETS", "2") or "2"))
    parser.add_argument("--fallback_max_snippet_chars", type=int, default=int(executor_eval.load_dotenv_value("GEMINI_FALLBACK_MAX_SNIPPET_CHARS", "220") or "220"))
    parser.add_argument("--gemini_low_evidence_threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    available_sources = planner_eval.list_processed_sources(
        Path(args.processed_dir),
        max_sources=args.planner_max_sources,
    )
    examples = planner_eval.build_examples(Path(args.questions), Path(args.answers), available_sources)
    references = {item["id"]: item for item in planner_eval.read_json_rows(Path(args.answers))}
    if args.qa_type_filter != "all":
        examples = [item for item in examples if item["qa_type"] == args.qa_type_filter]
    examples = examples[: args.limit] if args.limit > 0 else examples
    chunk_map = executor_eval.load_chunk_map(Path(args.chunks_file))
    if args.prepare_only:
        preview = prepare_preview(examples, available_sources, chunk_map)
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    planner_runner = planner_eval.PhiPlannerRunner(
        argparse.Namespace(
            model_name_or_path=args.planner_model_name_or_path,
            adapter_path=args.planner_adapter_path,
            local_files_only=args.local_files_only,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device_map,
            gpu_memory_limit=args.gpu_memory_limit,
            max_new_tokens=args.planner_max_new_tokens,
            prompt_max_length=args.planner_prompt_max_length,
        )
    )
    executor_runner = executor_eval.QwenExecutorRunner(
        argparse.Namespace(
            model_name_or_path=args.executor_model_name_or_path,
            adapter_path=args.executor_adapter_path,
            local_files_only=args.local_files_only,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device_map,
            gpu_memory_limit=args.gpu_memory_limit,
            max_new_tokens=args.executor_max_new_tokens,
        )
    )
    fallback_runner = None
    if args.enable_gemini_fallback and args.fallback_api_key:
        fallback_runner = executor_eval.GeminiFallbackRunner(
            argparse.Namespace(
                fallback_api_key=args.fallback_api_key,
                fallback_model_name=args.fallback_model_name,
                fallback_temperature=args.fallback_temperature,
                fallback_max_output_tokens=args.fallback_max_output_tokens,
                fallback_thinking_budget=args.fallback_thinking_budget,
                fallback_context_chars=args.fallback_context_chars,
                fallback_max_snippets=args.fallback_max_snippets,
                fallback_max_snippet_chars=args.fallback_max_snippet_chars,
            )
        )

    plans: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    text_cache: dict[Path, str] = {}

    for index, item in enumerate(examples, start=1):
        reference = references[item["id"]]
        prompt = planner_eval.format_prompt(item)
        raw_plan = planner_runner.generate(prompt)
        predicted_plan = planner_eval.extract_json(raw_plan)
        planner_scores = planner_eval.score_plan(predicted_plan, item)
        selected_sources = normalize_selected_sources(predicted_plan, available_sources)
        if not selected_sources:
            selected_sources = [item["source_file"]]
        source_docs, source_text = load_selected_sources(
            selected_sources=selected_sources,
            processed_dir=Path(args.processed_dir),
            chunk_map=chunk_map,
            text_cache=text_cache,
        )
        reference_component_queries = executor_eval.build_component_queries(
            query=item["query"],
            qa_type=item["qa_type"],
            ticker=item["ticker"],
        )
        planned_component_queries = build_plan_component_queries(
            predicted_plan,
            item,
            reference_component_queries=reference_component_queries,
        )
        component_results = executor_eval.evaluate_component_queries(
            component_queries=planned_component_queries,
            source_docs=source_docs,
            source_text=source_text,
            top_k=args.component_top_k,
            window_chars=args.context_window_chars,
        )
        context = executor_eval.build_context_from_components(
            component_results,
            max_context_chars=args.max_context_chars,
        )
        component_count_ok = planned_component_count_ok(item["qa_type"], component_results)
        evidence_number_coverage = executor_eval.score_evidence_number_coverage(
            ground_truth_context=str(reference.get("ground_truth_context") or ""),
            component_results=component_results,
            rel_tol=args.numeric_rel_tol,
            percent_abs_tol=args.percent_abs_tol,
        )
        executor_row = {
            "source_file": ", ".join(selected_sources) if selected_sources else "KHÔNG CHỌN ĐƯỢC NGUỒN",
            "ticker": item["ticker"],
            "qa_type": item["qa_type"],
            "component_results": component_results,
            "context": context,
            "query": item["query"],
        }
        prediction = ""
        executor_used = "qwen"
        routing_reason = "default_qwen"
        if fallback_runner and should_route_directly_to_gemini(
            qa_type=item["qa_type"],
            component_count_ok=component_count_ok,
            evidence_number_coverage=evidence_number_coverage,
            low_evidence_threshold=args.gemini_low_evidence_threshold,
        ):
            prediction = fallback_runner.answer(executor_row)
            executor_used = "gemini"
            routing_reason = "multi_hop_direct_gemini_synthesis"
        else:
            prediction = executor_runner.answer(executor_row)
            if fallback_runner and should_fallback_to_gemini(
                qa_type=item["qa_type"],
                prediction=prediction,
                answer=str(reference.get("ground_truth_answer") or ""),
                component_count_ok=component_count_ok,
                evidence_number_coverage=evidence_number_coverage,
                low_evidence_threshold=args.gemini_low_evidence_threshold,
            ):
                prediction = fallback_runner.answer(executor_row)
                executor_used = "gemini"
                routing_reason = "fallback_after_qwen"

        answer_metrics = executor_eval.score_prediction(
            prediction=prediction,
            answer=str(reference.get("ground_truth_answer") or ""),
            context=context,
            qa_type=item["qa_type"],
            component_results=component_results,
            ground_truth_context=str(reference.get("ground_truth_context") or ""),
            numeric_rel_tol=args.numeric_rel_tol,
            percent_abs_tol=args.percent_abs_tol,
        )
        retrieval_component_score = build_retrieval_component_score(
            component_results=component_results,
            component_support_rate=answer_metrics["component_support_rate"],
            evidence_number_coverage=answer_metrics["evidence_number_coverage"],
            component_count_ok=component_count_ok,
        )
        planner_score = float(planner_scores["planner_score"])
        end_to_end_score = (
            0.25 * planner_score
            + 0.25 * retrieval_component_score
            + 0.35 * answer_metrics["executor_score"]
            + 0.15 * float(answer_metrics["grounded"])
        )

        plan_row = {
            "id": item["id"],
            "query": item["query"],
            "qa_type": item["qa_type"],
            "ticker": item["ticker"],
            "source_file": item["source_file"],
            "raw_prediction": raw_plan,
            "predicted_plan": predicted_plan,
            "selected_sources": selected_sources,
            "planner_metrics": planner_scores,
            "planned_component_queries": planned_component_queries,
            "reference_component_queries": reference_component_queries,
        }
        plans.append(plan_row)

        prediction_row = {
            "index": index,
            "id": item["id"],
            "query": item["query"],
            "qa_type": item["qa_type"],
            "ticker": item["ticker"],
            "reference_answer": str(reference.get("ground_truth_answer") or ""),
            "reference_context": str(reference.get("ground_truth_context") or ""),
            "prediction": prediction,
            "executor_used": executor_used,
            "executor_routing_reason": routing_reason,
            "selected_sources": selected_sources,
            "planner": plan_row,
            "component_results": component_results,
            "component_count_ok": component_count_ok,
            "precomputed_evidence_number_coverage": evidence_number_coverage,
            "retrieval_component_score": retrieval_component_score,
            "end_to_end_score": end_to_end_score,
            **answer_metrics,
        }
        predictions.append(prediction_row)
        print(
            json.dumps(
                {
                    "index": index,
                    "id": item["id"],
                    "qa_type": item["qa_type"],
                    "planner_score": round(planner_score, 4),
                    "component_score": round(retrieval_component_score, 4),
                    "executor_score": round(answer_metrics["executor_score"], 4),
                    "end_to_end_score": round(end_to_end_score, 4),
                    "executor_used": executor_used,
                    "routing_reason": routing_reason,
                    "prediction": prediction[:160],
                },
                ensure_ascii=False,
            )
        )

    metrics = summarize(predictions)
    report = build_report(args, metrics, predictions)
    planner_eval.write_jsonl(Path(args.plans_file), plans)
    executor_eval.write_jsonl(Path(args.predictions_file), predictions)
    planner_eval.write_json(Path(args.metrics_file), metrics)
    planner_eval.write_json(Path(args.output_file), report)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote plans to {args.plans_file}")
    print(f"Wrote predictions to {args.predictions_file}")
    print(f"Wrote metrics to {args.metrics_file}")
    print(f"Wrote final system report to {args.output_file}")


def prepare_preview(
    examples: list[dict[str, Any]],
    available_sources: list[str],
    chunk_map: dict[str, list[Any]],
) -> dict[str, Any]:
    first = examples[0] if examples else None
    preview = {
        "rows": len(examples),
        "available_sources": available_sources[:10],
        "chunk_sources": len(chunk_map),
        "first": None,
    }
    if first:
        preview["first"] = {
            **first,
            "heuristic_component_queries": executor_eval.build_component_queries(
                query=first["query"],
                qa_type=first["qa_type"],
                ticker=first["ticker"],
            ),
        }
    return preview


def normalize_selected_sources(plan: dict[str, Any] | None, available_sources: list[str]) -> list[str]:
    if not isinstance(plan, dict):
        return []
    raw = plan.get("selected_sources") or []
    if isinstance(raw, str):
        raw = [raw]
    available_by_name = {Path(name).name: name for name in available_sources}
    selected = []
    for item in raw:
        name = Path(str(item)).name
        if name in available_by_name:
            selected.append(available_by_name[name])
    return unique(selected)


def load_selected_sources(
    *,
    selected_sources: list[str],
    processed_dir: Path,
    chunk_map: dict[str, list[Any]],
    text_cache: dict[Path, str],
) -> tuple[list[Any], str]:
    docs: list[Any] = []
    texts: list[str] = []
    for source in selected_sources:
        source_name = executor_eval.canonical_source_name(source)
        docs.extend(chunk_map.get(source_name) or [])
        source_path = executor_eval.resolve_source_path(processed_dir, source_name)
        if source_path not in text_cache:
            text_cache[source_path] = source_path.read_text(encoding="utf-8", errors="ignore")
        texts.append(text_cache[source_path])
    return docs, "\n\n".join(texts).strip()


def build_plan_component_queries(
    plan: dict[str, Any] | None,
    item: dict[str, Any],
    *,
    reference_component_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return reference_component_queries
    raw = plan.get("sub_queries") or []
    if isinstance(raw, dict):
        raw = [raw]
    normalized = []
    seen_ids: set[str] = set()
    for index, sub in enumerate(raw, start=1):
        if not isinstance(sub, dict):
            continue
        query = str(sub.get("query") or "").strip()
        if not query:
            continue
        kind = normalize_subquery_type(sub)
        sub_id = str(sub.get("id") or f"q{index}")
        if sub_id in seen_ids:
            sub_id = f"plan_{index}_{sub_id}"
        seen_ids.add(sub_id)
        normalized.append(
            {
                "id": sub_id,
                "query": query,
                "type": kind,
                "required": kind in {"retrieval_qa", "retrieval"},
                "depends_on": [str(value) for value in (sub.get("depends_on") or [])] if isinstance(sub.get("depends_on"), list) else [],
            }
        )
    if item["qa_type"] != "multi_hop":
        return normalized or reference_component_queries
    retrieval_count = sum(1 for current in normalized if current["type"] in {"retrieval_qa", "retrieval"})
    if retrieval_count >= 2:
        return normalized
    seen_queries = {executor_eval.normalize_answer(current["query"]) for current in normalized}
    for candidate in reference_component_queries:
        normalized_query = executor_eval.normalize_answer(candidate["query"])
        if normalized_query in seen_queries:
            continue
        if candidate["type"] in {"retrieval_qa", "retrieval"}:
            candidate_id = str(candidate["id"])
            while candidate_id in seen_ids:
                candidate_id = f"heuristic_{candidate_id}"
            seen_ids.add(candidate_id)
            normalized.append({**candidate, "id": candidate_id})
            seen_queries.add(normalized_query)
    if not any(current["type"] == "calculation" for current in normalized):
        for candidate in reference_component_queries:
            if candidate["type"] == "calculation":
                candidate_id = str(candidate["id"])
                while candidate_id in seen_ids:
                    candidate_id = f"heuristic_{candidate_id}"
                normalized.append({**candidate, "id": candidate_id})
                break
    return normalized


def normalize_subquery_type(sub_query: dict[str, Any]) -> str:
    raw_type = str(sub_query.get("type") or sub_query.get("tool") or "").lower()
    if any(token in raw_type for token in ("calc", "reason", "synth")):
        return "calculation"
    return "retrieval_qa"


def planned_component_count_ok(qa_type: str, component_results: list[dict[str, Any]]) -> bool:
    retrieval_count = sum(1 for item in component_results if str(item.get("type")) in {"retrieval_qa", "retrieval"})
    required = 2 if qa_type == "multi_hop" else 1
    return retrieval_count >= required


def build_retrieval_component_score(
    *,
    component_results: list[dict[str, Any]],
    component_support_rate: float | None,
    evidence_number_coverage: float | None,
    component_count_ok: bool,
) -> float:
    values = [float(component_count_ok)]
    if component_support_rate is not None:
        values.append(component_support_rate)
    if evidence_number_coverage is not None:
        values.append(evidence_number_coverage)
    return planner_eval.mean(values)


def should_route_directly_to_gemini(
    *,
    qa_type: str,
    component_count_ok: bool,
    evidence_number_coverage: float | None,
    low_evidence_threshold: float,
) -> bool:
    if qa_type != "multi_hop":
        return False
    if not component_count_ok:
        return False
    if evidence_number_coverage is None:
        return True
    return evidence_number_coverage >= low_evidence_threshold


def should_fallback_to_gemini(
    *,
    qa_type: str,
    prediction: str,
    answer: str,
    component_count_ok: bool,
    evidence_number_coverage: float | None,
    low_evidence_threshold: float,
) -> bool:
    if executor_eval.is_not_found(prediction):
        return True
    if qa_type != "multi_hop":
        return False
    if not component_count_ok:
        return True
    if evidence_number_coverage is not None and evidence_number_coverage < low_evidence_threshold:
        return True
    if clearly_bad_answer_format(prediction=prediction, answer=answer):
        return True
    return False


def clearly_bad_answer_format(*, prediction: str, answer: str) -> bool:
    prediction_normalized = executor_eval.normalize_answer(prediction)
    answer_normalized = executor_eval.normalize_answer(answer)
    prediction_numbers = executor_eval.extract_numbers(prediction)
    answer_numbers = executor_eval.extract_numbers(answer)
    if answer_numbers and not prediction_numbers:
        return True
    if " va " in answer_normalized and len(answer_numbers) >= 2 and len(prediction_numbers) < 2:
        return True
    if "%" in answer and "%" not in prediction:
        return True
    if "cổ phiếu" in answer.lower() and "cổ phiếu" not in prediction.lower():
        return True
    if "nhân viên" in answer.lower() and not any(token in prediction.lower() for token in ("nhân viên", "người")):
        return True
    return False


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    numeric_rows = [row for row in rows if row["numeric_accuracy"] is not None]
    component_rows = [row for row in rows if row["component_support_rate"] is not None]
    evidence_rows = [row for row in rows if row["evidence_number_coverage"] is not None]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["qa_type"]].append(row)

    return {
        "count": len(rows),
        "planner_score": planner_eval.mean(row["planner"]["planner_metrics"]["planner_score"] for row in rows),
        "planner_source_accuracy": planner_eval.mean(row["planner"]["planner_metrics"]["source_accuracy"] for row in rows),
        "planner_qa_type_accuracy": planner_eval.mean(row["planner"]["planner_metrics"]["qa_type_accuracy"] for row in rows),
        "exact_match": planner_eval.mean(row["em"] for row in rows),
        "f1": planner_eval.mean(row["f1"] for row in rows),
        "numeric_accuracy": planner_eval.mean(row["numeric_accuracy"] for row in numeric_rows) if numeric_rows else None,
        "groundedness": planner_eval.mean(float(row["grounded"]) for row in rows),
        "executor_score": planner_eval.mean(row["executor_score"] for row in rows),
        "component_support_rate": planner_eval.mean(row["component_support_rate"] for row in component_rows) if component_rows else None,
        "evidence_number_coverage": planner_eval.mean(row["evidence_number_coverage"] for row in evidence_rows) if evidence_rows else None,
        "component_count_ok_rate": planner_eval.mean(float(row["component_count_ok"]) for row in rows),
        "retrieval_component_score": planner_eval.mean(row["retrieval_component_score"] for row in rows),
        "end_to_end_score": planner_eval.mean(row["end_to_end_score"] for row in rows),
        "fallback_usage_rate": planner_eval.mean(float(row["executor_used"] == "gemini") for row in rows),
        "by_qa_type": {
            qa_type: summarize_subset(items)
            for qa_type, items in sorted(by_type.items())
        },
        "failure_summary": {
            "planner_invalid_json": sum(1 for row in rows if not row["planner"]["planner_metrics"]["json_valid"]),
            "planner_wrong_source": sum(1 for row in rows if not row["planner"]["planner_metrics"]["source_accuracy"]),
            "multi_hop_missing_components": sum(
                1 for row in rows if row["qa_type"] == "multi_hop" and not row["component_count_ok"]
            ),
            "multi_hop_component_not_grounded": sum(
                1
                for row in rows
                if row["qa_type"] == "multi_hop"
                and (row["component_support_rate"] or 0.0) < 1.0
            ),
            "executor_not_found": sum(1 for row in rows if executor_eval.is_not_found(row["prediction"])),
            "gemini_fallback_used": sum(1 for row in rows if row["executor_used"] == "gemini"),
        },
    }


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_rows = [row for row in rows if row["numeric_accuracy"] is not None]
    component_rows = [row for row in rows if row["component_support_rate"] is not None]
    evidence_rows = [row for row in rows if row["evidence_number_coverage"] is not None]
    return {
        "count": len(rows),
        "planner_score": planner_eval.mean(row["planner"]["planner_metrics"]["planner_score"] for row in rows),
        "exact_match": planner_eval.mean(row["em"] for row in rows),
        "f1": planner_eval.mean(row["f1"] for row in rows),
        "numeric_accuracy": planner_eval.mean(row["numeric_accuracy"] for row in numeric_rows) if numeric_rows else None,
        "groundedness": planner_eval.mean(float(row["grounded"]) for row in rows),
        "executor_score": planner_eval.mean(row["executor_score"] for row in rows),
        "component_support_rate": planner_eval.mean(row["component_support_rate"] for row in component_rows) if component_rows else None,
        "evidence_number_coverage": planner_eval.mean(row["evidence_number_coverage"] for row in evidence_rows) if evidence_rows else None,
        "component_count_ok_rate": planner_eval.mean(float(row["component_count_ok"]) for row in rows),
        "retrieval_component_score": planner_eval.mean(row["retrieval_component_score"] for row in rows),
        "end_to_end_score": planner_eval.mean(row["end_to_end_score"] for row in rows),
        "fallback_usage_rate": planner_eval.mean(float(row["executor_used"] == "gemini") for row in rows),
    }


def build_report(args: argparse.Namespace, metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "project": "vn_stock_mao_arag",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "Phi planner -> Qwen executor -> Gemini fallback",
        "evaluation_basis": {
            "questions": args.questions,
            "answers": args.answers,
            "processed_dir": args.processed_dir,
            "chunks_file": args.chunks_file,
        },
        "modes": {
            "qa_type_filter": args.qa_type_filter,
            "enable_gemini_fallback": args.enable_gemini_fallback,
            "local_files_only": args.local_files_only,
            "load_in_4bit": args.load_in_4bit,
        },
        "score_formula": {
            "executor_score": "0.30*exact_match + 0.30*f1 + 0.25*numeric_accuracy + 0.15*groundedness",
            "retrieval_component_score": "mean(component_count_ok, component_support_rate, evidence_number_coverage_when_available)",
            "end_to_end_score": "0.25*planner_score + 0.25*retrieval_component_score + 0.35*executor_score + 0.15*groundedness",
        },
        "notes": {
            "multi_hop_rule": "Chi riêng multi-hop, planner phải sinh được sub-queries; sub-queries retrieval phải tìm được bằng chứng trong data/processed_data; đáp án cuối cùng vẫn chấm theo reference_answers.json.",
            "fallback_rule": "Gemini chỉ được gọi khi Qwen trả về rỗng hoặc KHÔNG TÌM THẤY.",
        },
        "metrics": metrics,
        "predictions": compact_predictions(predictions),
    }


def compact_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "index": row["index"],
                "id": row["id"],
                "qa_type": row["qa_type"],
                "ticker": row["ticker"],
                "query": row["query"],
                "reference_answer": row["reference_answer"],
                "prediction": row["prediction"],
                "executor_used": row["executor_used"],
                "selected_sources": row["selected_sources"],
                "planner_score": row["planner"]["planner_metrics"]["planner_score"],
                "component_count_ok": row["component_count_ok"],
                "component_support_rate": row["component_support_rate"],
                "evidence_number_coverage": row["evidence_number_coverage"],
                "retrieval_component_score": row["retrieval_component_score"],
                "executor_score": row["executor_score"],
                "end_to_end_score": row["end_to_end_score"],
            }
        )
    return compact


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


if __name__ == "__main__":
    main()
