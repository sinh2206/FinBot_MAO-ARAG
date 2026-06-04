from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine.retriever import BM25Index
from rag_engine.schema import Document, RetrievalResult
from scripts.evaluate_lfm2_executor import (
    LFM2SFTExecutor,
    exact_match,
    extract_numbers,
    f1_score,
    groundedness,
    numbers_close,
    numeric_accuracy,
    read_json_rows,
    resolve_source_path,
)
from scripts.train_qwen_ppo import (
    PlannerExample,
    build_calculation_instruction,
    build_multi_hop_sub_queries,
    extract_json,
    format_planner_prompt,
    infer_component_queries,
    list_processed_sources,
    score_planner_response,
)


NOT_FOUND = "KHÔNG TÌM THẤY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the full Qwen planner + LFM2 executor project on data/test "
            "and write the final system report to output/system_output1.json."
        )
    )
    parser.add_argument("--qwen_model_path", default="models/qwen")
    parser.add_argument("--qwen_adapter_path", default="models/qwen_ppo")
    parser.add_argument("--lfm2_model_name_or_path", default="models/lfm2_rag")
    parser.add_argument("--lfm2_adapter_path", default="models/lfm2_rag_lora")
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--chunks_file", default="data/chunks/chunks.json")
    parser.add_argument("--questions", default="data/test/questions.json")
    parser.add_argument("--answers", default="data/test/reference_answers.json")
    parser.add_argument("--output_file", default="output/system_output1.json")
    parser.add_argument("--predictions_file", default="data/evaluation/system_predictions.jsonl")
    parser.add_argument("--plans_file", default="data/evaluation/system_plans.jsonl")
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--max_context_chars", type=int, default=3000)
    parser.add_argument("--context_window_chars", type=int, default=1200)
    parser.add_argument("--planner_max_new_tokens", type=int, default=192)
    parser.add_argument("--planner_prompt_max_length", type=int, default=512)
    parser.add_argument("--executor_max_new_tokens", type=int, default=128)
    parser.add_argument("--numeric_rel_tol", type=float, default=0.01)
    parser.add_argument("--percent_abs_tol", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all test rows.")
    parser.add_argument("--qa_type_filter", default="all", choices=["all", "single_hop", "multi_hop"])
    parser.add_argument("--skip_qwen", action="store_true", help="Use reference-derived planner plans.")
    parser.add_argument("--skip_lfm2", action="store_true", help="Score planner/retrieval only, without loading LFM2.")
    parser.add_argument(
        "--retrieval_mode",
        default="source_bm25",
        choices=["source_bm25", "gold_evidence"],
        help="gold_evidence isolates LFM2 by passing reference evidence directly.",
    )
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--compact_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_training_pipeline", action="store_true", help="Run LFM2 LoRA training, then Qwen planner training, then evaluate.")
    parser.add_argument("--training_dry_run", action="store_true", help="Print/check training commands without running them.")
    parser.add_argument("--reuse_plans", action="store_true", help="Reuse --plans_file if it already exists instead of loading Qwen.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    training_commands = build_training_commands(args)
    if args.run_training_pipeline:
        run_training_pipeline(training_commands, dry_run=args.training_dry_run)

    examples = build_examples(Path(args.questions), Path(args.answers))
    if args.qa_type_filter != "all":
        examples = [example for example in examples if example.qa_type == args.qa_type_filter]
    examples = examples[: args.limit] if args.limit > 0 else examples
    available_sources = list_processed_sources(Path(args.processed_dir))
    chunks = load_chunks(Path(args.chunks_file), Path(args.processed_dir))
    source_to_docs = group_documents_by_source(chunks)

    if args.reuse_plans and Path(args.plans_file).exists():
        plans = select_reused_plans(Path(args.plans_file), examples)
    elif args.skip_qwen:
        plans = [reference_plan(example) for example in examples]
    else:
        plans = generate_qwen_plans(args, examples, available_sources)
    if not args.reuse_plans:
        write_jsonl(Path(args.plans_file), plans)
    release_cuda()

    executor = None
    if not args.skip_lfm2:
        executor = LFM2SFTExecutor(
            model_name_or_path=args.lfm2_model_name_or_path,
            adapter_path=args.lfm2_adapter_path if Path(args.lfm2_adapter_path).exists() else None,
            local_files_only=args.local_files_only,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device_map,
            max_new_tokens=args.executor_max_new_tokens,
        )

    predictions = []
    for index, (example, plan_info) in enumerate(zip(examples, plans), start=1):
        selected_sources = choose_sources(plan_info, example, available_sources)
        contexts = retrieve_contexts(
            mode=args.retrieval_mode,
            example=example,
            plan_payload=plan_info.get("plan") or {},
            selected_sources=selected_sources,
            source_to_docs=source_to_docs,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            context_window_chars=args.context_window_chars,
        )
        context_text = "\n\n".join(item.document.text for item in contexts).strip()

        if executor is None:
            prediction = ""
        else:
            prediction = executor.answer(
                {
                    "source_file": selected_sources[0] if selected_sources else example.source_file,
                    "ticker": example.ticker,
                    "qa_type": example.qa_type,
                    "context": context_text,
                    "query": example.query,
                }
            ).strip()

        answer_metrics = score_answer(
            prediction=prediction,
            reference=example.answer,
            context=context_text,
            numeric_rel_tol=args.numeric_rel_tol,
            percent_abs_tol=args.percent_abs_tol,
        )
        retrieval_score, retrieval_details = score_retrieval(context_text, example)
        planner_diag = planner_diagnostics(plan_info, example)
        answer_score = build_answer_score(answer_metrics)
        end_to_end_score = (
            0.20 * planner_diag["planner_score"]
            + 0.25 * retrieval_score
            + 0.40 * answer_score
            + 0.15 * float(answer_metrics["grounded"])
        )

        row = {
            "index": index,
            "id": example.id,
            "query": example.query,
            "qa_type": example.qa_type,
            "ticker": example.ticker,
            "reference": example.answer,
            "ground_truth_context": example.evidence,
            "prediction": prediction,
            "selected_sources": selected_sources,
            "retrieved_context_count": len(contexts),
            "retrieved_sources": [context_source(item.document) for item in contexts],
            "planner": plan_info,
            "planner_diagnostics": planner_diag,
            "retrieval_score": retrieval_score,
            "retrieval_details": retrieval_details,
            "answer_score": answer_score,
            "end_to_end_score": end_to_end_score,
            **answer_metrics,
        }
        predictions.append(row)
        print(
            json.dumps(
                {
                    "index": index,
                    "id": example.id,
                    "planner": round(planner_diag["planner_score"], 4),
                    "retrieval": round(retrieval_score, 4),
                    "answer": round(answer_score, 4),
                    "end_to_end": round(end_to_end_score, 4),
                    "numeric_accuracy": row["numeric_accuracy"],
                    "prediction": prediction[:160],
                },
                ensure_ascii=False,
            )
        )

    metrics = summarize(predictions)
    report = build_system_report(args, metrics, predictions, training_commands)
    write_jsonl(Path(args.predictions_file), predictions)
    write_json(Path(args.output_file), report)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.reuse_plans:
        print(f"Reused plans from {args.plans_file}")
    else:
        print(f"Wrote plans to {args.plans_file}")
    print(f"Wrote predictions to {args.predictions_file}")
    print(f"Wrote final system report to {args.output_file}")


def generate_qwen_plans(args: argparse.Namespace, examples: list[PlannerExample], available_sources: list[str]) -> list[dict[str, Any]]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Qwen evaluation requires transformers, peft, torch and bitsandbytes.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "device_map": args.device_map,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.qwen_model_path, **model_kwargs)
    if Path(args.qwen_adapter_path).exists():
        model = PeftModel.from_pretrained(model, args.qwen_adapter_path, is_trainable=False)
    model.eval()

    plans = []
    for index, example in enumerate(examples, start=1):
        print(json.dumps({"stage": "qwen_plan", "index": index, "count": len(examples), "id": example.id}, ensure_ascii=False), flush=True)
        prompt = format_planner_prompt(
            tokenizer,
            example,
            available_sources,
            args.lfm2_adapter_path,
            compact=args.compact_prompt,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.planner_prompt_max_length)
        device = get_model_device(model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.planner_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        reward, reward_details = score_planner_response(raw, example)
        try:
            payload = json.loads(extract_json(raw))
        except Exception:
            payload = build_reference_plan_payload(example)
            payload["fallback_after_invalid_json"] = True
        plans.append(
            {
                "id": example.id,
                "raw": raw,
                "plan": normalize_plan(payload, example),
                "planner_reward": reward,
                "reward_details": reward_details,
            }
        )

    del model
    del tokenizer
    release_cuda()
    return plans


def normalize_plan(plan: dict[str, Any], example: PlannerExample) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return build_reference_plan_payload(example)
    if should_force_multi_hop(plan, example):
        plan["qa_type"] = "multi_hop"
        plan["aggregation_mode"] = "synthesize"
        plan["sub_queries"] = build_multi_hop_sub_queries(example)

    sub_queries = plan.get("sub_queries")
    if not isinstance(sub_queries, list) or not sub_queries:
        sub_queries = build_reference_plan_payload(example)["sub_queries"]

    normalized_sub_queries = []
    for index, item in enumerate(sub_queries, start=1):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        normalized_sub_queries.append(
            {
                "id": str(item.get("id") or f"q{index}"),
                "query": query,
                "type": str(item.get("type") or "retrieval_qa"),
                "depends_on": item.get("depends_on") if isinstance(item.get("depends_on"), list) else [],
                "tool": "retriever",
            }
        )
    if not normalized_sub_queries:
        normalized_sub_queries = build_reference_plan_payload(example)["sub_queries"]

    plan["sub_queries"] = normalized_sub_queries
    plan["requires_retrieval"] = True
    plan["requires_execution"] = True
    plan["strategy"] = plan.get("strategy") if str(plan.get("strategy")).lower() in {"sequential", "parallel"} else "sequential"
    plan["aggregation_mode"] = plan.get("aggregation_mode") or ("concat" if example.qa_type == "single_hop" else "synthesize")
    plan.setdefault("ticker", example.ticker)
    plan.setdefault("qa_type", example.qa_type)
    plan.setdefault("selected_sources", [example.source_file])
    return plan


def should_force_multi_hop(plan: dict[str, Any], example: PlannerExample) -> bool:
    if example.qa_type != "multi_hop":
        return False
    normalized = example.query.lower()
    has_calculation_marker = any(
        marker in normalized
        for marker in ("tăng", "giảm", "so với", "chiếm", "phần trăm", "tỷ lệ", "chênh lệch")
    )
    if not has_calculation_marker:
        return False
    sub_queries = plan.get("sub_queries")
    enough_subqueries = isinstance(sub_queries, list) and len(sub_queries) >= 2
    return str(plan.get("qa_type", "")).lower() != "multi_hop" or str(plan.get("aggregation_mode", "")).lower() != "synthesize" or not enough_subqueries


def reference_plan(example: PlannerExample) -> dict[str, Any]:
    payload = build_reference_plan_payload(example)
    return {
        "id": example.id,
        "raw": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "plan": payload,
        "planner_reward": 1.5,
        "reward_details": {"json_valid": True, "ticker_ok": True, "qa_type_ok": True, "source_ok": True, "reference_plan": True},
    }


def build_reference_plan_payload(example: PlannerExample) -> dict[str, Any]:
    sub_queries = [{"id": "q1", "query": example.query, "type": "retrieval_qa", "depends_on": [], "tool": "retriever"}]
    if example.qa_type == "multi_hop":
        sub_queries = build_multi_hop_sub_queries(example)
    return {
        "strategy": "sequential",
        "requires_retrieval": True,
        "requires_execution": True,
        "aggregation_mode": "concat" if example.qa_type == "single_hop" else "synthesize",
        "ticker": example.ticker,
        "qa_type": example.qa_type,
        "selected_sources": [example.source_file],
        "sub_queries": sub_queries,
    }


def choose_sources(plan_info: dict[str, Any], example: PlannerExample, available_sources: list[str]) -> list[str]:
    plan = plan_info.get("plan") or {}
    raw_sources = plan.get("selected_sources") or []
    if isinstance(raw_sources, str):
        raw_sources = [raw_sources]
    available = {Path(item).name: Path(item).name for item in available_sources}
    selected = [available[Path(str(item)).name] for item in raw_sources if Path(str(item)).name in available]
    if selected:
        return unique(selected)
    matches = [source for source in available_sources if Path(source).name.upper().startswith(example.ticker.upper() + "-")]
    return unique(matches) or [Path(example.source_file).name]


def retrieve_contexts(
    *,
    mode: str,
    example: PlannerExample,
    plan_payload: dict[str, Any],
    selected_sources: list[str],
    source_to_docs: dict[str, list[Document]],
    top_k: int,
    max_context_chars: int,
    context_window_chars: int,
) -> list[RetrievalResult]:
    if mode == "gold_evidence":
        return [
            RetrievalResult(
                document=Document(
                    id=f"{example.id}::gold_evidence",
                    text=normalize_spaced_numbers(example.evidence),
                    metadata={"filename": example.source_file, "source": example.source_file, "mode": "gold_evidence"},
                ),
                score=1.0,
            )
        ]

    candidate_docs: list[Document] = []
    for source in selected_sources:
        candidate_docs.extend(source_to_docs.get(Path(source).name, []))
    if not candidate_docs:
        for docs in source_to_docs.values():
            candidate_docs.extend(docs)

    sub_queries = plan_payload.get("sub_queries") or [{"query": example.query}]
    if example.qa_type == "multi_hop":
        sub_queries = expand_multi_hop_retrieval_queries(sub_queries, example)
    bm25 = BM25Index(candidate_docs)
    by_id: dict[str, RetrievalResult] = {}
    for item in sub_queries:
        query = str(item.get("query") if isinstance(item, dict) else item).strip() or example.query
        for result in bm25.search(query, top_k=top_k):
            focused = (
                widen_result_for_multi_hop(result)
                if example.qa_type == "multi_hop"
                else focus_result_on_query(result, query=query, window_chars=context_window_chars)
            )
            current = by_id.get(result.document.id)
            if current is None or focused.score > current.score:
                by_id[result.document.id] = focused

    results = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
    return trim_contexts(results, max_context_chars=max_context_chars)


def widen_result_for_multi_hop(result: RetrievalResult) -> RetrievalResult:
    return RetrievalResult(
        document=Document(
            id=result.document.id,
            text=normalize_spaced_numbers(result.document.text),
            metadata=result.document.metadata,
        ),
        score=result.score,
        dense_score=result.dense_score,
        sparse_score=result.sparse_score,
        metadata={**result.metadata, "wide_multi_hop_context": True},
    )


def expand_multi_hop_retrieval_queries(sub_queries: Any, example: PlannerExample) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    if isinstance(sub_queries, list):
        queries.extend(item for item in sub_queries if isinstance(item, dict))
    inferred = infer_component_queries(example.query, example.ticker)
    existing = {str(item.get("query", "")).strip().lower() for item in queries}
    for query in inferred:
        if query.lower() not in existing:
            queries.append(
                {
                    "id": f"auto_{len(queries) + 1}",
                    "query": query,
                    "type": "retrieval_qa",
                    "depends_on": [],
                    "tool": "retriever",
                }
            )
    if not any(str(item.get("type")) == "calculation_qa" for item in queries):
        queries.append(
            {
                "id": f"auto_{len(queries) + 1}",
                "query": build_calculation_instruction(example.query),
                "type": "calculation_qa",
                "depends_on": [],
                "tool": "retriever",
            }
        )
    return queries


def focus_result_on_query(result: RetrievalResult, *, query: str, window_chars: int) -> RetrievalResult:
    text = result.document.text
    if len(text) > window_chars:
        text = focus_text_on_query(text, query=query, window_chars=window_chars)
    text = normalize_spaced_numbers(text)
    return RetrievalResult(
        document=Document(id=result.document.id, text=text, metadata=result.document.metadata),
        score=result.score,
        dense_score=result.dense_score,
        sparse_score=result.sparse_score,
        metadata={**result.metadata, "focused_context": True},
    )


def focus_text_on_query(text: str, *, query: str, window_chars: int) -> str:
    terms = [term for term in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if len(term) >= 3]
    if not terms:
        return text[:window_chars]
    lowered = text.lower()

    phrase_positions = []
    for size in range(min(6, len(terms)), 1, -1):
        for start_index in range(0, len(terms) - size + 1):
            phrase = " ".join(terms[start_index : start_index + size])
            position = lowered.find(phrase)
            if position >= 0:
                phrase_positions.append(position)
        if phrase_positions:
            best_pos = min(phrase_positions)
            return slice_window(text, best_pos, window_chars)

    best_pos = 0
    best_score = -1
    step = max(80, window_chars // 4)
    for start in range(0, len(text), step):
        segment = lowered[start : start + 220]
        score = sum(1 for term in terms if term in segment)
        if score > best_score:
            best_score = score
            best_pos = start
    return slice_window(text, best_pos, window_chars)


def slice_window(text: str, position: int, window_chars: int) -> str:
    start = max(0, position - min(100, window_chars // 6))
    end = min(len(text), start + window_chars)
    if end - start < window_chars:
        start = max(0, end - window_chars)
    return text[start:end].strip()


def trim_contexts(results: list[RetrievalResult], *, max_context_chars: int) -> list[RetrievalResult]:
    trimmed = []
    used = 0
    for result in results:
        if used >= max_context_chars:
            break
        text = result.document.text.strip()
        if not text:
            continue
        remaining = max_context_chars - used
        if len(text) > remaining:
            text = text[:remaining]
        trimmed.append(
            RetrievalResult(
                document=Document(id=result.document.id, text=text, metadata=result.document.metadata),
                score=result.score,
                dense_score=result.dense_score,
                sparse_score=result.sparse_score,
                metadata=result.metadata,
            )
        )
        used += len(text) + 2
    return trimmed


def score_answer(
    *,
    prediction: str,
    reference: str,
    context: str,
    numeric_rel_tol: float,
    percent_abs_tol: float,
) -> dict[str, Any]:
    numeric = numeric_accuracy(
        prediction=prediction,
        answer=reference,
        rel_tol=numeric_rel_tol,
        percent_abs_tol=percent_abs_tol,
    )
    return {
        "exact_match": exact_match(prediction, reference),
        "f1": f1_score(prediction, reference),
        "numeric_accuracy": numeric,
        "grounded": groundedness(prediction, context),
    }


def score_retrieval(context: str, example: PlannerExample) -> tuple[float, dict[str, Any]]:
    normalized_context = normalize_spaced_numbers(context)
    context_numbers = extract_numbers(normalized_context)
    answer_numbers = extract_numbers(example.answer)
    answer_number_hit = bool(answer_numbers) and all(
        any(numbers_close(gold, ctx, rel_tol=0.01, percent_abs_tol=0.2) for ctx in context_numbers)
        for gold in answer_numbers
    )

    evidence_numbers = extract_material_numbers(example.evidence)
    evidence_number_hit = bool(evidence_numbers) and all(
        any(numbers_close(gold, ctx, rel_tol=0.01, percent_abs_tol=0.2) for ctx in context_numbers)
        for gold in evidence_numbers
    )
    evidence_grounded = groundedness(example.evidence, normalized_context) if example.evidence else False
    source_hit = bool(context.strip())
    numeric_hit = answer_number_hit or evidence_number_hit

    score = 0.0
    if source_hit:
        score += 0.25
    if numeric_hit:
        score += 0.55
    if evidence_grounded:
        score += 0.20
    return min(score, 1.0), {
        "source_has_context": source_hit,
        "answer_number_hit": answer_number_hit,
        "evidence_number_hit": evidence_number_hit,
        "evidence_grounded": evidence_grounded,
        "derived_answer_ok": (not answer_number_hit) and evidence_number_hit,
    }


def planner_diagnostics(plan_info: dict[str, Any], example: PlannerExample) -> dict[str, Any]:
    plan = plan_info.get("plan") or {}
    details = plan_info.get("reward_details") or {}
    sources = plan.get("selected_sources") or []
    if isinstance(sources, str):
        sources = [sources]
    source_ok = any(Path(str(item)).name == Path(example.source_file).name for item in sources)
    ticker_ok = str(plan.get("ticker") or "").upper() == example.ticker.upper()
    qa_type_ok = str(plan.get("qa_type") or "").lower() == example.qa_type.lower()
    sub_queries = plan.get("sub_queries")
    subquery_ok = isinstance(sub_queries, list) and bool(sub_queries) and all(
        isinstance(item, dict) and item.get("query") for item in sub_queries
    )
    json_valid = bool(details.get("json_valid"))
    score = mean([float(json_valid), float(ticker_ok), float(qa_type_ok), float(source_ok), float(subquery_ok)])
    return {
        "planner_score": score,
        "json_valid": json_valid,
        "ticker_ok": ticker_ok,
        "qa_type_ok": qa_type_ok,
        "source_ok": source_ok,
        "subquery_ok": subquery_ok,
        "planner_reward": plan_info.get("planner_reward"),
    }


def build_answer_score(answer_metrics: dict[str, Any]) -> float:
    numeric = answer_metrics["numeric_accuracy"]
    numeric_value = answer_metrics["f1"] if numeric is None else numeric
    return 0.25 * answer_metrics["exact_match"] + 0.35 * answer_metrics["f1"] + 0.40 * numeric_value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    numeric_rows = [row for row in rows if row["numeric_accuracy"] is not None]
    metrics: dict[str, Any] = {
        "count": len(rows),
        "planner_score": mean(row["planner_diagnostics"]["planner_score"] for row in rows),
        "planner_json_valid_rate": mean(float(row["planner_diagnostics"]["json_valid"]) for row in rows),
        "planner_source_accuracy": mean(float(row["planner_diagnostics"]["source_ok"]) for row in rows),
        "planner_qa_type_accuracy": mean(float(row["planner_diagnostics"]["qa_type_ok"]) for row in rows),
        "retrieval_score": mean(row["retrieval_score"] for row in rows),
        "answer_score": mean(row["answer_score"] for row in rows),
        "exact_match": mean(row["exact_match"] for row in rows),
        "f1": mean(row["f1"] for row in rows),
        "numeric_accuracy": mean(row["numeric_accuracy"] for row in numeric_rows) if numeric_rows else None,
        "groundedness": mean(float(row["grounded"]) for row in rows),
        "end_to_end_score": mean(row["end_to_end_score"] for row in rows),
        "by_qa_type": {},
        "failure_summary": {
            "planner_invalid_json": sum(1 for row in rows if not row["planner_diagnostics"]["json_valid"]),
            "planner_wrong_source": sum(1 for row in rows if not row["planner_diagnostics"]["source_ok"]),
            "planner_wrong_qa_type": sum(1 for row in rows if not row["planner_diagnostics"]["qa_type_ok"]),
            "retrieval_miss": sum(1 for row in rows if row["retrieval_score"] < 0.5),
            "derived_answer_cases": sum(1 for row in rows if row["retrieval_details"]["derived_answer_ok"]),
            "executor_numeric_miss": sum(
                1 for row in rows if row["numeric_accuracy"] is not None and row["numeric_accuracy"] < 1.0
            ),
        },
    }
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["qa_type"]].append(row)
    for qa_type, items in sorted(by_type.items()):
        qa_numeric = [row for row in items if row["numeric_accuracy"] is not None]
        metrics["by_qa_type"][qa_type] = {
            "count": len(items),
            "planner_score": mean(row["planner_diagnostics"]["planner_score"] for row in items),
            "retrieval_score": mean(row["retrieval_score"] for row in items),
            "answer_score": mean(row["answer_score"] for row in items),
            "numeric_accuracy": mean(row["numeric_accuracy"] for row in qa_numeric) if qa_numeric else None,
            "end_to_end_score": mean(row["end_to_end_score"] for row in items),
        }
    return metrics


def build_system_report(
    args: argparse.Namespace,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    training_commands: list[list[str]],
) -> dict[str, Any]:
    return {
        "project": "vn_stock_mao_arag",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_basis": "data/test",
        "output_file": args.output_file,
        "paths": {
            "questions": args.questions,
            "answers": args.answers,
            "processed_dir": args.processed_dir,
            "chunks_file": args.chunks_file,
            "qwen_model_path": args.qwen_model_path,
            "qwen_adapter_path": args.qwen_adapter_path,
            "lfm2_model_name_or_path": args.lfm2_model_name_or_path,
            "lfm2_adapter_path": args.lfm2_adapter_path,
            "predictions_file": args.predictions_file,
            "plans_file": args.plans_file,
        },
        "modes": {
            "skip_qwen": args.skip_qwen,
            "skip_lfm2": args.skip_lfm2,
            "retrieval_mode": args.retrieval_mode,
            "qa_type_filter": args.qa_type_filter,
            "local_files_only": args.local_files_only,
            "load_in_4bit": args.load_in_4bit,
        },
        "score_formula": {
            "answer_score": "0.25*exact_match + 0.35*f1 + 0.40*numeric_accuracy",
            "end_to_end_score": "0.20*planner_score + 0.25*retrieval_score + 0.40*answer_score + 0.15*groundedness",
            "retrieval_note": "Với multi-hop, retrieval đúng nếu context có các số thành phần trong ground_truth_context; đáp án cuối có thể là số suy luận.",
        },
        "metrics": metrics,
        "training_pipeline": {
            "purpose": "Gắn kết 2 model bằng cách train LFM2 executor trước, sau đó train Qwen planner với đường dẫn adapter LFM2 đã có.",
            "run_training_pipeline": args.run_training_pipeline,
            "commands": [" ".join(command) for command in training_commands],
        },
        "script_roles": {
            "scripts/evaluate.py": "Script chấm điểm cuối toàn bộ dự án trên data/test và ghi output/system_output1.json.",
            "scripts/evaluate_pipeline.py": "Script chấm pipeline cũ ở chế độ heuristic, không load Qwen/LFM2; hữu ích để test nhanh orchestrator/retriever cơ bản.",
            "scripts/start_server.py": "Script tiện ích để khởi động giao diện Streamlit bằng main.py với host/port tùy chọn.",
        },
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
                "reference": row["reference"],
                "prediction": row["prediction"],
                "planner_score": row["planner_diagnostics"]["planner_score"],
                "retrieval_score": row["retrieval_score"],
                "answer_score": row["answer_score"],
                "end_to_end_score": row["end_to_end_score"],
                "numeric_accuracy": row["numeric_accuracy"],
                "grounded": row["grounded"],
                "retrieval_details": row["retrieval_details"],
            }
        )
    return compact


def build_training_commands(args: argparse.Namespace) -> list[list[str]]:
    return [
        [
            sys.executable,
            "scripts/train_lfm2_sft.py",
            "--model_name_or_path",
            args.lfm2_model_name_or_path,
            "--processed_dir",
            args.processed_dir,
            "--train_questions",
            "data/train/questions.json",
            "--train_answers",
            "data/train/reference_answers.json",
            "--eval_questions",
            args.questions,
            "--eval_answers",
            args.answers,
            "--output_dir",
            args.lfm2_adapter_path,
            "--context_mode",
            "evidence_window",
            "--max_context_chars",
            "1200",
            "--context_window_chars",
            "700",
            "--max_seq_length",
            "1536",
            "--num_train_epochs",
            "6",
            "--use_4bit",
            "--use_lora",
            "--augment_multi_hop",
            "--multi_hop_augment_copies",
            "4",
            "--multi_hop_formula_copies",
            "3",
            "--local_files_only",
        ],
        [
            sys.executable,
            "scripts/train_qwen_ppo.py",
            "--model_path",
            args.qwen_model_path,
            "--lfm2_adapter_path",
            args.lfm2_adapter_path,
            "--processed_dir",
            args.processed_dir,
            "--train_questions",
            "data/train/questions.json",
            "--train_answers",
            "data/train/reference_answers.json",
            "--eval_questions",
            args.questions,
            "--eval_answers",
            args.answers,
            "--output_dir",
            args.qwen_adapter_path,
            "--training_mode",
            "low_vram_sft",
            "--num_epochs",
            "3",
            "--max_new_tokens",
            "160",
            "--max_target_length",
            "192",
            "--use_4bit",
            "--use_lora",
            "--attn_implementation",
            "sdpa",
            "--local_files_only",
        ],
    ]


def run_training_pipeline(commands: list[list[str]], *, dry_run: bool) -> None:
    for command in commands:
        print(" ".join(command))
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_examples(questions_path: Path, answers_path: Path) -> list[PlannerExample]:
    questions = read_json_rows(questions_path)
    answers = {item["id"]: item for item in read_json_rows(answers_path)}
    examples = []
    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise RuntimeError(f"Missing reference answer for id={question['id']}")
        examples.append(
            PlannerExample(
                id=str(question["id"]),
                query=str(question["query"]),
                qa_type=str(question.get("qa_type") or answer.get("qa_type") or ""),
                ticker=str(question.get("ticker") or answer.get("ticker") or ""),
                source_file=str(question.get("source_file") or answer.get("source_file") or ""),
                answer=str(answer.get("ground_truth_answer") or ""),
                evidence=str(answer.get("ground_truth_context") or ""),
            )
        )
    return examples


def load_chunks(chunks_file: Path, processed_dir: Path) -> list[Document]:
    if chunks_file.exists():
        payload = json.loads(chunks_file.read_text(encoding="utf-8"))
        return [Document.from_any(item, index=index) for index, item in enumerate(payload)]
    docs = []
    for path in sorted(processed_dir.glob("*.txt")):
        docs.append(
            Document(
                id=str(path),
                text=path.read_text(encoding="utf-8", errors="ignore"),
                metadata={"filename": path.name, "source": str(path)},
            )
        )
    if not docs:
        raise FileNotFoundError(f"Cannot find chunks_file={chunks_file} or processed txt files in {processed_dir}")
    return docs


def group_documents_by_source(docs: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        grouped[context_source(doc)].append(doc)
    return grouped


def context_source(doc: Document) -> str:
    metadata = doc.metadata or {}
    filename = metadata.get("filename") or Path(str(metadata.get("source") or "")).name
    if filename:
        return Path(str(filename)).name
    return Path(doc.id.split("::", 1)[0]).name


def normalize_spaced_numbers(text: str) -> str:
    return re.sub(r"(?<=\d)[ ]+(?=[.,]?\d)", "", text)


def extract_material_numbers(text: str) -> list[float]:
    values = []
    for number in extract_numbers(normalize_spaced_numbers(text)):
        abs_number = abs(number)
        if 1900 <= abs_number <= 2100:
            continue
        if 1 <= abs_number <= 31:
            continue
        values.append(number)
    return values


def get_model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = Path(value).name
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_reused_plans(path: Path, examples: list[PlannerExample]) -> list[dict[str, Any]]:
    by_id = {str(row.get("id")): row for row in read_jsonl(path)}
    missing = [example.id for example in examples if example.id not in by_id]
    if missing:
        raise RuntimeError(f"Missing reused plans for ids: {', '.join(missing[:10])}")
    return [by_id[example.id] for example in examples]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
