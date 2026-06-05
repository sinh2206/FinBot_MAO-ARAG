#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from rag_engine.retriever import BM25Index
from rag_engine.schema import Document
from tools.evaluation import exact_match, f1_score, normalize_answer
from tools.text_splitter import ChunkConfig, split_documents


SYSTEM_PROMPT = (
    "Bạn là Qwen executor cho hệ thống RAG báo cáo tài chính chứng khoán Việt Nam. "
    "Chỉ trả lời dựa trên context được cung cấp. "
    "Nếu câu hỏi là multi-hop, hãy đọc từng bằng chứng thành phần rồi mới tổng hợp đáp án cuối cùng. "
    "Nếu câu hỏi hỏi số liệu, hãy trả về số kèm đơn vị phù hợp. "
    "Nếu không đủ căn cứ, trả lời: KHÔNG TÌM THẤY."
)
NOT_FOUND = "KHÔNG TÌM THẤY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen executor on QA data.")
    parser.add_argument("--model_name_or_path", default="models/qwen")
    parser.add_argument("--adapter_path", default="models/qwen_executor_lora")
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--chunks_file", default="data/chunks/chunks.json")
    parser.add_argument("--questions", default="data/test/questions.json")
    parser.add_argument("--answers", default="data/test/reference_answers.json")
    parser.add_argument("--predictions_file", default="data/evaluation/qwen_executor_predictions.jsonl")
    parser.add_argument("--metrics_file", default="data/evaluation/qwen_executor_metrics.json")
    parser.add_argument("--qa_type_filter", default="all", choices=["all", "single_hop", "multi_hop"])
    parser.add_argument("--component_top_k", type=int, default=3)
    parser.add_argument("--max_context_chars", type=int, default=1800)
    parser.add_argument("--context_window_chars", type=int, default=480)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--numeric_rel_tol", type=float, default=0.01)
    parser.add_argument("--percent_abs_tol", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--gpu_memory_limit", default=None)
    parser.add_argument("--enable_gemini_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback_model_name", default=load_dotenv_value("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
    parser.add_argument("--fallback_api_key", default=load_dotenv_value("GEMINI_API_KEY", None))
    parser.add_argument("--fallback_temperature", type=float, default=0.0)
    parser.add_argument("--fallback_max_output_tokens", type=int, default=int(load_dotenv_value("GEMINI_FALLBACK_MAX_OUTPUT_TOKENS", "64") or "64"))
    parser.add_argument("--fallback_thinking_budget", type=int, default=int(load_dotenv_value("GEMINI_FALLBACK_THINKING_BUDGET", "0") or "0"))
    parser.add_argument("--fallback_context_chars", type=int, default=int(load_dotenv_value("GEMINI_FALLBACK_CONTEXT_CHARS", "900") or "900"))
    parser.add_argument("--fallback_max_snippets", type=int, default=int(load_dotenv_value("GEMINI_FALLBACK_MAX_SNIPPETS", "2") or "2"))
    parser.add_argument("--fallback_max_snippet_chars", type=int, default=int(load_dotenv_value("GEMINI_FALLBACK_MAX_SNIPPET_CHARS", "220") or "220"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_eval_rows(
        questions_path=Path(args.questions),
        answers_path=Path(args.answers),
        processed_dir=Path(args.processed_dir),
        chunks_file=Path(args.chunks_file),
        component_top_k=args.component_top_k,
        max_context_chars=args.max_context_chars,
        context_window_chars=args.context_window_chars,
    )
    if args.qa_type_filter != "all":
        rows = [row for row in rows if row["qa_type"] == args.qa_type_filter]
    rows = rows[: args.limit] if args.limit > 0 else rows
    if args.prepare_only:
        print(json.dumps({"rows": len(rows), "first": rows[0] if rows else None}, ensure_ascii=False, indent=2))
        return

    runner = QwenExecutorRunner(args)
    fallback_runner = GeminiFallbackRunner(args) if args.enable_gemini_fallback and args.fallback_api_key else None
    predictions = []
    for index, row in enumerate(rows, start=1):
        prediction = runner.answer(row)
        executor_used = "qwen"
        if fallback_runner and is_not_found(prediction):
            prediction = fallback_runner.answer(row)
            executor_used = "gemini"
        scored = score_prediction(
            prediction=prediction,
            answer=row["answer"],
            context=row["context"],
            qa_type=row["qa_type"],
            component_results=row["component_results"],
            ground_truth_context=row["ground_truth_context"],
            numeric_rel_tol=args.numeric_rel_tol,
            percent_abs_tol=args.percent_abs_tol,
        )
        payload = {**row, "prediction": prediction, "executor_used": executor_used, **scored}
        predictions.append(payload)
        print(
            json.dumps(
                {
                    "index": index,
                    "id": row["id"],
                    "executor": executor_used,
                    "qa_type": row["qa_type"],
                    "component_support_rate": payload["component_support_rate"],
                    "evidence_number_coverage": payload["evidence_number_coverage"],
                    "em": payload["em"],
                    "f1": round(payload["f1"], 4),
                    "numeric_accuracy": payload["numeric_accuracy"],
                    "grounded": payload["grounded"],
                    "prediction": prediction[:180],
                },
                ensure_ascii=False,
            )
        )

    metrics = summarize(predictions)
    write_jsonl(Path(args.predictions_file), predictions)
    write_json(Path(args.metrics_file), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote predictions to {args.predictions_file}")
    print(f"Wrote metrics to {args.metrics_file}")


class QwenExecutorRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install torch, transformers, peft and bitsandbytes to evaluate Qwen executor.") from exc

        self.torch = torch
        self.max_new_tokens = args.max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": args.local_files_only,
            "device_map": args.device_map,
        }
        if args.gpu_memory_limit and args.device_map == "auto":
            model_kwargs["max_memory"] = {0: args.gpu_memory_limit}
        if args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        if args.adapter_path and Path(args.adapter_path).exists():
            model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
        model.eval()
        self.model = model

    def answer(self, row: dict[str, Any]) -> str:
        component_block = format_component_block(row["component_results"])
        user_prompt = (
            f"Tài liệu nguồn: {row['source_file']}\n"
            f"Mã cổ phiếu: {row['ticker']}\n"
            f"Loại câu hỏi: {row['qa_type']}\n\n"
            f"<sub_queries>\n{component_block}\n</sub_queries>\n\n"
            f"<context>\n{row['context']}\n</context>\n\n"
            f"Câu hỏi cuối cùng: {row['query']}\n"
            "Chỉ trả lời đáp án cuối cùng, ngắn gọn. Không chép lại bảng."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = self.tokenizer(f"System: {SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:", return_tensors="pt")

        device = get_model_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return clean_prediction(self.tokenizer.decode(generated, skip_special_tokens=True))


class GeminiFallbackRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install google-genai to enable the Gemini fallback executor.") from exc
        self.client = genai.Client(api_key=args.fallback_api_key)
        self.types = types
        self.model_name = args.fallback_model_name
        self.temperature = args.fallback_temperature
        self.max_output_tokens = args.fallback_max_output_tokens
        self.thinking_budget = args.fallback_thinking_budget
        self.max_context_chars = args.fallback_context_chars
        self.max_snippets = args.fallback_max_snippets
        self.max_snippet_chars = args.fallback_max_snippet_chars

    def answer(self, row: dict[str, Any]) -> str:
        user_prompt = build_gemini_fallback_prompt(
            row=row,
            max_context_chars=self.max_context_chars,
            max_snippets=self.max_snippets,
            max_snippet_chars=self.max_snippet_chars,
        )
        config = self.types.GenerateContentConfig(
            system_instruction=build_gemini_fallback_system_prompt(),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=self.types.ThinkingConfig(thinking_budget=self.thinking_budget),
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )
        text = getattr(response, "text", None) or str(response)
        return clean_prediction(text)


def build_eval_rows(
    *,
    questions_path: Path,
    answers_path: Path,
    processed_dir: Path,
    chunks_file: Path,
    component_top_k: int,
    max_context_chars: int,
    context_window_chars: int,
) -> list[dict[str, Any]]:
    questions = read_json_rows(questions_path)
    answers = {item["id"]: item for item in read_json_rows(answers_path)}
    chunk_map = load_chunk_map(chunks_file)
    text_cache: dict[Path, str] = {}
    doc_cache: dict[str, list[Document]] = {}
    rows = []

    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise RuntimeError(f"Missing reference answer for id={question['id']}")
        source_path = resolve_source_path(processed_dir, question["source_file"])
        if source_path not in text_cache:
            text_cache[source_path] = source_path.read_text(encoding="utf-8", errors="ignore")
        source_text = text_cache[source_path]
        source_key = canonical_source_name(question["source_file"])
        if source_key not in doc_cache:
            docs = chunk_map.get(source_key) or build_documents_from_text(source_text, source_key)
            doc_cache[source_key] = docs
        source_docs = doc_cache[source_key]

        component_queries = build_component_queries(
            query=question["query"],
            qa_type=str(question.get("qa_type", "")),
            ticker=str(question.get("ticker", "")),
        )
        component_results = evaluate_component_queries(
            component_queries=component_queries,
            source_docs=source_docs,
            source_text=source_text,
            top_k=component_top_k,
            window_chars=context_window_chars,
        )
        context = build_context_from_components(component_results, max_context_chars=max_context_chars)
        rows.append(
            {
                "id": question["id"],
                "query": question["query"],
                "qa_type": question.get("qa_type", ""),
                "ticker": question.get("ticker", ""),
                "source_file": question["source_file"],
                "answer": answer.get("ground_truth_answer", ""),
                "ground_truth_context": str(answer.get("ground_truth_context") or ""),
                "component_queries": component_queries,
                "component_results": component_results,
                "context": context,
            }
        )
    return rows


def build_component_queries(*, query: str, qa_type: str, ticker: str) -> list[dict[str, Any]]:
    if qa_type != "multi_hop":
        return [{"id": "q1", "query": query, "type": "retrieval_qa", "required": True}]

    normalized = normalize_answer(query)
    if "hai mang" in normalized or "hai khoan" in normalized:
        return [
            {
                "id": "q1",
                "query": f"Báo cáo bộ phận của {ticker}: giá trị thứ nhất cần để trả lời câu hỏi '{query}' là gì?",
                "type": "retrieval_qa",
                "required": True,
            },
            {
                "id": "q2",
                "query": f"Báo cáo bộ phận của {ticker}: giá trị thứ hai cần để trả lời câu hỏi '{query}' là gì?",
                "type": "retrieval_qa",
                "required": True,
            },
            {
                "id": "q3",
                "query": f"Tổng hợp hai giá trị cần trả lời cho câu hỏi: {query}",
                "type": "synthesis",
                "required": False,
                "depends_on": ["q1", "q2"],
            },
        ]

    if "chiem" in normalized or "phan tram" in normalized:
        numerator_query, denominator_query = build_ratio_queries(query, ticker)
        return [
            {"id": "q1", "query": numerator_query, "type": "retrieval_qa", "required": True},
            {"id": "q2", "query": denominator_query, "type": "retrieval_qa", "required": True},
            {
                "id": "q3",
                "query": f"Phép tính tỷ lệ cần thiết để trả lời: {query}",
                "type": "calculation",
                "required": False,
                "depends_on": ["q1", "q2"],
            },
        ]

    if any(marker in normalized for marker in ("tang bao nhieu", "giam bao nhieu", "so voi", "chenh lech", "bao nhieu diem")):
        current_query, compare_query = build_comparison_queries(query, ticker)
        return [
            {"id": "q1", "query": current_query, "type": "retrieval_qa", "required": True},
            {"id": "q2", "query": compare_query, "type": "retrieval_qa", "required": True},
            {
                "id": "q3",
                "query": f"Phép tính chênh lệch cần thiết để trả lời: {query}",
                "type": "calculation",
                "required": False,
                "depends_on": ["q1", "q2"],
            },
        ]

    return [
        {"id": "q1", "query": query, "type": "retrieval_qa", "required": True},
        {
            "id": "q2",
            "query": f"Đoạn thuyết minh hoặc bằng chứng trực tiếp trong báo cáo cho câu hỏi: {query}",
            "type": "retrieval_qa",
            "required": True,
        },
    ]


def build_comparison_queries(query: str, ticker: str) -> tuple[str, str]:
    current_query = query
    compare_query = query
    marker_match = re.search(
        r"\s+(tăng bao nhiêu|giảm bao nhiêu điểm|giảm bao nhiêu|tăng bao nhiêu điểm|bao nhiêu điểm so với|so với)\s+",
        query,
        flags=re.IGNORECASE,
    )
    periods = extract_periods(query)
    if marker_match and periods:
        left = query[: marker_match.start()].strip().rstrip(",:;")
        current_period = periods[0]
        base = remove_first_occurrence(left, current_period).strip()
        current_query = ensure_question_suffix(left)
        if len(periods) >= 2:
            compare_period = periods[1]
            compare_query = ensure_question_suffix(insert_period_before_ticker(base, compare_period, ticker))
        else:
            compare_query = ensure_question_suffix(f"Giá trị kỳ so sánh của {ticker} cho chỉ tiêu: {query}")
    else:
        current_query = ensure_question_suffix(query)
        compare_query = ensure_question_suffix(f"Giá trị kỳ so sánh của {ticker} cho chỉ tiêu: {query}")
    return current_query, compare_query


def build_ratio_queries(query: str, ticker: str) -> tuple[str, str]:
    left, _, right = query.partition("chiếm")
    numerator = ensure_question_suffix(ensure_ticker(left.strip().rstrip(",."), ticker))
    periods = extract_periods(query)
    current_period = periods[0] if periods else ""
    base = remove_first_occurrence(left.strip(), current_period).strip() if current_period else left.strip()
    denominator_period = extract_denominator_period(right) or (periods[1] if len(periods) >= 2 else "lũy kế 9 tháng")
    denominator = ensure_question_suffix(insert_period_before_ticker(base, denominator_period, ticker))
    return numerator, denominator


def extract_periods(text: str) -> list[str]:
    patterns = [
        r"Quý\s*[IVXLC]+\s*/\s*\d{4}",
        r"Quy\s*[IVXLC]+\s*/\s*\d{4}",
        r"lũy kế\s*\d+\s*tháng(?:\s*\d{4})?",
        r"luy ke\s*\d+\s*tháng(?:\s*\d{4})?",
        r"\d+\s*tháng đầu năm\s*\d{4}",
        r"\d+\s*tháng\s*\d{4}",
        r"\d{2}/\d{2}/\d{4}",
        r"đầu năm(?:\s*\d{4})?",
        r"cuối kỳ",
        r"cùng kỳ",
    ]
    seen: set[str] = set()
    items: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            key = normalize_answer(value)
            if key in seen:
                continue
            seen.add(key)
            items.append((match.start(), value))
    items.sort(key=lambda item: item[0])
    return [value for _, value in items]


def extract_denominator_period(text: str) -> str | None:
    lowered = text.lower()
    patterns = [
        r"lũy kế\s*\d+\s*tháng(?:\s*\d{4})?",
        r"\d+\s*tháng đầu năm\s*\d{4}",
        r"\d+\s*tháng\s*\d{4}",
        r"\d{2}/\d{2}/\d{4}",
        r"cùng kỳ",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return text[match.start() : match.end()].strip()
    return None


def ensure_ticker(text: str, ticker: str) -> str:
    if not ticker:
        return text
    normalized = normalize_answer(text)
    if normalize_answer(ticker) in normalized:
        return text
    return f"{text} của {ticker}".strip()


def ensure_question_suffix(text: str) -> str:
    value = text.strip().rstrip(".")
    if not value.endswith("?"):
        if re.search(r"\b(là gì|bao nhiêu|như thế nào)\s*$", value, flags=re.IGNORECASE):
            return value + "?"
        return value + " là bao nhiêu?"
    return value


def remove_first_occurrence(text: str, phrase: str) -> str:
    if not phrase:
        return text
    return re.sub(re.escape(phrase), "", text, count=1, flags=re.IGNORECASE).replace("  ", " ").strip()


def insert_period_before_ticker(base: str, period: str, ticker: str) -> str:
    cleaned = base.strip().rstrip(",.")
    if ticker and re.search(rf"\bcủa\s+{re.escape(ticker)}\b", cleaned, flags=re.IGNORECASE):
        return re.sub(rf"\bcủa\s+{re.escape(ticker)}\b", f"{period} của {ticker}", cleaned, count=1, flags=re.IGNORECASE)
    if ticker and re.search(rf"\b{re.escape(ticker)}\b", cleaned, flags=re.IGNORECASE):
        return re.sub(rf"\b{re.escape(ticker)}\b", f"{period} của {ticker}", cleaned, count=1, flags=re.IGNORECASE)
    return ensure_ticker(f"{cleaned} {period}".strip(), ticker)


def evaluate_component_queries(
    *,
    component_queries: list[dict[str, Any]],
    source_docs: list[Document],
    source_text: str,
    top_k: int,
    window_chars: int,
) -> list[dict[str, Any]]:
    bm25 = BM25Index(source_docs)
    results = []
    for item in component_queries:
        query_type = str(item.get("type") or "retrieval_qa")
        if query_type not in {"retrieval_qa", "retrieval"}:
            results.append(
                {
                    **item,
                    "supported": None,
                    "evidence_snippets": [],
                    "top_score": None,
                }
            )
            continue
        evidence = search_evidence(
            query=str(item["query"]),
            bm25=bm25,
            source_text=source_text,
            top_k=top_k,
            window_chars=window_chars,
        )
        results.append(
            {
                **item,
                "supported": bool(evidence),
                "evidence_snippets": evidence,
                "top_score": evidence[0]["score"] if evidence else None,
            }
        )
    return results


def search_evidence(
    *,
    query: str,
    bm25: BM25Index,
    source_text: str,
    top_k: int,
    window_chars: int,
) -> list[dict[str, Any]]:
    evidences: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in bm25.search(query, top_k=top_k):
        snippet = focus_text_on_query(result.document.text, query=query, window_chars=window_chars)
        norm = normalize_answer(snippet)
        if not snippet or norm in seen:
            continue
        seen.add(norm)
        evidences.append(
            {
                "snippet": snippet,
                "score": float(result.score),
                "source": str(result.document.metadata.get("source_file") or result.document.id),
                "mode": "chunk_bm25",
            }
        )
    if evidences:
        return evidences

    fallback = focus_text_on_query(source_text, query=query, window_chars=window_chars)
    if fallback.strip():
        evidences.append(
            {
                "snippet": fallback.strip(),
                "score": 0.0,
                "source": "processed_data_window",
                "mode": "full_text_window",
            }
        )
    return evidences


def build_context_from_components(component_results: list[dict[str, Any]], *, max_context_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for item in component_results:
        snippets = item.get("evidence_snippets") or []
        if not snippets:
            continue
        lines = [f"[{item['id']}] {item['query']}"]
        for evidence in snippets:
            lines.append(evidence["snippet"])
        block = "\n".join(lines).strip()
        if not block:
            continue
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks).strip()


def format_component_block(component_results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in component_results:
        status = "supported" if item.get("supported") else ("n/a" if item.get("supported") is None else "missing")
        lines.append(f"[{item['id']}] ({item.get('type', 'retrieval_qa')}, {status}) {item['query']}")
        for evidence in item.get("evidence_snippets") or []:
            lines.append(f"- {evidence['snippet']}")
    return "\n".join(lines).strip()


def build_gemini_fallback_system_prompt() -> str:
    return (
        "Bạn là Gemini fallback cực ngắn cho hệ thống RAG báo cáo tài chính chứng khoán Việt Nam. "
        "Chỉ dùng dữ liệu trong <sub_queries> và <context>. "
        "Nhiệm vụ của bạn chỉ là tổng hợp đáp án cuối cùng khi Qwen không trả lời được. "
        "Nếu chưa đủ căn cứ, trả lời đúng: KHÔNG TÌM THẤY. "
        "Chỉ trả lời 1 dòng, không giải thích, không nhắc lại câu hỏi, giữ nguyên đơn vị. "
        "Nếu câu hỏi hỏi hai giá trị, trả về cả hai giá trị, không cộng."
    )


def build_gemini_fallback_prompt(
    *,
    row: dict[str, Any],
    max_context_chars: int,
    max_snippets: int,
    max_snippet_chars: int,
) -> str:
    component_block = format_component_block_for_gemini(
        row.get("component_results") or [],
        max_snippets=max_snippets,
        max_snippet_chars=max_snippet_chars,
    )
    compact_context = compact_text(str(row.get("context") or ""), max_context_chars)
    return (
        f"Tài liệu nguồn: {row['source_file']}\n"
        f"Mã cổ phiếu: {row['ticker']}\n"
        f"Loại câu hỏi: {row['qa_type']}\n\n"
        f"<sub_queries>\n{component_block}\n</sub_queries>\n\n"
        f"<context>\n{compact_context}\n</context>\n\n"
        f"Câu hỏi cuối cùng: {row['query']}\n"
        "Trả lời đúng 1 dòng với đáp án cuối cùng."
    )


def format_component_block_for_gemini(
    component_results: list[dict[str, Any]],
    *,
    max_snippets: int,
    max_snippet_chars: int,
) -> str:
    lines: list[str] = []
    for item in component_results:
        query_type = str(item.get("type") or "")
        if query_type not in {"retrieval_qa", "retrieval"}:
            continue
        status = "supported" if item.get("supported") else "missing"
        lines.append(f"[{item['id']}] ({status}) {item['query']}")
        snippets = item.get("evidence_snippets") or []
        for evidence in snippets[:max_snippets]:
            lines.append(f"- {compact_text(str(evidence.get('snippet') or ''), max_snippet_chars)}")
    return "\n".join(lines).strip()


def compact_text(text: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def score_prediction(
    *,
    prediction: str,
    answer: str,
    context: str,
    qa_type: str,
    component_results: list[dict[str, Any]],
    ground_truth_context: str,
    numeric_rel_tol: float,
    percent_abs_tol: float,
) -> dict[str, Any]:
    em = exact_match(prediction, answer)
    f1 = f1_score(prediction, answer)
    numeric = numeric_accuracy(prediction=prediction, answer=answer, rel_tol=numeric_rel_tol, percent_abs_tol=percent_abs_tol)
    grounded = groundedness(prediction, context)
    numeric_value = 1.0 if numeric is None else numeric
    score = 0.30 * em + 0.30 * f1 + 0.25 * numeric_value + 0.15 * float(grounded)
    component_support_rate, all_components_supported = component_support_metrics(component_results)
    evidence_number_coverage = score_evidence_number_coverage(
        ground_truth_context=ground_truth_context,
        component_results=component_results,
        rel_tol=numeric_rel_tol,
        percent_abs_tol=percent_abs_tol,
    )
    return {
        "em": em,
        "f1": f1,
        "numeric_accuracy": numeric,
        "grounded": grounded,
        "executor_score": score,
        "component_support_rate": component_support_rate if qa_type == "multi_hop" else None,
        "all_components_supported": all_components_supported if qa_type == "multi_hop" else None,
        "evidence_number_coverage": evidence_number_coverage if qa_type == "multi_hop" else None,
    }


def component_support_metrics(component_results: list[dict[str, Any]]) -> tuple[float | None, bool | None]:
    retrieval_items = [item for item in component_results if str(item.get("type")) in {"retrieval_qa", "retrieval"}]
    if not retrieval_items:
        return None, None
    supported = sum(1 for item in retrieval_items if item.get("supported"))
    rate = supported / len(retrieval_items)
    return rate, supported == len(retrieval_items)


def score_evidence_number_coverage(
    *,
    ground_truth_context: str,
    component_results: list[dict[str, Any]],
    rel_tol: float,
    percent_abs_tol: float,
) -> float | None:
    gold_numbers = extract_numbers(ground_truth_context)
    if not gold_numbers:
        return None
    evidence_text = "\n".join(
        evidence["snippet"]
        for item in component_results
        for evidence in (item.get("evidence_snippets") or [])
    )
    evidence_numbers = extract_numbers(evidence_text)
    if not evidence_numbers:
        return 0.0
    matched = 0
    for gold in gold_numbers:
        if any(numbers_close(gold, current, rel_tol=rel_tol, percent_abs_tol=percent_abs_tol) for current in evidence_numbers):
            matched += 1
    return matched / len(gold_numbers)


def is_not_found(prediction: str) -> bool:
    normalized = normalize_answer(prediction)
    return not normalized or NOT_FOUND in prediction.upper()


def numeric_accuracy(*, prediction: str, answer: str, rel_tol: float, percent_abs_tol: float) -> float | None:
    gold_numbers = extract_numbers(answer)
    if not gold_numbers:
        return None
    pred_numbers = extract_numbers(prediction)
    if not pred_numbers:
        return 0.0
    matched = 0
    for gold in gold_numbers:
        if any(numbers_close(gold, pred, rel_tol=rel_tol, percent_abs_tol=percent_abs_tol) for pred in pred_numbers):
            matched += 1
    return matched / len(gold_numbers)


def groundedness(prediction: str, context: str) -> bool:
    if not prediction or prediction.strip().upper() == NOT_FOUND:
        return False
    pred_norm = normalize_answer(prediction)
    context_norm = normalize_answer(context)
    if pred_norm and pred_norm in context_norm:
        return True
    pred_numbers = extract_numbers(prediction)
    context_numbers = extract_numbers(context)
    return bool(pred_numbers) and all(
        any(numbers_close(num, ctx, rel_tol=0.0, percent_abs_tol=0.0) for ctx in context_numbers)
        for num in pred_numbers
    )


def extract_numbers(text: str) -> list[float]:
    values = []
    for match in re.finditer(r"\(?-?\d[\d.,]*\)?", text):
        parsed = parse_vietnamese_number(match.group(0))
        if parsed is not None and math.isfinite(parsed):
            values.append(parsed)
    return values


def parse_vietnamese_number(raw: str) -> float | None:
    value = raw.strip().strip(".,;:")
    negative = value.startswith("-") or (value.startswith("(") and value.endswith(")"))
    value = value.strip("()").lstrip("-").strip(".,;:")
    if not re.search(r"\d", value):
        return None
    if "." in value and "," in value:
        value = value.replace(".", "").replace(",", ".") if value.rfind(",") > value.rfind(".") else value.replace(",", "")
    elif "," in value:
        tail = value.rsplit(",", 1)[-1]
        value = value.replace(",", ".") if len(tail) <= 2 else value.replace(",", "")
    elif "." in value:
        parts = value.split(".")
        value = value.replace(".", "") if all(len(part) == 3 for part in parts[1:]) else value
    try:
        number = float(value)
    except ValueError:
        return None
    return -number if negative else number


def numbers_close(gold: float, pred: float, *, rel_tol: float, percent_abs_tol: float) -> bool:
    if abs(gold) <= 100:
        return abs(gold - pred) <= percent_abs_tol
    return abs(gold - pred) / max(abs(gold), 1.0) <= rel_tol


def load_chunk_map(chunks_file: Path) -> dict[str, list[Document]]:
    if not chunks_file.exists():
        return {}
    payload = json.loads(chunks_file.read_text(encoding="utf-8"))
    by_source: dict[str, list[Document]] = {}
    for index, item in enumerate(payload):
        document = Document.from_any(item, index=index)
        source_file = canonical_source_name(
            str(
                document.metadata.get("source_file")
                or document.metadata.get("file_name")
                or document.metadata.get("relative_path")
                or ""
            )
        )
        if not source_file:
            continue
        by_source.setdefault(source_file, []).append(document)
    return by_source


def canonical_source_name(source_file: str) -> str:
    return Path(source_file).name


def build_documents_from_text(source_text: str, source_file: str) -> list[Document]:
    base_doc = Document(
        id=f"{source_file}:full",
        text=source_text,
        metadata={"source_file": source_file, "file_name": source_file},
    )
    return split_documents([base_doc], config=ChunkConfig(chunk_size=256, chunk_overlap_ratio=0.15))


def focus_text_on_query(text: str, *, query: str, window_chars: int) -> str:
    normalized = text.strip()
    if len(normalized) <= window_chars:
        return normalized
    terms = [term for term in re.findall(r"\w+", normalize_answer(query), flags=re.UNICODE) if len(term) >= 3]
    lowered = normalize_answer(normalized)
    if not terms:
        return normalized[:window_chars].strip()

    best_pos = 0
    best_score = -1
    step = max(50, window_chars // 4)
    for start in range(0, len(normalized), step):
        segment = lowered[start : start + window_chars]
        score = sum(1 for term in terms if term in segment)
        if score > best_score:
            best_score = score
            best_pos = start
    start = max(0, best_pos - min(80, window_chars // 6))
    end = min(len(normalized), start + window_chars)
    if end - start < window_chars:
        start = max(0, end - window_chars)
    return normalized[start:end].strip()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"count": 0}
    numeric_rows = [row for row in rows if row["numeric_accuracy"] is not None]
    component_rows = [row for row in rows if row["component_support_rate"] is not None]
    evidence_rows = [row for row in rows if row["evidence_number_coverage"] is not None]
    by_qa_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_qa_type.setdefault(str(row["qa_type"]), []).append(row)

    metrics: dict[str, Any] = {
        "count": count,
        "exact_match": mean(row["em"] for row in rows),
        "f1": mean(row["f1"] for row in rows),
        "numeric_accuracy": mean(row["numeric_accuracy"] for row in numeric_rows) if numeric_rows else None,
        "groundedness": mean(float(row["grounded"]) for row in rows),
        "executor_score": mean(row["executor_score"] for row in rows),
        "component_support_rate": mean(row["component_support_rate"] for row in component_rows) if component_rows else None,
        "all_components_supported_rate": mean(float(row["all_components_supported"]) for row in component_rows) if component_rows else None,
        "evidence_number_coverage": mean(row["evidence_number_coverage"] for row in evidence_rows) if evidence_rows else None,
        "by_qa_type": {},
    }
    for qa_type, items in sorted(by_qa_type.items()):
        qa_numeric = [row for row in items if row["numeric_accuracy"] is not None]
        qa_component = [row for row in items if row["component_support_rate"] is not None]
        qa_evidence = [row for row in items if row["evidence_number_coverage"] is not None]
        metrics["by_qa_type"][qa_type] = {
            "count": len(items),
            "exact_match": mean(row["em"] for row in items),
            "f1": mean(row["f1"] for row in items),
            "numeric_accuracy": mean(row["numeric_accuracy"] for row in qa_numeric) if qa_numeric else None,
            "groundedness": mean(float(row["grounded"]) for row in items),
            "executor_score": mean(row["executor_score"] for row in items),
            "component_support_rate": mean(row["component_support_rate"] for row in qa_component) if qa_component else None,
            "all_components_supported_rate": mean(float(row["all_components_supported"]) for row in qa_component) if qa_component else None,
            "evidence_number_coverage": mean(row["evidence_number_coverage"] for row in qa_evidence) if qa_evidence else None,
        }
    return metrics


def clean_prediction(text: str) -> str:
    value = text.strip().strip("\"'“”‘’")
    value = re.sub(r"^Câu trả lời(?: nguyên văn)?:\s*", "", value, flags=re.IGNORECASE)
    return NOT_FOUND if "KHÔNG TÌM THẤY" in value.upper() else value.strip()


def resolve_source_path(processed_dir: Path, source_file: str) -> Path:
    candidates = [processed_dir / source_file]
    if source_file.endswith(".ocr_text.txt"):
        candidates.append(processed_dir / source_file.replace(".ocr_text.txt", ".txt"))
    if source_file.endswith(".txt"):
        candidates.append(processed_dir / source_file.replace(".txt", ".ocr_text.txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve source_file={source_file} in {processed_dir}")


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [dict(item) for item in payload]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_dotenv_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return default
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() == name:
            return val.strip().strip("\"'") or default
    return default


def get_model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
