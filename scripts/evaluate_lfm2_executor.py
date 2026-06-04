from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.executor_agent import ExecutorAgent
from rag_engine.schema import Document, RetrievalResult
from tools.evaluation import exact_match, f1_score, normalize_answer


NOT_FOUND = "KHÔNG TÌM THẤY"
LFM2_SYSTEM_PROMPT = (
    "Bạn là executor RAG cho báo cáo tài chính chứng khoán Việt Nam. "
    "Chỉ trả lời dựa trên tài liệu được cung cấp. "
    "Nếu câu hỏi hỏi số liệu, hãy chọn đúng số ở cùng dòng/cột với chỉ tiêu trong câu hỏi và trả về số kèm đơn vị. "
    "Không chọn số ở bảng hoặc dòng khác. Nếu không đủ căn cứ, trả lời: KHÔNG TÌM THẤY."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LFM2 executor separately on QA test data.")
    parser.add_argument("--model_name_or_path", default="models/lfm2_rag")
    parser.add_argument("--adapter_path", default="models/lfm2_rag_lora")
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--questions", default="data/test/questions.json")
    parser.add_argument("--answers", default="data/test/reference_answers.json")
    parser.add_argument("--eval_data_file", default="data/evaluation/lfm2_executor_eval_data.jsonl")
    parser.add_argument("--predictions_file", default="data/evaluation/lfm2_executor_predictions.jsonl")
    parser.add_argument("--metrics_file", default="data/evaluation/lfm2_executor_metrics.json")
    parser.add_argument("--max_context_chars", type=int, default=1200)
    parser.add_argument("--context_window_chars", type=int, default=400)
    parser.add_argument(
        "--context_mode",
        default="evidence",
        choices=["evidence", "evidence_window", "full_window"],
        help="evidence uses only ground_truth_context, the fairest isolated executor test.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--prompt_style", default="sft", choices=["sft", "executor_agent"])
    parser.add_argument("--numeric_rel_tol", type=float, default=0.01)
    parser.add_argument("--percent_abs_tol", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--prepare_only", action="store_true", help="Only create eval_data_file, do not load model.")
    parser.add_argument("--disable_model", action="store_true", help="Use executor heuristic fallback only.")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = build_eval_rows(
        questions_path=Path(args.questions),
        answers_path=Path(args.answers),
        processed_dir=Path(args.processed_dir),
        max_context_chars=args.max_context_chars,
        context_window_chars=args.context_window_chars,
        context_mode=args.context_mode,
    )
    write_jsonl(Path(args.eval_data_file), all_rows)
    rows = all_rows[: args.limit] if args.limit > 0 else all_rows
    if args.prepare_only:
        print(json.dumps({"eval_rows": len(all_rows), "eval_data_file": args.eval_data_file}, ensure_ascii=False, indent=2))
        return

    runner: Any
    if args.disable_model or args.prompt_style == "executor_agent":
        runner = ExecutorAgent(
            model_name=args.model_name_or_path,
            adapter_name=args.adapter_path if Path(args.adapter_path).exists() else None,
            enable_model=not args.disable_model,
            local_files_only=args.local_files_only,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
            fallback_to_heuristic=True,
        )
    else:
        runner = LFM2SFTExecutor(
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.adapter_path if Path(args.adapter_path).exists() else None,
            local_files_only=args.local_files_only,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
        )

    predictions = []
    for index, row in enumerate(rows, start=1):
        if isinstance(runner, ExecutorAgent):
            contexts = [
                RetrievalResult(
                    document=Document(
                        id=row["source_file"],
                        text=row["context"],
                        metadata={"source_file": row["source_file"], "qa_id": row["id"]},
                    ),
                    score=1.0,
                )
            ]
            result = runner.answer(row["query"], contexts)
            prediction = result.answer.strip()
            confidence = result.confidence
            metadata = result.metadata
        else:
            prediction = runner.answer(row).strip()
            confidence = None
            metadata = {"executor": "lfm2_sft_prompt"}
        scored = score_prediction(
            prediction=prediction,
            answer=row["answer"],
            context=row["context"],
            numeric_rel_tol=args.numeric_rel_tol,
            percent_abs_tol=args.percent_abs_tol,
        )
        payload = {
            **row,
            "prediction": prediction,
            "confidence": confidence,
            "executor_metadata": metadata,
            **scored,
        }
        predictions.append(payload)
        print(
            json.dumps(
                {
                    "index": index,
                    "id": row["id"],
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
    print(f"Wrote eval data to {args.eval_data_file}")
    print(f"Wrote predictions to {args.predictions_file}")
    print(f"Wrote metrics to {args.metrics_file}")


def build_eval_rows(
    *,
    questions_path: Path,
    answers_path: Path,
    processed_dir: Path,
    max_context_chars: int,
    context_window_chars: int,
    context_mode: str,
) -> list[dict[str, Any]]:
    questions = read_json_rows(questions_path)
    answers = {item["id"]: item for item in read_json_rows(answers_path)}
    text_cache: dict[Path, str] = {}
    rows = []

    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise RuntimeError(f"Missing reference answer for id={question['id']}")
        if question.get("qa_type") != answer.get("qa_type"):
            raise RuntimeError(f"qa_type mismatch for id={question['id']}")

        source_path = resolve_source_path(processed_dir, question["source_file"])
        if source_path not in text_cache:
            text_cache[source_path] = source_path.read_text(encoding="utf-8", errors="ignore")

        evidence = str(answer.get("ground_truth_context") or "")
        context = select_context(
            document_text=text_cache[source_path],
            evidence=evidence,
            max_context_chars=max_context_chars,
            context_window_chars=context_window_chars,
            context_mode=context_mode,
        )
        rows.append(
            {
                "id": question["id"],
                "query": question["query"],
                "qa_type": question.get("qa_type", ""),
                "ticker": question.get("ticker", ""),
                "source_file": question["source_file"],
                "answer": answer.get("ground_truth_answer", ""),
                "ground_truth_context": evidence,
                "context": context,
            }
        )
    return rows


class LFM2SFTExecutor:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_path: str | None,
        local_files_only: bool,
        load_in_4bit: bool,
        device_map: str,
        max_new_tokens: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("transformers, torch and bitsandbytes are required to evaluate LFM2") from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": True,
            "device_map": device_map,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("peft is required to load LFM2 adapter") from exc
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        self.model = model

    def answer(self, row: dict[str, Any]) -> str:
        user_prompt = (
            f"Tài liệu nguồn: {row['source_file']}\n"
            f"Mã cổ phiếu: {row['ticker']}\n"
            f"Loại câu hỏi: {row['qa_type']}\n\n"
            f"<context>\n{row['context']}\n</context>\n\n"
            f"Câu hỏi: {row['query']}\n"
            "Chỉ trả lời đáp án cuối cùng, ngắn gọn. "
            "Nếu đáp án là số tiền, chỉ trả về số tiền đúng và đơn vị VND. Không chép lại câu hỏi hoặc bảng."
        )
        messages = [
            {"role": "system", "content": LFM2_SYSTEM_PROMPT},
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
            prompt = f"System: {LFM2_SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt")

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
        prompt_length = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_length:]
        return clean_prediction(self.tokenizer.decode(generated, skip_special_tokens=True))


def get_model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def clean_prediction(text: str) -> str:
    value = text.strip().strip("\"'“”‘’")
    value = re.sub(r"^Câu trả lời(?: nguyên văn)?:\s*", "", value, flags=re.IGNORECASE)
    if "KHÔNG TÌM THẤY" in value.upper():
        return NOT_FOUND
    return value.strip()


def score_prediction(
    *,
    prediction: str,
    answer: str,
    context: str,
    numeric_rel_tol: float,
    percent_abs_tol: float,
) -> dict[str, Any]:
    em = exact_match(prediction, answer)
    f1 = f1_score(prediction, answer)
    numeric = numeric_accuracy(
        prediction=prediction,
        answer=answer,
        rel_tol=numeric_rel_tol,
        percent_abs_tol=percent_abs_tol,
    )
    grounded = groundedness(prediction, context)
    numeric_value = 1.0 if numeric is None else numeric
    lfm2_score = 0.30 * em + 0.30 * f1 + 0.25 * numeric_value + 0.15 * float(grounded)
    return {
        "em": em,
        "f1": f1,
        "numeric_accuracy": numeric,
        "grounded": grounded,
        "lfm2_score": lfm2_score,
    }


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


def extract_numbers(text: str) -> list[float]:
    values = []
    pattern = re.compile(r"\(?-?\d[\d.,]*\)?")
    for match in pattern.finditer(text):
        raw = match.group(0)
        parsed = parse_vietnamese_number(raw)
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
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
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


def groundedness(prediction: str, context: str) -> bool:
    if not prediction or prediction.strip().upper() == NOT_FOUND:
        return False
    pred_norm = normalize_answer(prediction)
    context_norm = normalize_answer(context)
    if pred_norm and pred_norm in context_norm:
        return True
    pred_numbers = extract_numbers(prediction)
    if pred_numbers:
        context_numbers = extract_numbers(context)
        return all(any(numbers_close(num, ctx, rel_tol=0.0, percent_abs_tol=0.0) for ctx in context_numbers) for num in pred_numbers)
    return False


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"count": 0}

    numeric_rows = [row for row in rows if row["numeric_accuracy"] is not None]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row.get("qa_type", ""), []).append(row)

    metrics = {
        "count": count,
        "exact_match": mean(row["em"] for row in rows),
        "f1": mean(row["f1"] for row in rows),
        "numeric_accuracy": mean(row["numeric_accuracy"] for row in numeric_rows) if numeric_rows else None,
        "groundedness": mean(float(row["grounded"]) for row in rows),
        "lfm2_score": mean(row["lfm2_score"] for row in rows),
        "by_qa_type": {},
    }
    for qa_type, items in sorted(by_type.items()):
        qa_numeric = [row for row in items if row["numeric_accuracy"] is not None]
        metrics["by_qa_type"][qa_type] = {
            "count": len(items),
            "exact_match": mean(row["em"] for row in items),
            "f1": mean(row["f1"] for row in items),
            "numeric_accuracy": mean(row["numeric_accuracy"] for row in qa_numeric) if qa_numeric else None,
            "groundedness": mean(float(row["grounded"]) for row in items),
            "lfm2_score": mean(row["lfm2_score"] for row in items),
        }
    return metrics


def mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [dict(item) for item in payload]
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


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


def select_context(
    *,
    document_text: str,
    evidence: str,
    max_context_chars: int,
    context_window_chars: int,
    context_mode: str,
) -> str:
    if context_mode == "evidence" and evidence:
        return evidence.strip()[:max_context_chars]

    if evidence and evidence in document_text:
        start = max(document_text.find(evidence) - context_window_chars // 2, 0)
        end = min(start + max_context_chars, len(document_text))
        return document_text[start:end].strip()
    if evidence:
        combined = (evidence.strip() + "\n\n" + document_text).strip()
        return combined[:max_context_chars].strip()
    return document_text[:max_context_chars].strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
