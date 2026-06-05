#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


SYSTEM_PROMPT = (
    "Bạn là Phi planner/coordinator cho hệ thống RAG báo cáo tài chính chứng khoán Việt Nam. "
    "Nhiệm vụ của bạn là lập kế hoạch truy xuất cho executor. "
    "Luôn trả về JSON hợp lệ, không thêm giải thích ngoài JSON."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Phi planner/coordinator on QA data.")
    parser.add_argument("--model_name_or_path", default="models/phi")
    parser.add_argument("--adapter_path", default="models/phi_planner_lora")
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--questions", default="data/test/questions.json")
    parser.add_argument("--answers", default="data/test/reference_answers.json")
    parser.add_argument("--predictions_file", default="data/evaluation/phi_planner_predictions.jsonl")
    parser.add_argument("--metrics_file", default="data/evaluation/phi_planner_metrics.json")
    parser.add_argument("--max_sources", type=int, default=80)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--gpu_memory_limit", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = list_processed_sources(Path(args.processed_dir), max_sources=args.max_sources)
    examples = build_examples(Path(args.questions), Path(args.answers), sources)
    examples = examples[: args.limit] if args.limit > 0 else examples
    if args.prepare_only:
        print(json.dumps({"rows": len(examples), "sources": sources, "first": examples[0] if examples else None}, ensure_ascii=False, indent=2))
        return

    runner = PhiPlannerRunner(args)
    predictions = []
    for index, item in enumerate(examples, start=1):
        prompt = format_prompt(item)
        raw = runner.generate(prompt)
        plan = extract_json(raw)
        scored = score_plan(plan, item)
        payload = {**item, "raw_prediction": raw, "predicted_plan": plan, **scored}
        predictions.append(payload)
        print(
            json.dumps(
                {
                    "index": index,
                    "id": item["id"],
                    "json_valid": scored["json_valid"],
                    "source_accuracy": scored["source_accuracy"],
                    "qa_type_accuracy": scored["qa_type_accuracy"],
                    "planner_score": round(scored["planner_score"], 4),
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


class PhiPlannerRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install torch, transformers, peft and bitsandbytes to evaluate Phi planner.") from exc
        ensure_phi_transformers_compat()

        self.torch = torch
        self.max_new_tokens = args.max_new_tokens
        self.prompt_max_length = args.prompt_max_length
        if args.local_files_only:
            validate_model_folder(Path(args.model_name_or_path))
        self.tokenizer = load_tokenizer(AutoTokenizer, args.model_name_or_path, local_files_only=args.local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

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

    def generate(self, user_prompt: str) -> str:
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
                truncation=True,
                max_length=self.prompt_max_length,
            )
        else:
            inputs = self.tokenizer(
                f"System: {SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:",
                return_tensors="pt",
                truncation=True,
                max_length=self.prompt_max_length,
            )
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
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_examples(questions_path: Path, answers_path: Path, available_sources: list[str]) -> list[dict[str, Any]]:
    questions = read_json_rows(questions_path)
    answers = {item["id"]: item for item in read_json_rows(answers_path)}
    examples = []
    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise RuntimeError(f"Missing reference answer for id={question['id']}")
        expected_plan = build_reference_plan(question, answer)
        examples.append(
            {
                "id": question["id"],
                "query": question["query"],
                "qa_type": question.get("qa_type", ""),
                "ticker": question.get("ticker", ""),
                "source_file": question["source_file"],
                "available_sources": available_sources,
                "expected_plan": expected_plan,
            }
        )
    return examples


def build_reference_plan(question: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    qa_type = str(question.get("qa_type") or answer.get("qa_type") or "single_hop")
    source_file = str(question["source_file"])
    query = str(question["query"])
    sub_queries = [{"id": "q1", "query": query, "source_file": source_file, "tool": "retriever"}]
    if qa_type == "multi_hop":
        sub_queries.append(
            {
                "id": "q2",
                "query": f"Kiểm tra phép tính hoặc quan hệ cần thiết để trả lời: {query}",
                "source_file": source_file,
                "tool": "calculator_or_reasoning",
                "depends_on": ["q1"],
            }
        )
    return {
        "strategy": "sequential",
        "qa_type": qa_type,
        "ticker": question.get("ticker", ""),
        "selected_sources": [source_file],
        "sub_queries": sub_queries,
        "executor_instruction": "Truy xuất đúng tài liệu nguồn, ưu tiên dòng/cột chứa chỉ tiêu trong câu hỏi.",
    }


def format_prompt(item: dict[str, Any]) -> str:
    return (
        f"Câu hỏi: {item['query']}\n"
        f"Mã cổ phiếu: {item['ticker']}\n"
        f"Loại câu hỏi: {item['qa_type']}\n"
        f"Nguồn gợi ý từ dữ liệu: {item['source_file']}\n"
        f"Các nguồn hiện có: {', '.join(item['available_sources'])}\n\n"
        "Hãy trả về JSON plan với các khóa: strategy, qa_type, ticker, selected_sources, "
        "sub_queries, executor_instruction."
    )


def score_plan(plan: dict[str, Any] | None, item: dict[str, Any]) -> dict[str, Any]:
    if not plan:
        return {
            "json_valid": False,
            "source_accuracy": 0.0,
            "qa_type_accuracy": 0.0,
            "ticker_accuracy": 0.0,
            "subquery_present": 0.0,
            "planner_score": 0.0,
        }

    selected = plan.get("selected_sources") or []
    if isinstance(selected, str):
        selected = [selected]
    selected = [str(value) for value in selected]
    source_accuracy = float(item["source_file"] in selected)
    qa_type_accuracy = float(str(plan.get("qa_type", "")).lower() == str(item["qa_type"]).lower())
    ticker_accuracy = float(str(plan.get("ticker", "")).upper() == str(item["ticker"]).upper())
    sub_queries = plan.get("sub_queries") or []
    subquery_present = float(isinstance(sub_queries, list) and len(sub_queries) > 0)
    planner_score = 0.45 * source_accuracy + 0.20 * qa_type_accuracy + 0.15 * ticker_accuracy + 0.20 * subquery_present
    return {
        "json_valid": True,
        "source_accuracy": source_accuracy,
        "qa_type_accuracy": qa_type_accuracy,
        "ticker_accuracy": ticker_accuracy,
        "subquery_present": subquery_present,
        "planner_score": planner_score,
    }


def extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        "json_valid": mean(float(row["json_valid"]) for row in rows),
        "source_accuracy": mean(row["source_accuracy"] for row in rows),
        "qa_type_accuracy": mean(row["qa_type_accuracy"] for row in rows),
        "ticker_accuracy": mean(row["ticker_accuracy"] for row in rows),
        "subquery_present": mean(row["subquery_present"] for row in rows),
        "planner_score": mean(row["planner_score"] for row in rows),
    }


def list_processed_sources(processed_dir: Path, *, max_sources: int) -> list[str]:
    if not processed_dir.exists():
        raise FileNotFoundError(processed_dir)
    sources = sorted(
        path.name
        for path in processed_dir.glob("**/*")
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() == ".txt"
    )
    return sources[:max_sources]


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


def validate_model_folder(model_path: Path) -> None:
    tokenizer_files = {"tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"}
    weight_patterns = ("*.safetensors", "*.bin", "*.pt")
    present = {path.name for path in model_path.glob("*") if path.is_file()}
    has_tokenizer = any(name in present for name in tokenizer_files)
    has_weights = any(any(model_path.glob(pattern)) for pattern in weight_patterns)
    if has_tokenizer and has_weights:
        return
    missing = []
    if not has_tokenizer:
        missing.append("tokenizer files")
    if not has_weights:
        missing.append("model weight files")
    raise RuntimeError(
        f"{model_path} is incomplete for local Phi loading; missing {', '.join(missing)}. "
        "Download the planner model again:\n"
        "huggingface-cli download microsoft/Phi-4-mini-instruct "
        "--local-dir models/phi --local-dir-use-symlinks False\n"
        "Also ensure tokenizer dependencies are installed: pip install sentencepiece tiktoken"
    )


def load_tokenizer(auto_tokenizer: Any, model_name_or_path: str, *, local_files_only: bool) -> Any:
    try:
        return auto_tokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except ValueError as exc:
        if "sentencepiece or tiktoken" not in str(exc):
            raise
        try:
            return auto_tokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
                use_fast=False,
            )
        except Exception as slow_exc:
            raise RuntimeError(
                "Cannot load the Phi tokenizer. Install tokenizer dependencies with "
                "`pip install sentencepiece tiktoken`, and verify models/phi contains tokenizer files, "
                "not only LICENSE/README."
            ) from slow_exc


def ensure_phi_transformers_compat() -> None:
    try:
        import transformers
        import transformers.utils as transformers_utils
        from typing import TypedDict
    except ImportError as exc:
        raise RuntimeError(
            "Phi planner requires transformers and typing_extensions. "
            "Run: pip install -U transformers typing_extensions"
        ) from exc

    if hasattr(transformers_utils, "LossKwargs"):
        return

    class LossKwargs(TypedDict, total=False):
        pass

    transformers_utils.LossKwargs = LossKwargs
    print(
        "[compat] transformers.utils.LossKwargs is missing; installed a local shim for Phi remote code. "
        f"Current transformers version: {getattr(transformers, '__version__', 'unknown')}. "
        "Prefer upgrading transformers if this keeps failing."
    )


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
