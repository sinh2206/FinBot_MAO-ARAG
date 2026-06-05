#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
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

HIGH_VRAM_PRESET = "20GiB"


SYSTEM_PROMPT = (
    "Bạn là Microsoft Phi-4-mini-instruct, đóng vai trò agent planner/coordinator cho hệ thống RAG "
    "báo cáo tài chính chứng khoán Việt Nam. Nhiệm vụ của bạn là lập kế hoạch truy xuất cho executor. "
    "Luôn trả về JSON hợp lệ, không thêm giải thích ngoài JSON."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Microsoft Phi-4-mini-instruct as the planner.")
    parser.add_argument(
        "--model_name_or_path",
        default=load_dotenv_value("PHI_MODEL_NAME", load_dotenv_value("PLANNER_MODEL_NAME", "models/phi")),
    )
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--train_questions", default="data/train/questions.json")
    parser.add_argument("--train_answers", default="data/train/reference_answers.json")
    parser.add_argument("--eval_questions", default="data/test/questions.json")
    parser.add_argument("--eval_answers", default="data/test/reference_answers.json")
    parser.add_argument("--output_dir", default="models/phi_planner_lora")
    parser.add_argument("--predictions_file", default="data/evaluation/phi_planner_predictions.jsonl")
    parser.add_argument("--metrics_file", default="data/evaluation/phi_planner_metrics.json")
    parser.add_argument("--max_sources", type=int, default=80)
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--prompt_max_length", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_strategy", default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--target_modules",
        default="all-linear",
        help='Use "all-linear" for the strong 20GiB preset, or pass specific modules if you need a lighter run.',
    )
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch_empty_cache_steps", type=int, default=1)
    parser.add_argument(
        "--prepare_kbit_mode",
        default="minimal",
        choices=["minimal", "peft", "none"],
        help=(
            "minimal freezes the base model and enables input grads without PEFT's fp32 cast. "
            "peft calls prepare_model_for_kbit_training and needs more VRAM."
        ),
    )
    parser.add_argument(
        "--gpu_memory_limit",
        default=HIGH_VRAM_PRESET,
        help='Optional per-GPU load limit for transformers device_map. Default uses the 20GiB preset.',
    )
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_cpu_train", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Print dataset summary and exit.")
    parser.add_argument("--dry_run", action="store_true", help="Alias for --prepare_only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True

    available_sources = list_processed_sources(Path(args.processed_dir), max_sources=args.max_sources)
    train_examples = build_examples(
        questions_path=Path(args.train_questions),
        answers_path=Path(args.train_answers),
        available_sources=available_sources,
    )
    eval_examples = build_examples(
        questions_path=Path(args.eval_questions),
        answers_path=Path(args.eval_answers),
        available_sources=available_sources,
    )

    if args.prepare_only or args.dry_run:
        print_summary(train_examples, eval_examples, args)
        return

    trained_output_dir = Path(args.output_dir)
    if args.train:
        train_phi_planner(args, train_examples, eval_examples if args.evaluate else None)

    if args.evaluate:
        model_path, adapter_path = resolve_inference_artifacts(args.model_name_or_path, trained_output_dir)
        runner = PhiPlannerRunner(
            model_name_or_path=model_path,
            adapter_path=adapter_path,
            max_new_tokens=args.max_new_tokens,
            prompt_max_length=args.prompt_max_length,
            load_in_4bit=args.use_4bit,
            local_files_only=args.local_files_only,
            device_map="auto",
            gpu_memory_limit=args.gpu_memory_limit,
        )
        predictions = evaluate_planner(
            runner,
            eval_examples,
            predictions_file=Path(args.predictions_file),
            metrics_file=Path(args.metrics_file),
        )
        print(json.dumps(summarize(predictions), ensure_ascii=False, indent=2))
        print(f"Wrote predictions to {args.predictions_file}")
        print(f"Wrote metrics to {args.metrics_file}")


def train_phi_planner(
    args: argparse.Namespace,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]] | None,
) -> None:
    if not train_examples:
        raise RuntimeError("No training examples were built.")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning Phi requires: pip install torch transformers trl peft datasets accelerate bitsandbytes"
        ) from exc

    ensure_phi_transformers_compat()
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu_train:
        raise RuntimeError("CUDA is unavailable. Use --allow_cpu_train only for a tiny smoke test.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_list([{"text": format_example(tokenizer, item)} for item in train_examples])
    eval_dataset = Dataset.from_list([{"text": format_example(tokenizer, item)} for item in eval_examples]) if eval_examples else None

    bf16_enabled = args.bf16 and cuda_available and torch.cuda.is_bf16_supported()
    fp16_enabled = args.fp16 and cuda_available and not bf16_enabled
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "local_files_only": args.local_files_only}
    if cuda_available:
        model_kwargs["device_map"] = "auto"
        if args.gpu_memory_limit:
            model_kwargs["max_memory"] = {0: args.gpu_memory_limit}
    if bf16_enabled:
        model_kwargs["dtype"] = torch.bfloat16
    elif fp16_enabled:
        model_kwargs["dtype"] = torch.float16
    if args.use_4bit and cuda_available:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.use_4bit and cuda_available:
        if args.prepare_kbit_mode == "peft":
            torch.cuda.empty_cache()
            try:
                model = prepare_model_for_kbit_training(model)
            except torch.OutOfMemoryError as exc:
                raise RuntimeError(
                    "CUDA OOM while running PEFT prepare_model_for_kbit_training. "
                    "Re-run with --prepare_kbit_mode minimal, reduce --max_seq_length, "
                    "or free the other GPU process shown by nvidia-smi."
                ) from exc
        elif args.prepare_kbit_mode == "minimal":
            model = prepare_kbit_model_minimal(
                model,
                gradient_checkpointing=args.gradient_checkpointing,
                use_reentrant=args.gradient_checkpointing_use_reentrant,
            )

    peft_config = None
    if args.use_lora:
        target_modules: str | list[str]
        target_modules = (
            "all-linear"
            if args.target_modules == "all-linear"
            else [item.strip() for item in args.target_modules.split(",") if item.strip()]
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    sft_kwargs = filter_supported_kwargs(
        SFTConfig,
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to="none",
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant},
        torch_empty_cache_steps=args.torch_empty_cache_steps,
        use_cache=not args.gradient_checkpointing,
        seed=args.seed,
        packing=False,
        use_cpu=not cuda_available,
    )
    sft_config = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved Phi planner adapter/model to {args.output_dir}")


def evaluate_planner(
    runner: "PhiPlannerRunner",
    rows: list[dict[str, Any]],
    *,
    predictions_file: Path,
    metrics_file: Path,
) -> list[dict[str, Any]]:
    predictions = []
    for index, item in enumerate(rows, start=1):
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
    write_jsonl(predictions_file, predictions)
    write_json(metrics_file, metrics)
    return predictions


class PhiPlannerRunner:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_path: Path | None,
        max_new_tokens: int,
        prompt_max_length: int,
        load_in_4bit: bool,
        local_files_only: bool,
        device_map: str,
        gpu_memory_limit: str | None,
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install torch, transformers, peft and bitsandbytes to evaluate Phi planner.") from exc

        ensure_phi_transformers_compat()
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.prompt_max_length = prompt_max_length
        if local_files_only:
            validate_model_folder(Path(model_name_or_path))
        self.tokenizer = load_tokenizer(AutoTokenizer, model_name_or_path, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "device_map": device_map,
        }
        if gpu_memory_limit and device_map == "auto":
            model_kwargs["max_memory"] = {0: gpu_memory_limit}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if adapter_path and adapter_path.exists():
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
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


def build_examples(*, questions_path: Path, answers_path: Path, available_sources: list[str]) -> list[dict[str, Any]]:
    questions = read_json_rows(questions_path)
    answers = {item["id"]: item for item in read_json_rows(answers_path)}
    examples = []
    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise RuntimeError(f"Missing reference answer for id={question['id']}")
        reference_plan = build_reference_plan(question, answer)
        examples.append(
            {
                "id": str(question["id"]),
                "query": str(question["query"]),
                "qa_type": str(question.get("qa_type") or answer.get("qa_type") or ""),
                "ticker": str(question.get("ticker") or answer.get("ticker") or ""),
                "source_file": str(question.get("source_file") or answer.get("source_file") or ""),
                "available_sources": available_sources,
                "reference_plan": reference_plan,
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


def format_example(tokenizer: Any, item: dict[str, Any]) -> str:
    user_prompt = format_prompt(item)
    assistant_response = json.dumps(item["reference_plan"], ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant: {assistant_response}"


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


def print_summary(train_examples: list[dict[str, Any]], eval_examples: list[dict[str, Any]], args: argparse.Namespace) -> None:
    qa_counts: dict[str, int] = {}
    for item in train_examples:
        qa_counts[item["qa_type"]] = qa_counts.get(item["qa_type"], 0) + 1
    print(
        json.dumps(
            {
                "model_name_or_path": args.model_name_or_path,
                "output_dir": args.output_dir,
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "train_qa_type_counts": qa_counts,
                "first_train_example": scrub_example(train_examples[0]) if train_examples else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def scrub_example(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "reference_plan": item["reference_plan"],
    }


def resolve_inference_artifacts(model_name_or_path: str, output_dir: Path) -> tuple[str, Path | None]:
    if output_dir.exists():
        adapter_config = output_dir / "adapter_config.json"
        if adapter_config.exists():
            return model_name_or_path, output_dir
        return str(output_dir), None
    return model_name_or_path, None


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
        f"Current transformers version: {getattr(transformers, '__version__', 'unknown')}."
    )


def prepare_kbit_model_minimal(model: Any, *, gradient_checkpointing: bool, use_reentrant: bool) -> Any:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if not gradient_checkpointing:
        return model

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
        except TypeError:
            model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return model

    input_embeddings = model.get_input_embeddings()

    def make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> None:
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(make_inputs_require_grad)
    return model


def filter_supported_kwargs(callable_obj: Any, **kwargs: Any) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


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
