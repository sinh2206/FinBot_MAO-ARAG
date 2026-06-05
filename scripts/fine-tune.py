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
    "Bạn là Qwen executor cho hệ thống RAG báo cáo tài chính chứng khoán Việt Nam. "
    "Chỉ trả lời dựa trên context được cung cấp. "
    "Nếu câu hỏi hỏi số liệu, hãy chọn đúng số ở cùng dòng/cột với chỉ tiêu trong câu hỏi và trả về số kèm đơn vị. "
    "Nếu không đủ căn cứ, trả lời: KHÔNG TÌM THẤY."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B-Instruct executor with LoRA/SFT.")
    parser.add_argument("--model_name_or_path", default="models/qwen")
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--train_questions", default="data/train/questions.json")
    parser.add_argument("--train_answers", default="data/train/reference_answers.json")
    parser.add_argument("--eval_questions", default="data/test/questions.json")
    parser.add_argument("--eval_answers", default="data/test/reference_answers.json")
    parser.add_argument("--output_dir", default="models/qwen_executor_lora")
    parser.add_argument("--max_context_chars", type=int, default=2200)
    parser.add_argument("--context_window_chars", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=3072)
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
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_examples = build_examples(
        questions_path=Path(args.train_questions),
        answers_path=Path(args.train_answers),
        processed_dir=Path(args.processed_dir),
        max_context_chars=args.max_context_chars,
        context_window_chars=args.context_window_chars,
    )
    eval_examples = build_examples(
        questions_path=Path(args.eval_questions),
        answers_path=Path(args.eval_answers),
        processed_dir=Path(args.processed_dir),
        max_context_chars=args.max_context_chars,
        context_window_chars=args.context_window_chars,
    )

    if args.dry_run:
        print_summary(train_examples, eval_examples, args)
        return
    if not Path(args.model_name_or_path).exists():
        raise FileNotFoundError(f"Model folder not found: {args.model_name_or_path}")
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
            "Fine-tune requires: pip install torch transformers trl peft datasets accelerate bitsandbytes"
        ) from exc

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
    eval_dataset = Dataset.from_list([{"text": format_example(tokenizer, item)} for item in eval_examples])

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
    print(f"Saved Qwen executor adapter/model to {args.output_dir}")


def build_examples(
    *,
    questions_path: Path,
    answers_path: Path,
    processed_dir: Path,
    max_context_chars: int,
    context_window_chars: int,
) -> list[dict[str, Any]]:
    questions = read_json_rows(questions_path)
    answers = {item["id"]: item for item in read_json_rows(answers_path)}
    text_cache: dict[Path, str] = {}
    examples = []
    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise RuntimeError(f"Missing reference answer for id={question['id']}")
        source_path = resolve_source_path(processed_dir, question["source_file"])
        if source_path not in text_cache:
            text_cache[source_path] = source_path.read_text(encoding="utf-8", errors="ignore")
        evidence = str(answer.get("ground_truth_context") or "")
        context = select_context(
            document_text=text_cache[source_path],
            evidence=evidence,
            max_context_chars=max_context_chars,
            context_window_chars=context_window_chars,
        )
        examples.append(
            {
                "id": question["id"],
                "query": question["query"],
                "qa_type": question.get("qa_type", ""),
                "ticker": question.get("ticker", ""),
                "source_file": question["source_file"],
                "context": context,
                "answer": answer.get("ground_truth_answer", ""),
                "evidence": evidence,
            }
        )
    return examples


def format_example(tokenizer: Any, item: dict[str, Any]) -> str:
    user_prompt = (
        f"Tài liệu nguồn: {item['source_file']}\n"
        f"Mã cổ phiếu: {item['ticker']}\n"
        f"Loại câu hỏi: {item['qa_type']}\n\n"
        f"<context>\n{item['context']}\n</context>\n\n"
        f"Câu hỏi: {item['query']}\n"
        "Chỉ trả lời đáp án cuối cùng, ngắn gọn. Không chép lại bảng."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(item["answer"])},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant: {item['answer']}"


def select_context(*, document_text: str, evidence: str, max_context_chars: int, context_window_chars: int) -> str:
    if evidence and evidence in document_text:
        start = max(document_text.find(evidence) - context_window_chars // 2, 0)
        return document_text[start : start + max_context_chars].strip()
    if evidence:
        return (evidence.strip() + "\n\n" + document_text[:max_context_chars]).strip()[:max_context_chars]
    return document_text[:max_context_chars].strip()


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
    return {**item, "context": re.sub(r"\s+", " ", item["context"])[:240]}


def filter_supported_kwargs(callable_obj: Any, **kwargs: Any) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


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


if __name__ == "__main__":
    main()
