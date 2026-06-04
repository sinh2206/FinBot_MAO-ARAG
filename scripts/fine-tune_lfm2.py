from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SYSTEM_PROMPT = (
    "Bạn là executor RAG cho báo cáo tài chính chứng khoán Việt Nam. "
    "Chỉ trả lời dựa trên tài liệu được cung cấp. "
    "Nếu câu hỏi hỏi số liệu, hãy chọn đúng số ở cùng dòng/cột với chỉ tiêu trong câu hỏi và trả về số kèm đơn vị. "
    "Không chọn số ở bảng hoặc dòng khác. Nếu không đủ căn cứ, trả lời: KHÔNG TÌM THẤY."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune LiquidAI/LFM2-1.2B-RAG directly from data/processed_data, "
            "data/train and data/test without creating intermediate dataset files."
        )
    )
    parser.add_argument("--model_name_or_path", default="models/lfm2_rag")
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--train_questions", default="data/train/questions.json")
    parser.add_argument("--train_answers", default="data/train/reference_answers.json")
    parser.add_argument("--eval_questions", default="data/test/questions.json")
    parser.add_argument("--eval_answers", default="data/test/reference_answers.json")
    parser.add_argument("--output_dir", default="models/lfm2_rag_lora")
    parser.add_argument("--max_context_chars", type=int, default=1200)
    parser.add_argument("--context_window_chars", type=int, default=400)
    parser.add_argument(
        "--context_mode",
        default="evidence_window",
        choices=["evidence", "evidence_window", "full_window"],
        help="Use shorter evidence-focused contexts so the executor learns the final answer instead of copying tables.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_strategy", default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_strategy", default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--target_modules",
        default="all-linear",
        help='LoRA target modules. Use "all-linear" or comma-separated module names.',
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require_cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail early when CUDA is unavailable. By default, CPU smoke tests are allowed.",
    )
    parser.add_argument(
        "--allow_cpu_train",
        action="store_true",
        help="Allow full CPU training when CUDA is unavailable. This is usually very slow.",
    )
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment_multi_hop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--multi_hop_augment_copies",
        type=int,
        default=4,
        help="Number of query-variant copies to add for each multi_hop training row.",
    )
    parser.add_argument(
        "--multi_hop_formula_copies",
        type=int,
        default=3,
        help="Number of explicit arithmetic/formula examples to add for each multi_hop training row.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print dataset stats without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)

    train_examples = build_examples(
        questions_path=Path(args.train_questions),
        answers_path=Path(args.train_answers),
        processed_dir=processed_dir,
        max_context_chars=args.max_context_chars,
        context_window_chars=args.context_window_chars,
        context_mode=args.context_mode,
    )
    eval_examples = build_examples(
        questions_path=Path(args.eval_questions),
        answers_path=Path(args.eval_answers),
        processed_dir=processed_dir,
        max_context_chars=args.max_context_chars,
        context_window_chars=args.context_window_chars,
        context_mode=args.context_mode,
    )
    if args.augment_multi_hop:
        train_examples = augment_multi_hop_examples(
            train_examples,
            copies=args.multi_hop_augment_copies,
            formula_copies=args.multi_hop_formula_copies,
        )

    if args.dry_run:
        print_summary(train_examples, eval_examples, args)
        return

    if not train_examples:
        raise RuntimeError("No training examples were built.")
    if not Path(args.model_name_or_path).exists():
        raise FileNotFoundError(f"Model folder not found: {args.model_name_or_path}")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "LFM2 SFT requires: pip install transformers trl peft datasets accelerate bitsandbytes torch"
        ) from exc

    configure_torch_runtime(torch, tf32=args.tf32)
    cuda_available = torch.cuda.is_available()
    print_device_summary(torch)
    validate_device_args(args, torch, cuda_available=cuda_available)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = [format_prompt_completion(tokenizer, item) for item in train_examples]
    eval_rows = [format_prompt_completion(tokenizer, item) for item in eval_examples]
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if cuda_available:
        model_kwargs["device_map"] = "auto"

    bf16_enabled = args.bf16 and cuda_available and torch.cuda.is_bf16_supported()
    fp16_enabled = args.fp16 and cuda_available and not bf16_enabled

    if bf16_enabled:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif fp16_enabled:
        model_kwargs["torch_dtype"] = torch.float16

    if args.use_4bit and cuda_available:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.use_4bit and cuda_available:
        model = prepare_model_for_kbit_training(model)

    peft_config = None
    if args.use_lora:
        target_modules: str | list[str]
        if args.target_modules == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    sft_config = SFTConfig(
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
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to="none",
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        tf32=args.tf32 and cuda_available,
        gradient_checkpointing=args.gradient_checkpointing,
        use_cache=not args.gradient_checkpointing,
        seed=args.seed,
        packing=False,
        completion_only_loss=True,
        use_cpu=not cuda_available,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned LFM2 adapter/model to {args.output_dir}")


def build_examples(
    *,
    questions_path: Path,
    answers_path: Path,
    processed_dir: Path,
    max_context_chars: int,
    context_window_chars: int,
    context_mode: str,
) -> list[dict[str, Any]]:
    questions = read_jsonl(questions_path)
    answers = {item["id"]: item for item in read_jsonl(answers_path)}
    text_cache: dict[Path, str] = {}
    examples: list[dict[str, Any]] = []

    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise RuntimeError(f"Missing reference answer for id={question['id']}")
        if question.get("qa_type") != answer.get("qa_type"):
            raise RuntimeError(f"qa_type mismatch for id={question['id']}")

        source_path = resolve_source_path(processed_dir, question["source_file"])
        if source_path not in text_cache:
            text_cache[source_path] = source_path.read_text(encoding="utf-8", errors="ignore")

        document_text = text_cache[source_path]
        evidence = answer.get("ground_truth_context", "")
        context = select_context(
            document_text=document_text,
            evidence=evidence,
            max_context_chars=max_context_chars,
            context_window_chars=context_window_chars,
            context_mode=context_mode,
        )
        examples.append(
            {
                "id": question["id"],
                "query": question["query"],
                "answer": answer["ground_truth_answer"],
                "context": context,
                "evidence": evidence,
                "qa_type": question.get("qa_type", ""),
                "ticker": question.get("ticker", ""),
                "source_file": question["source_file"],
            }
        )
    return examples


def augment_multi_hop_examples(examples: list[dict[str, Any]], *, copies: int, formula_copies: int) -> list[dict[str, Any]]:
    if copies <= 0 and formula_copies <= 0:
        return examples
    augmented = list(examples)
    for item in examples:
        if item.get("qa_type") != "multi_hop":
            continue
        for index, query in enumerate(build_multi_hop_query_variants(item)[:copies], start=1):
            clone = dict(item)
            clone["id"] = f"{item['id']}__aug_mh_{index}"
            clone["query"] = query
            clone["augmented_from"] = item["id"]
            clone["augmentation_type"] = classify_multi_hop_operation(item)
            augmented.append(clone)
        for index, clone in enumerate(build_multi_hop_formula_examples(item)[:formula_copies], start=1):
            clone["id"] = f"{item['id']}__aug_formula_{index}"
            clone["augmented_from"] = item["id"]
            augmented.append(clone)
    return augmented


def build_multi_hop_query_variants(item: dict[str, Any]) -> list[str]:
    query = str(item["query"]).strip()
    evidence = str(item.get("evidence") or "").strip()
    answer = str(item.get("answer") or "").strip()
    operation = classify_multi_hop_operation(item)

    variants = [
        f"{query} Hãy tính từ các số thành phần trong ngữ cảnh và chỉ trả lời đáp án cuối.",
        f"Dựa vào ngữ cảnh, tính {operation} cho câu hỏi: {query}",
        f"Từ các số liệu sau: {evidence} Trả lời ngắn gọn: {query}",
        f"Hãy suy ra đáp án cuối cho {item.get('ticker', '')}: {query}",
    ]
    if "%" in answer or "tỷ lệ" in query.lower() or "phần trăm" in query.lower():
        variants.append(f"Tính tỷ lệ phần trăm từ các số trong ngữ cảnh và trả lời: {query}")
    if "eps" in query.lower() or "cổ phiếu" in query.lower():
        variants.append(f"Tính chênh lệch EPS/lãi cơ bản trên cổ phiếu và trả lời: {query}")
    if "tăng" in query.lower() or "giảm" in query.lower() or "so với" in query.lower():
        variants.append(f"Tính mức tăng/giảm so với kỳ gốc và trả lời đúng đơn vị: {query}")
    return unique_preserve_order(variants)


def classify_multi_hop_operation(item: dict[str, Any]) -> str:
    text = f"{item.get('query', '')} {item.get('answer', '')}".lower()
    if "%" in text or "phần trăm" in text or "tỷ lệ" in text or "chiếm" in text:
        return "tỷ lệ phần trăm"
    if "eps" in text or "cổ phiếu" in text:
        return "EPS chênh lệch"
    if "tăng" in text or "giảm" in text or "so với" in text:
        return "hiệu số tăng/giảm"
    return "hiệu số"


def build_multi_hop_formula_examples(item: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = str(item.get("evidence") or "").strip()
    numbers = extract_numbers_from_text(evidence)
    if len(numbers) < 2:
        return []

    answer = str(item.get("answer") or "").strip()
    operation = classify_multi_hop_operation(item)
    formulas = build_formula_descriptions(operation, numbers[:2], answer)
    examples = []
    for formula in formulas:
        clone = dict(item)
        clone["context"] = (
            f"{evidence}\n\n"
            f"Yêu cầu tính toán: {formula}\n"
            "Quy tắc: chỉ trả lời kết quả cuối cùng, giữ đúng đơn vị trong đáp án mẫu."
        )
        clone["query"] = f"{formula}. Câu hỏi gốc: {item['query']}"
        clone["augmentation_type"] = f"formula_{operation}"
        examples.append(clone)
    return examples


def build_formula_descriptions(operation: str, numbers: list[str], answer: str) -> list[str]:
    a, b = numbers[0], numbers[1]
    if operation == "tỷ lệ phần trăm":
        return [
            f"Tính A / B * 100 với A = {a}, B = {b}; kết quả đúng là {answer}",
            f"Tính tỷ lệ phần trăm giữa {a} và {b}; trả lời {answer}",
        ]
    if operation == "EPS chênh lệch":
        return [
            f"Tính EPS chênh lệch A - B với A = {a}, B = {b}; kết quả đúng là {answer}",
            f"Tính lãi cơ bản trên cổ phiếu tăng/giảm từ {b} lên {a}; trả lời {answer}",
        ]
    if "nhân viên" in answer.lower():
        return [
            f"Tính số nhân viên giảm/tăng A - B với A = {a}, B = {b}; kết quả đúng là {answer}",
            f"Tính chênh lệch số lượng nhân viên giữa {a} và {b}; trả lời {answer}",
        ]
    return [
        f"Tính A - B với A = {a}, B = {b}; kết quả đúng là {answer}",
        f"Tính mức tăng/giảm so với cùng kỳ từ {b} lên {a}; trả lời {answer}",
    ]


def extract_numbers_from_text(text: str) -> list[str]:
    values = []
    for match in re.finditer(r"\(?-?\d[\d., ]*\)?%?", text):
        raw = re.sub(r"(?<=\d)[ ]+(?=[.,]?\d)", "", match.group(0)).strip(" .;:")
        if not raw or raw in {"2025", "2024"}:
            continue
        if raw.isdigit() and 1 <= int(raw) <= 31:
            continue
        values.append(raw)
    return values


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def format_chat_text(tokenizer: Any, item: dict[str, Any]) -> str:
    prompt = format_prompt_text(tokenizer, item)
    completion = item["answer"]
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item)},
            {"role": "assistant", "content": completion},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{prompt}{completion}{tokenizer.eos_token or ''}"


def format_prompt_completion(tokenizer: Any, item: dict[str, Any]) -> dict[str, str]:
    completion = item["answer"]
    if tokenizer.eos_token:
        completion = f"{completion}{tokenizer.eos_token}"
    return {"prompt": format_prompt_text(tokenizer, item), "completion": completion}


def format_prompt_text(tokenizer: Any, item: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {build_user_prompt(item)}\n\nAssistant:"


def build_user_prompt(item: dict[str, Any]) -> str:
    user_prompt = (
        f"Tài liệu nguồn: {item['source_file']}\n"
        f"Mã cổ phiếu: {item['ticker']}\n"
        f"Loại câu hỏi: {item['qa_type']}\n\n"
        f"<context>\n{item['context']}\n</context>\n\n"
        f"Câu hỏi: {item['query']}\n"
        "Chỉ trả lời đáp án cuối cùng, ngắn gọn. "
        "Nếu đáp án là số tiền, chỉ trả về số tiền đúng và đơn vị VND. Không chép lại câu hỏi hoặc bảng."
    )
    return user_prompt


def print_summary(train_examples: list[dict[str, Any]], eval_examples: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "model_name_or_path": args.model_name_or_path,
        "processed_dir": args.processed_dir,
        "output_dir": args.output_dir,
        "context_mode": args.context_mode,
        "max_context_chars": args.max_context_chars,
        "context_window_chars": args.context_window_chars,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "augmented_train_examples": sum(1 for item in train_examples if item.get("augmented_from")),
        "train_qa_type": dict(Counter(item["qa_type"] for item in train_examples)),
        "eval_qa_type": dict(Counter(item["qa_type"] for item in eval_examples)),
        "train_ticker": dict(Counter(item["ticker"] for item in train_examples)),
        "eval_ticker": dict(Counter(item["ticker"] for item in eval_examples)),
        "sample": {
            "id": train_examples[0]["id"] if train_examples else None,
            "source_file": train_examples[0]["source_file"] if train_examples else None,
            "context_chars": len(train_examples[0]["context"]) if train_examples else 0,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def configure_torch_runtime(torch: Any, *, tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    try:
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    except Exception:
        pass


def validate_device_args(args: argparse.Namespace, torch: Any, *, cuda_available: bool) -> None:
    if cuda_available:
        return

    torch_cuda = getattr(torch.version, "cuda", None)
    message = (
        "CUDA is not available to PyTorch, so GPU fine-tuning cannot start.\n"
        f"Detected torch={torch.__version__}, torch_cuda={torch_cuda}.\n"
        "This usually means the installed PyTorch CUDA build does not match the server driver/runtime,\n"
        "or the torch / torchvision / torchaudio packages were installed from mixed CUDA versions.\n\n"
        "To train on GPU, fix one of these first:\n"
        "1. Install one consistent torch / torchvision / torchaudio build for the server CUDA runtime.\n"
        "2. Or update the NVIDIA driver if it is genuinely older than the CUDA build used by PyTorch.\n\n"
        "For a fast validation without training, rerun with:\n"
        "python scripts/train_lfm2_sft.py --dry_run\n"
    )
    if args.require_cuda:
        raise RuntimeError(message)

    if not args.allow_cpu_train:
        raise RuntimeError(
            message
            + "\nCPU training is blocked by default because loading and fine-tuning LFM2-1.2B on CPU is very slow.\n"
            "Add --allow_cpu_train only if you intentionally want CPU training.\n"
        )

    if args.use_4bit:
        print(
            json.dumps(
                {
                    "warning": "CUDA is unavailable, so --use_4bit is disabled automatically for CPU mode.",
                    "torch": torch.__version__,
                    "torch_cuda": torch_cuda,
                },
                ensure_ascii=False,
            )
        )
        args.use_4bit = False
    if args.bf16 or args.fp16:
        print(
            json.dumps(
                {
                    "warning": "CUDA is unavailable, so bf16/fp16 training flags are disabled for CPU mode.",
                    "bf16": args.bf16,
                    "fp16": args.fp16,
                },
                ensure_ascii=False,
            )
        )
        args.bf16 = False
        args.fp16 = False


def print_device_summary(torch: Any) -> None:
    cuda_available = torch.cuda.is_available()
    payload: dict[str, Any] = {
        "cuda_available": cuda_available,
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
    }
    if cuda_available:
        payload["device_count"] = torch.cuda.device_count()
        payload["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
        payload["bf16_supported"] = torch.cuda.is_bf16_supported()
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
