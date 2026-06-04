from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PLANNER_SYSTEM_PROMPT = """Bạn là planner_agent cho hệ thống RAG báo cáo tài chính chứng khoán Việt Nam.
Nhiệm vụ của bạn là điều phối retriever và executor LFM2-1.2B-RAG đã fine-tune bằng LoRA.

Chỉ trả về một JSON hợp lệ, không markdown, không giải thích.
Schema bắt buộc:
{
  "strategy": "sequential" | "parallel",
  "requires_retrieval": true,
  "requires_execution": true,
  "aggregation_mode": "concat" | "synthesize",
  "ticker": "MÃ_CỔ_PHIẾU",
  "qa_type": "single_hop" | "multi_hop",
  "selected_sources": ["ten_file_txt"],
  "sub_queries": [
    {
      "id": "q1",
      "query": "câu hỏi con cụ thể",
      "type": "retrieval_qa" | "calculation_qa",
      "depends_on": [],
      "tool": "retriever"
    }
  ]
}

Quy tắc:
- single_hop: thường dùng 1 sub_query, aggregation_mode="concat".
- multi_hop: tách các số liệu cần truy xuất hoặc phép tính thành nhiều sub_queries, aggregation_mode="synthesize".
- Luôn chọn đúng ticker và selected_sources nếu câu hỏi có mã cổ phiếu.
- Không tự trả lời câu hỏi tài chính trong planner; planner chỉ tạo workflow JSON.
"""


@dataclass(slots=True)
class PlannerExample:
    id: str
    query: str
    qa_type: str
    ticker: str
    source_file: str
    answer: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU PPO-style training for Qwen planner using data/train and data/test. "
            "The planner learns to route questions to the fine-tuned LFM2-1.2B-RAG executor."
        )
    )
    parser.add_argument("--model_path", default="models/qwen", help="Local Qwen/Qwen2.5-7B-Instruct folder.")
    parser.add_argument("--lfm2_adapter_path", default="models/lfm2_rag_lora")
    parser.add_argument("--processed_dir", default="data/processed_data")
    parser.add_argument("--train_questions", default="data/train/questions.json")
    parser.add_argument("--train_answers", default="data/train/reference_answers.json")
    parser.add_argument("--eval_questions", default="data/test/questions.json")
    parser.add_argument("--eval_answers", default="data/test/reference_answers.json")
    parser.add_argument("--output_dir", default="models/qwen_ppo")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1, help="Limit optimizer steps. -1 means all epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--reward_baseline_momentum", type=float, default=0.9)
    parser.add_argument("--sft_coef", type=float, default=0.02, help="Small supervised JSON-planner loss mixed into PPO.")
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--eval_max_samples", type=int, default=24)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help='Comma-separated LoRA modules. Use "all-linear" if needed.',
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--min_free_vram_gb",
        type=float,
        default=8.0,
        help="Fail early if the selected GPU has less free VRAM before loading Qwen. Use 0 to disable.",
    )
    parser.add_argument(
        "--kbit_prepare_mode",
        default="light",
        choices=["light", "peft"],
        help=(
            "light freezes the base model and enables input grads without fp32 casting. "
            "peft calls prepare_model_for_kbit_training and may need more VRAM."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional transformers attention backend. sdpa is usually memory-friendly on modern PyTorch.",
    )
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_lfm2_adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Validate data/model paths and print stats only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    processed_dir = Path(args.processed_dir)
    train_examples = build_examples(Path(args.train_questions), Path(args.train_answers))
    eval_examples = build_examples(Path(args.eval_questions), Path(args.eval_answers))
    available_sources = list_processed_sources(processed_dir)

    validate_paths(args, available_sources)
    if args.dry_run:
        print_summary(args, train_examples, eval_examples, available_sources)
        return

    try:
        import torch
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Qwen PPO training requires: pip install transformers peft accelerate bitsandbytes torch"
        ) from exc

    configure_torch_runtime(torch, tf32=args.tf32)
    print_device_summary(torch)
    validate_cuda(args, torch)
    validate_free_vram(args, torch)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    bf16_enabled = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_enabled = args.fp16 and torch.cuda.is_available() and not bf16_enabled
    dtype = torch.bfloat16 if bf16_enabled else torch.float16 if fp16_enabled else None

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "device_map": "auto",
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    if args.use_4bit:
        if args.kbit_prepare_mode == "peft":
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            prepare_model_for_low_memory_kbit_training(model, gradient_checkpointing=args.gradient_checkpointing)
    if args.use_lora:
        model = get_peft_model(model, build_lora_config(args))
        model.print_trainable_parameters()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    reward_baseline = 0.0
    stop_training = False
    for epoch in range(1, args.num_epochs + 1):
        random.shuffle(train_examples)
        for example in train_examples:
            step += 1
            train_stats = ppo_update(
                args=args,
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                example=example,
                available_sources=available_sources,
                reward_baseline=reward_baseline,
            )
            reward_baseline = update_baseline(
                reward_baseline,
                train_stats["reward"],
                momentum=args.reward_baseline_momentum,
                initialized=step > 1,
            )
            payload = {
                "step": step,
                "epoch": epoch,
                "id": example.id,
                "reward": round(train_stats["reward"], 4),
                "baseline": round(reward_baseline, 4),
                "advantage": round(train_stats["advantage"], 4),
                "ppo_loss": round(train_stats["ppo_loss"], 4),
                "sft_loss": round(train_stats["sft_loss"], 4),
                "json_valid": train_stats["json_valid"],
            }
            print(json.dumps(payload, ensure_ascii=False))

            if args.eval_every > 0 and step % args.eval_every == 0:
                evaluate(args, torch, model, tokenizer, eval_examples, available_sources, prefix=f"eval_step_{step}")
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(model, tokenizer, output_dir / f"checkpoint-{step}")
            if args.max_steps > 0 and step >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            break

    evaluate(args, torch, model, tokenizer, eval_examples, available_sources, prefix="final_eval")
    save_checkpoint(model, tokenizer, output_dir)
    write_training_metadata(args, output_dir, train_examples, eval_examples, available_sources)
    print(f"Saved PPO-tuned Qwen planner to {output_dir}")


def ppo_update(
    *,
    args: argparse.Namespace,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    example: PlannerExample,
    available_sources: list[str],
    reward_baseline: float,
) -> dict[str, Any]:
    model.eval()
    prompt = format_planner_prompt(tokenizer, example, available_sources, args.lfm2_adapter_path)
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    )
    device = get_model_device(model)
    prompt_inputs = move_batch_to_device(prompt_inputs, device)
    prompt_len = prompt_inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated = model.generate(
            **prompt_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response_ids = generated[:, prompt_len:]
        old_logprob = sequence_mean_logprob(model, generated, prompt_len).detach()

    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
    reward, details = score_planner_response(response_text, example)
    advantage_value = reward - reward_baseline
    advantage = torch.tensor(advantage_value, device=device, dtype=old_logprob.dtype)

    model.train()
    new_logprob = sequence_mean_logprob(model, generated.detach(), prompt_len)
    ratio = torch.exp(torch.clamp(new_logprob - old_logprob, min=-10.0, max=10.0))
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantage
    ppo_loss = -torch.minimum(unclipped, clipped).mean()

    sft_loss = torch.zeros((), device=device, dtype=ppo_loss.dtype)
    if args.sft_coef > 0:
        sft_loss = planner_sft_loss(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_json=build_target_plan(example),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
        )

    loss = ppo_loss + args.sft_coef * sft_loss
    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(f"Invalid PPO loss at id={example.id}: {float(loss.detach().cpu())}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=1.0)
    optimizer.step()

    return {
        "reward": float(reward),
        "advantage": float(advantage_value),
        "ppo_loss": float(ppo_loss.detach().cpu()),
        "sft_loss": float(sft_loss.detach().cpu()),
        "json_valid": bool(details.get("json_valid")),
        "response": response_text,
    }


def sequence_mean_logprob(model: Any, full_ids: Any, prompt_len: int) -> Any:
    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :]
    labels = full_ids[:, 1:]
    token_logprobs = logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    response_logprobs = token_logprobs[:, max(prompt_len - 1, 0) :]
    if response_logprobs.numel() == 0:
        return token_logprobs.new_tensor(-20.0)
    return response_logprobs.mean()


def planner_sft_loss(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_json: str,
    max_prompt_length: int,
    max_new_tokens: int,
) -> Any:
    device = get_model_device(model)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length).input_ids.to(device)
    target = target_json.strip()
    if tokenizer.eos_token:
        target += tokenizer.eos_token
    target_ids = tokenizer(
        target,
        return_tensors="pt",
        truncation=True,
        max_length=max_new_tokens,
        add_special_tokens=False,
    ).input_ids.to(device)
    input_ids = tokenizer.pad_token_id * target_ids.new_ones((1, prompt_ids.shape[-1] + target_ids.shape[-1]))
    input_ids[:, : prompt_ids.shape[-1]] = prompt_ids
    input_ids[:, prompt_ids.shape[-1] :] = target_ids
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[-1]] = -100
    return model(input_ids=input_ids, labels=labels).loss


def evaluate(
    args: argparse.Namespace,
    torch: Any,
    model: Any,
    tokenizer: Any,
    examples: list[PlannerExample],
    available_sources: list[str],
    *,
    prefix: str,
) -> dict[str, float]:
    model.eval()
    rows = examples[: args.eval_max_samples] if args.eval_max_samples > 0 else examples
    if not rows:
        return {}

    rewards: list[float] = []
    json_valid = 0
    source_ok = 0
    ticker_ok = 0
    qa_type_ok = 0
    with torch.no_grad():
        for example in rows:
            prompt = format_planner_prompt(tokenizer, example, available_sources, args.lfm2_adapter_path)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length,
            )
            inputs = move_batch_to_device(inputs, get_model_device(model))
            prompt_len = inputs["input_ids"].shape[-1]
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response_text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
            reward, details = score_planner_response(response_text, example)
            rewards.append(reward)
            json_valid += int(details.get("json_valid", False))
            source_ok += int(details.get("source_ok", False))
            ticker_ok += int(details.get("ticker_ok", False))
            qa_type_ok += int(details.get("qa_type_ok", False))

    n = len(rows)
    payload = {
        "metric": prefix,
        "samples": n,
        "avg_reward": round(sum(rewards) / n, 4),
        "json_valid_rate": round(json_valid / n, 4),
        "source_accuracy": round(source_ok / n, 4),
        "ticker_accuracy": round(ticker_ok / n, 4),
        "qa_type_accuracy": round(qa_type_ok / n, 4),
    }
    print(json.dumps(payload, ensure_ascii=False))
    model.train()
    return {key: float(value) for key, value in payload.items() if isinstance(value, (int, float))}


def score_planner_response(text: str, example: PlannerExample) -> tuple[float, dict[str, Any]]:
    details: dict[str, Any] = {"json_valid": False}
    try:
        payload = json.loads(extract_json(text))
    except Exception:
        return -1.0, details
    if not isinstance(payload, dict):
        return -1.0, details

    details["json_valid"] = True
    score = 0.25

    strategy = str(payload.get("strategy", "")).lower()
    aggregation = str(payload.get("aggregation_mode", "")).lower()
    expected_aggregation = "concat" if example.qa_type == "single_hop" else "synthesize"

    if strategy in {"sequential", "parallel"}:
        score += 0.08
    if bool(payload.get("requires_retrieval")) is True:
        score += 0.12
    if bool(payload.get("requires_execution")) is True:
        score += 0.12
    if aggregation == expected_aggregation:
        score += 0.08

    ticker_ok = str(payload.get("ticker", "")).upper() == example.ticker.upper()
    qa_type_ok = str(payload.get("qa_type", "")).lower() == example.qa_type.lower()
    sources = payload.get("selected_sources") or []
    if isinstance(sources, str):
        sources = [sources]
    source_ok = any(Path(str(item)).name == Path(example.source_file).name for item in sources)
    details.update({"ticker_ok": ticker_ok, "qa_type_ok": qa_type_ok, "source_ok": source_ok})

    if ticker_ok:
        score += 0.15
    if qa_type_ok:
        score += 0.12
    if source_ok:
        score += 0.18

    sub_queries = payload.get("sub_queries")
    if isinstance(sub_queries, list) and sub_queries:
        score += 0.15
        if all(isinstance(item, dict) and item.get("query") for item in sub_queries):
            score += 0.08
        if all(isinstance(item, dict) and item.get("tool") == "retriever" for item in sub_queries):
            score += 0.05
        if example.qa_type == "single_hop" and len(sub_queries) == 1:
            score += 0.08
        if example.qa_type == "multi_hop" and (len(sub_queries) >= 2 or aggregation == "synthesize"):
            score += 0.08
        if any(example.ticker.lower() in str(item).lower() for item in sub_queries):
            score += 0.04

    if contains_direct_answer(text, example.answer):
        score -= 0.15
    if "```" in text:
        score -= 0.08

    return max(-1.0, min(1.5, score)), details


def contains_direct_answer(text: str, answer: str) -> bool:
    answer = normalize_for_match(answer)
    if len(answer) < 4:
        return False
    text_norm = normalize_for_match(text)
    return answer in text_norm


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def extract_json(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    return cleaned[start : end + 1]


def build_target_plan(example: PlannerExample) -> str:
    sub_queries = [{"id": "q1", "query": example.query, "type": "retrieval_qa", "depends_on": [], "tool": "retriever"}]
    if example.qa_type == "multi_hop":
        sub_queries = [
            {
                "id": "q1",
                "query": f"Tìm các số liệu gốc cần thiết trong báo cáo {example.ticker}: {example.query}",
                "type": "retrieval_qa",
                "depends_on": [],
                "tool": "retriever",
            },
            {
                "id": "q2",
                "query": f"Tổng hợp hoặc tính toán câu trả lời cho: {example.query}",
                "type": "calculation_qa",
                "depends_on": ["q1"],
                "tool": "retriever",
            },
        ]
    payload = {
        "strategy": "sequential",
        "requires_retrieval": True,
        "requires_execution": True,
        "aggregation_mode": "concat" if example.qa_type == "single_hop" else "synthesize",
        "ticker": example.ticker,
        "qa_type": example.qa_type,
        "selected_sources": [example.source_file],
        "sub_queries": sub_queries,
    }
    return json.dumps(payload, ensure_ascii=False)


def format_planner_prompt(
    tokenizer: Any,
    example: PlannerExample,
    available_sources: list[str],
    lfm2_adapter_path: str,
) -> str:
    source_list = ", ".join(available_sources)
    user_prompt = (
        f"Executor hiện tại: LFM2-1.2B-RAG LoRA tại {lfm2_adapter_path}\n"
        f"Nguồn tài liệu hợp lệ trong processed_data: {source_list}\n\n"
        f"Câu hỏi người dùng: {example.query}"
    )
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "apply_chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {PLANNER_SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"


def build_examples(questions_path: Path, answers_path: Path) -> list[PlannerExample]:
    questions = read_json_rows(questions_path)
    answers = {item["id"]: item for item in read_json_rows(answers_path)}
    examples: list[PlannerExample] = []
    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            raise RuntimeError(f"Missing reference answer for id={question['id']}")
        if question.get("qa_type") != answer.get("qa_type"):
            raise RuntimeError(f"qa_type mismatch for id={question['id']}")
        examples.append(
            PlannerExample(
                id=str(question["id"]),
                query=str(question["query"]),
                qa_type=str(question.get("qa_type") or ""),
                ticker=str(question.get("ticker") or ""),
                source_file=str(question["source_file"]),
                answer=str(answer.get("ground_truth_answer") or ""),
                evidence=str(answer.get("ground_truth_context") or ""),
            )
        )
    return examples


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {path}")
        return [dict(item) for item in payload]
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def list_processed_sources(processed_dir: Path) -> list[str]:
    if not processed_dir.exists():
        raise FileNotFoundError(processed_dir)
    return sorted(path.name for path in processed_dir.glob("*.txt") if path.is_file())


def validate_paths(args: argparse.Namespace, available_sources: list[str]) -> None:
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Qwen model folder not found: {args.model_path}")
    if args.require_lfm2_adapter and not Path(args.lfm2_adapter_path).exists():
        raise FileNotFoundError(
            f"LFM2 adapter folder not found: {args.lfm2_adapter_path}. "
            "Run scripts/train_lfm2_sft.py first or pass --no-require_lfm2_adapter."
        )
    if not available_sources:
        raise RuntimeError(f"No .txt files found in processed_dir={args.processed_dir}")


def build_lora_config(args: argparse.Namespace) -> Any:
    from peft import LoraConfig

    if args.target_modules == "all-linear":
        target_modules: str | list[str] = "all-linear"
    else:
        target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def move_batch_to_device(batch: Any, device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def prepare_model_for_low_memory_kbit_training(model: Any, *, gradient_checkpointing: bool) -> Any:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "config"):
        model.config.use_cache = False

    if not gradient_checkpointing:
        return model

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> None:
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    return model


def get_model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def update_baseline(current: float, reward: float, *, momentum: float, initialized: bool) -> float:
    if not initialized or math.isclose(current, 0.0):
        return reward
    return momentum * current + (1.0 - momentum) * reward


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def write_training_metadata(
    args: argparse.Namespace,
    output_dir: Path,
    train_examples: list[PlannerExample],
    eval_examples: list[PlannerExample],
    available_sources: list[str],
) -> None:
    payload = {
        "model_path": args.model_path,
        "lfm2_adapter_path": args.lfm2_adapter_path,
        "processed_dir": args.processed_dir,
        "train_questions": args.train_questions,
        "train_answers": args.train_answers,
        "eval_questions": args.eval_questions,
        "eval_answers": args.eval_answers,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "processed_sources": available_sources,
        "method": "clipped PPO-style policy optimization + small supervised planner loss",
    }
    (output_dir / "ppo_training_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(
    args: argparse.Namespace,
    train_examples: list[PlannerExample],
    eval_examples: list[PlannerExample],
    available_sources: list[str],
) -> None:
    payload = {
        "model_path": args.model_path,
        "lfm2_adapter_path": args.lfm2_adapter_path,
        "processed_dir": args.processed_dir,
        "output_dir": args.output_dir,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train_qa_type": dict(Counter(item.qa_type for item in train_examples)),
        "eval_qa_type": dict(Counter(item.qa_type for item in eval_examples)),
        "train_ticker": dict(Counter(item.ticker for item in train_examples)),
        "eval_ticker": dict(Counter(item.ticker for item in eval_examples)),
        "processed_sources": available_sources,
        "sample_target_plan": json.loads(build_target_plan(train_examples[0])) if train_examples else None,
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


def validate_cuda(args: argparse.Namespace, torch: Any) -> None:
    if torch.cuda.is_available():
        return
    if not args.require_cuda:
        return
    torch_cuda = getattr(torch.version, "cuda", None)
    raise RuntimeError(
        "CUDA is required for Qwen PPO training, but PyTorch cannot see a CUDA device.\n"
        f"Detected torch={torch.__version__}, torch_cuda={torch_cuda}.\n"
        "Fix the NVIDIA driver / PyTorch CUDA build mismatch first, then rerun this script. "
        "Qwen2.5-7B PPO is intentionally blocked on CPU."
    )


def validate_free_vram(args: argparse.Namespace, torch: Any) -> None:
    if not torch.cuda.is_available() or args.min_free_vram_gb <= 0:
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    if free_gb >= args.min_free_vram_gb:
        return
    raise RuntimeError(
        "Not enough free GPU memory for Qwen PPO before loading the model.\n"
        f"Free VRAM: {free_gb:.2f}GB / {total_gb:.2f}GB; required at least {args.min_free_vram_gb:.2f}GB.\n"
        "Run `nvidia-smi` and stop unrelated GPU processes first, or rerun with a lower threshold "
        "using `--min_free_vram_gb 0` if you intentionally want to risk OOM.\n"
        "For a 24GB GPU, close the process that is using ~14GB before PPO training."
    )


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
