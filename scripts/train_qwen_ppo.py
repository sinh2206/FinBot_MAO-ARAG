from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO reinforcement learning skeleton for the Qwen planner agent.")
    parser.add_argument("--model_path", default="models/qwen", help="Local Qwen model folder.")
    parser.add_argument("--dataset", default="data/training/cafef_planner_workflows.jsonl")
    parser.add_argument("--output_dir", default="models/qwen_ppo")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.dataset)
    if args.dry_run:
        print(json.dumps({"prompts": len(prompts), "model_path": args.model_path}, indent=2))
        return
    if not prompts:
        raise RuntimeError(f"No prompts found in {args.dataset}")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Qwen model folder not found: {args.model_path}")

    try:
        import torch
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "PPO training requires extra dependencies. Install: pip install trl peft accelerate torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
    )

    config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
    )
    trainer = PPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=tokenizer)

    for step, prompt in enumerate(prompts[: args.max_steps], start=1):
        query = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
        response = trainer.generate(query, max_new_tokens=512, do_sample=True, top_p=0.9)
        response_text = tokenizer.decode(response[0][query.shape[-1] :], skip_special_tokens=True)
        reward = torch.tensor([score_planner_response(response_text)], device=model.device)
        trainer.step([query[0]], [response[0][query.shape[-1] :]], reward)
        print(json.dumps({"step": step, "reward": float(reward.item())}, ensure_ascii=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved PPO-tuned Qwen planner to {output_dir}")


def load_prompts(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.exists():
        return []
    prompts = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            system = next((item["content"] for item in messages if item.get("role") == "system"), "")
            user = next((item["content"] for item in messages if item.get("role") == "user"), "")
            prompts.append(f"{system}\n\nUser: {user}\nAssistant:")
    return prompts


def score_planner_response(text: str) -> float:
    try:
        payload = json.loads(extract_json(text))
    except Exception:
        return -1.0
    score = 0.0
    if payload.get("strategy") in {"sequential", "parallel"}:
        score += 0.4
    if isinstance(payload.get("sub_queries"), list) and payload["sub_queries"]:
        score += 0.4
    if payload.get("requires_retrieval") is not None:
        score += 0.1
    if payload.get("requires_execution") is not None:
        score += 0.1
    return score


def extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("No JSON object found")
    return text[start : end + 1]


if __name__ == "__main__":
    main()
