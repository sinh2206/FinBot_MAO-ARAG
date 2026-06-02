from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODELS = {
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
    "minimax": ("MiniMaxAI/MiniMax-M2.1", "minimax"),
    "embedder": ("sentence-transformers/all-MiniLM-L6-v2", "embedder"),
    "cross_encoder": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross_encoder"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Hugging Face models into the local models/ folder.")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--only", nargs="*", choices=sorted(DEFAULT_MODELS), help="Download only selected model groups.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token, or set HF_TOKEN in the environment.")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.only or list(DEFAULT_MODELS)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: pip install huggingface-hub") from exc

    for key in selected:
        repo_id, folder = DEFAULT_MODELS[key]
        local_dir = models_dir / folder
        print(f"{key}: {repo_id} -> {local_dir}")
        if args.dry_run:
            continue
        snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            token=args.token,
            local_dir=str(local_dir),
            force_download=args.force_download,
        )

    print("Done.")


if __name__ == "__main__":
    main()
