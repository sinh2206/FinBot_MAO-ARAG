# Local Models

This folder stores downloaded model snapshots. Large model files are ignored by git.

Expected layout:

```text
models/
  qwen/           # Qwen/Qwen2.5-7B-Instruct, planner_agent
  minimax/        # MiniMaxAI/MiniMax-M2.1, executor_agent
  embedder/       # sentence-transformers/all-MiniLM-L6-v2
  cross_encoder/  # cross-encoder/ms-marco-MiniLM-L-6-v2
```

Download commands:

```bash
python scripts/download_models.py --only embedder cross_encoder
python scripts/download_models.py --only qwen
python scripts/download_models.py --only minimax
```

After download, point `.env` to local folders:

```env
QWEN_MODEL_NAME=models/qwen
MINIMAX_MODEL_NAME=models/minimax
EMBEDDING_MODEL_NAME=models/embedder
RERANKER_MODEL_NAME=models/cross_encoder
```
