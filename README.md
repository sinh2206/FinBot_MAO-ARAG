# VN Stock MAO ARAG

Du an minh hoa RAG agent cho truy van chung khoan Viet Nam. Luong chinh:

1. `core/orchestrator.py` nhan cau hoi tu UI.
2. `agents/planner_agent.py` lap workflow bang Qwen local hoac heuristic fallback.
3. `agents/retriever_agent.py` truy xuat tai lieu bang FAISS + BM25.
4. `agents/reranker_agent.py` sap xep lai bang cross-encoder.
5. `agents/executor_agent.py` tra loi extractive QA bang MiniMax local hoac heuristic fallback.
6. `agents/aggregator_agent.py` tong hop cac sub-answer.

## Cai dat

Yeu cau nen co:

- Python 3.10+.
- Git va quyen ghi vao thu muc du an.
- GPU CUDA neu muon chay Qwen/MiniMax local. CPU van chay duoc che do demo/sparse-only, nhung khong phu hop cho LLM lon.
- Tai khoan Hugging Face va `HF_TOKEN` neu model can xac thuc.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Kiem tra nhanh moi truong:

```bash
python -m compileall main.py core agents rag_engine tools ui scripts tests
python scripts/download_models.py --dry_run --only embedder cross_encoder
```

## Chuan bi cau hinh

Tao file cau hinh local tu template:

```bash
copy .env.example .env
```

Tren macOS/Linux:

```bash
cp .env.example .env
```

Che do nhe de kiem tra pipeline truoc khi tai LLM:

```env
ENABLE_LOCAL_PLANNER=false
ENABLE_LOCAL_EXECUTOR=false
ENABLE_RERANKER=false
RAG_RETRIEVAL_MODE=sparse_only
```

Sau khi da tai model ve `models/`, doi sang local path:

```env
ENABLE_LOCAL_PLANNER=true
ENABLE_LOCAL_EXECUTOR=true
ENABLE_RERANKER=true

QWEN_MODEL_NAME=models/qwen
MINIMAX_MODEL_NAME=models/minimax
EMBEDDING_MODEL_NAME=models/embedder
RERANKER_MODEL_NAME=models/cross_encoder

RAG_RETRIEVAL_MODE=hybrid
RAG_DOCUMENT_PATH=data/chunks/chunks.json
RAG_INDEX_PATH=data/index
```

Y nghia cac file config:

- `config/agent_config.yaml`: tham so cho planner, executor, retriever, reranker.
- `config/model_config.yaml`: repo Hugging Face va local folder cho tung model.
- `config/retrieval_config.yaml`: `chunk_size`, overlap, `top_k`, dense/sparse/hybrid weight.
- `.env`: cau hinh runtime uu tien cao nhat cho UI va scripts local.

## Chuan bi model local

Model duoc tai ve thu muc `models/`:

```text
models/
  qwen/           # Qwen/Qwen2.5-7B-Instruct, dung cho planner_agent
  minimax/        # MiniMaxAI/MiniMax-M2.1, dung cho executor_agent
  embedder/       # sentence-transformers/all-MiniLM-L6-v2
  cross_encoder/  # cross-encoder/ms-marco-MiniLM-L-6-v2
```

Tai rieng model nhe truoc de build index va test retrieval:

```bash
python scripts/download_models.py --only embedder cross_encoder
```

Tai Qwen planner:

```bash
python scripts/download_models.py --only qwen
```

Tai MiniMax executor:

```bash
python scripts/download_models.py --only minimax
```

Tai tat ca:

```bash
python scripts/download_models.py
```

Neu model yeu cau xac thuc, dat token truoc khi tai:

```bash
set HF_TOKEN=hf_xxx
python scripts/download_models.py --only qwen minimax
```

Tren macOS/Linux:

```bash
export HF_TOKEN=hf_xxx
python scripts/download_models.py --only qwen minimax
```

Luu y thuc te:

- Qwen2.5-7B-Instruct duoc dung lam `planner_agent`: tach sub-query, quyet dinh sequential/parallel, tra workflow JSON.
- MiniMax-M2.1 duoc dung lam `executor_agent`: extractive QA tren cac doan van da retrieve/rerank.
- MiniMax-M2.1 co the rat nang. Neu may khong du GPU/RAM, giu `ENABLE_LOCAL_EXECUTOR=false` de dung heuristic fallback trong khi phat trien pipeline.
- Tham so `quantize: 4bit` trong YAML la cau hinh muc tieu. Code hien tai load qua `transformers`; neu muon ep 4-bit that su, can bo sung `bitsandbytes`/`BitsAndBytesConfig` phu hop voi moi truong CUDA.

## Chay UI

```bash
streamlit run main.py
```

Mac dinh `.env` dang de cac model nang o che do tat (`false`) va retrieval la `sparse_only` de demo chay nhe. Khi da tai model local, co the doi:

```env
ENABLE_LOCAL_PLANNER=true
ENABLE_LOCAL_EXECUTOR=true
ENABLE_RERANKER=true
RAG_RETRIEVAL_MODE=hybrid
```

Khong can OpenAI API hay API ngoai. Tat ca model duoc cau hinh de chay local.

## Du lieu

Dat tai lieu goc trong `data/documents/` voi dinh dang `.txt`, `.md`, `.pdf` hoac `.docx`, sau do build index:

```bash
python scripts/build_index.py --data_dir data/documents
```

Script se tao:

- `data/chunks/chunks.json`
- `data/index/`
- `data/embeddings/embeddings.npy`
- `data/metadata/`

Co the nap nhanh corpus dang JSON tai `data/chunks/chunks.json` voi moi item co dang:

```json
{"id": "doc_1", "text": "VNINDEX dong cua o 1.280 diem.", "metadata": {"source": "demo"}}
```

Neu chua co corpus/index, UI se dung vai doan demo ngan de van khoi dong duoc.

Checklist truoc khi training/fine-tune hoac danh gia nghiem tuc:

- Cai dependency va compile code thanh cong.
- Tai it nhat `embedder` de build vector index.
- Build index tu `data/documents`.
- Xac nhan `.env` dang tro toi local model path trong `models/`.
- Chay UI voi mot cau hoi mau va kiem tra debug panel co workflow, passages va score.
- Khong commit `models/`, index, embeddings hoac `.env`; cac muc nay da duoc `.gitignore` bo qua.

## Lenh van hanh nhanh

```bash
python scripts/download_models.py --only embedder cross_encoder
python scripts/build_index.py --data_dir data/documents
streamlit run main.py
```

## Chay tren Google Colab T4

Mo [main.ipynb](main.ipynb) tren Colab, chon runtime GPU T4, sau do chon `Runtime > Run all`.

Notebook se tu dong:

- Clone `https://github.com/sinh2206/vn_stock_mao_arag.git` vao `/content/vn_stock_mao_arag`.
- Cai dependencies tu `requirements.txt`.
- Tao `.env` cho Colab.
- Chuan bi chunks tu CafeF data neu co, neu khong thi dung `storage_rag`, neu van khong co thi dung demo fallback.
- Tai model can thiet vao `models/`.
- Chay sanity test pipeline RAG.
- Chay dry-run PPO readiness cho Qwen planner.

Mac dinh tren T4:

- Qwen2.5-7B-Instruct planner duoc bat va load 4-bit neu model tai thanh cong.
- MiniMax-M2.1 executor tat de tranh OOM tren T4; executor dung extractive heuristic fallback.
- Retrieval mode la `sparse_only` de dam bao notebook chay on dinh. Co the doi sang `hybrid` sau khi da build dense index.

Neu muon thu MiniMax tren runtime lon hon T4, sua cell cau hinh:

```python
USE_MINIMAX_EXECUTOR = True
DOWNLOAD_MINIMAX = True
```

Danh gia pipeline voi tap QA JSON/JSONL:

```bash
python scripts/evaluate_pipeline.py --qa_file data/metadata/qa_eval.json
```

## Xu ly data CafeF

Neu co thu muc `cafef_news-20260211T204544Z-1-001/`, chuyen no thanh corpus RAG, training JSONL va artifact giong `storage_rag`:

```bash
python scripts/process_cafef_news.py --input_dir cafef_news-20260211T204544Z-1-001
```

Output chinh:

- `data/documents/cafef_news.jsonl`: documents chuan hoa tu CSV.
- `data/chunks/cafef_news_chunks.json`: chunks de build/search RAG.
- `data/training/cafef_extractive_qa.jsonl`: QA extractive cho executor.
- `data/training/cafef_planner_workflows.jsonl`: workflow examples cho planner.
- `storage_rag/cafef_news/<TICKER>/`: `docstore.json`, `default__vector_store.json`, `image__vector_store.json`, `graph_store.json`, `index_store.json`.
- `data/metadata/cafef_processing_summary.json`: thong ke output.

Chay dry-run PPO cho Qwen planner:

```bash
python scripts/train_qwen_ppo.py --dry_run
```

Chay PPO that su can GPU, `models/qwen` da tai ve, va dependency `trl`, `peft`, `torch`:

```bash
python scripts/train_qwen_ppo.py --model_path models/qwen --dataset data/training/cafef_planner_workflows.jsonl --output_dir models/qwen_ppo
```
