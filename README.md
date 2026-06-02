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

## Chay tu dau den cuoi tren Linux server

Phan nay gia dinh server Ubuntu/Linux, co GPU NVIDIA neu muon chay Qwen/MiniMax local. Neu chi test pipeline RAG, CPU van chay duoc voi `sparse_only`.

### 1. Kiem tra server

```bash
pwd
python3 --version
git --version
nvidia-smi || true
```

Khuyen nghi:

- Python 3.10 hoac 3.11.
- GPU NVIDIA 16GB+ VRAM de thu Qwen 7B 4-bit.
- MiniMax-M2.1 rat nang; khong nen bat tren GPU T4/16GB neu chua test rieng.
- Dung `tmux` hoac `screen` khi tai model/build index/training de tranh mat session SSH.

### 2. Clone repo

```bash
cd ~
git clone https://github.com/sinh2206/vn_stock_mao_arag.git
cd vn_stock_mao_arag
```

Neu repo da clone:

```bash
cd ~/vn_stock_mao_arag
git pull --ff-only
```

### 3. Tao moi truong Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Neu server dung CUDA va can cai lai PyTorch dung ban CUDA phu hop, cai PyTorch truoc, sau do moi cai requirements. Vi du voi CUDA 12.1:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

Kiem tra import co ban:

```bash
python -m compileall main.py core agents rag_engine tools ui scripts tests
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

### 4. Cau hinh runtime

```bash
cp .env.example .env
```

Che do an toan de test pipeline truoc:

```env
ENABLE_LOCAL_PLANNER=false
ENABLE_LOCAL_EXECUTOR=false
ENABLE_RERANKER=false
LOCAL_FILES_ONLY=true
LOAD_IN_4BIT=false
RAG_RETRIEVAL_MODE=sparse_only
RAG_DOCUMENT_PATH=data/chunks/chunks.json
RAG_INDEX_PATH=data/index
```

Sau khi model da tai xong va muon bat Qwen planner:

```env
ENABLE_LOCAL_PLANNER=true
ENABLE_LOCAL_EXECUTOR=false
ENABLE_RERANKER=false
LOCAL_FILES_ONLY=true
LOAD_IN_4BIT=true
QWEN_MODEL_NAME=models/qwen
MINIMAX_MODEL_NAME=models/minimax
EMBEDDING_MODEL_NAME=models/embedder
RERANKER_MODEL_NAME=models/cross_encoder
RAG_RETRIEVAL_MODE=sparse_only
```

Chi bat MiniMax khi server du RAM/VRAM:

```env
ENABLE_LOCAL_EXECUTOR=true
```

### 5. Tai model

Neu model tren Hugging Face yeu cau token:

```bash
export HF_TOKEN=hf_xxx
```

Tai model nhe cho retrieval/index:

```bash
python scripts/download_models.py --only embedder cross_encoder
```

Tai Qwen planner:

```bash
python scripts/download_models.py --only qwen
```

Tai MiniMax executor neu server du tai nguyen:

```bash
python scripts/download_models.py --only minimax
```

Kiem tra file model:

```bash
find models -maxdepth 2 -type f | head -50
```

### 6. Chuan bi data

Neu co raw CafeF folder:

```bash
python scripts/process_cafef_news.py --input_dir cafef_news-20260211T204544Z-1-001
```

Lenh nay tao:

```text
data/documents/cafef_news.jsonl
data/chunks/cafef_news_chunks.json
data/training/cafef_extractive_qa.jsonl
data/training/cafef_planner_workflows.jsonl
storage_rag/cafef_news/<TICKER>/
```

Neu dung document rieng, copy vao:

```bash
mkdir -p data/documents
cp /path/to/files/* data/documents/
```

Build index tu documents:

```bash
python scripts/build_index.py --data_dir data/documents --local_files_only
```

Neu chi muon test nhanh bang chunks CafeF da process, co the cau hinh:

```env
RAG_DOCUMENT_PATH=data/chunks/cafef_news_chunks.json
RAG_RETRIEVAL_MODE=sparse_only
```

### 7. Chay test/smoke test

Chay unit tests:

```bash
pytest -q
```

Neu chua cai pytest:

```bash
python -m pip install pytest
pytest -q
```

Smoke test pipeline bang Python:

```bash
python - <<'PY'
import json
from pathlib import Path
from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.reranker_agent import RerankerAgent
from agents.retriever_agent import RetrieverAgent
from core.orchestrator import Orchestrator
from rag_engine.retriever import HybridRetriever, HybridRetrieverConfig
from rag_engine.schema import Document

path = Path("data/chunks/cafef_news_chunks.json")
if not path.exists():
    path = Path("data/chunks/chunks.json")
docs = [Document.from_any(x, i) for i, x in enumerate(json.loads(path.read_text(encoding="utf-8"))[:5000])]
retriever = HybridRetriever(documents=docs, config=HybridRetrieverConfig(mode="sparse_only"))
app = Orchestrator(
    planner_agent=PlannerAgent(enable_llm=False),
    retriever_agent=RetrieverAgent(retriever=retriever),
    reranker_agent=RerankerAgent(enable_model=False),
    executor_agent=ExecutorAgent(enable_model=False),
)
result = app.run("Tóm tắt tin liên quan đến FPT")
print(result["answer"][:1000])
print(result["plan"])
PY
```

### 8. Chay UI tren server

Chay noi bo:

```bash
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

Mo firewall/security group port `8501`, hoac tunnel qua SSH:

```bash
ssh -L 8501:localhost:8501 user@server_ip
```

Sau do mo tren may ca nhan:

```text
http://localhost:8501
```

### 9. Chay nhu service nen

Dung `tmux`:

```bash
tmux new -s mao-rag
source .venv/bin/activate
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

Detach: `Ctrl-b`, sau do bam `d`.

Attach lai:

```bash
tmux attach -t mao-rag
```

## Training va test hieu qua

Can tach 3 viec khac nhau:

1. **RAG indexing**: khong phai training LLM. Day la buoc bien documents/chunks thanh index de truy xuat.
2. **Planner training**: huan luyen Qwen tao workflow JSON dung.
3. **Executor training**: huan luyen/kiem tra extractive QA, uu tien tra loi dung span trong context.

### 1. Chuan bi dataset dung

Dataset nen nam trong:

```text
data/training/cafef_planner_workflows.jsonl
data/training/cafef_extractive_qa.jsonl
data/metadata/qa_eval.json
```

Planner row nen co dang chat messages:

```json
{"messages":[{"role":"system","content":"Bạn là planner_agent..."},{"role":"user","content":"So sánh FPT và HPG"},{"role":"assistant","content":"{\"strategy\":\"parallel\",\"sub_queries\":[...]}"}]}
```

Executor row nen co:

```json
{"question":"Tin CafeF về FPT có tiêu đề gì?","context":"...","answer":"..."}
```

Nguyen tac chat luong:

- Bo dong answer khong nam nguyen van trong context.
- Tach train/validation/test theo thoi gian de tranh leakage. Vi du train truoc 2025-08, validation 2025-08, test 2025-09.
- Khong dua prediction/report sinh boi model vao train neu chua audit.
- Giu metadata `ticker`, `date`, `source`, `url` de debug loi.

### 2. Baseline truoc khi training

Luon do baseline truoc:

```bash
python scripts/evaluate_pipeline.py \
  --qa_file data/metadata/qa_eval.json \
  --chunks_file data/chunks/cafef_news_chunks.json \
  --retrieval_mode sparse_only \
  --output_file data/metadata/evaluation_sparse.json
```

Sau do moi so sanh voi hybrid/reranker/local LLM:

```bash
python scripts/evaluate_pipeline.py \
  --qa_file data/metadata/qa_eval.json \
  --chunks_file data/chunks/cafef_news_chunks.json \
  --retrieval_mode hybrid \
  --output_file data/metadata/evaluation_hybrid.json
```

Metric can theo doi:

- Retrieval: recall@5, recall@10 neu co expected doc ids.
- Executor: Exact Match, token F1, ty le answer nam trong context.
- Planner: ty le JSON parse duoc, dung strategy, dung so sub-query, dung tool.
- End-to-end: answer EM/F1 va latency.

### 3. PPO cho Qwen planner

Kiem tra dataset/prompt truoc:

```bash
python scripts/train_qwen_ppo.py --dry_run
```

Chay thu rat nho:

```bash
python scripts/train_qwen_ppo.py \
  --model_path models/qwen \
  --dataset data/training/cafef_planner_workflows.jsonl \
  --output_dir models/qwen_ppo \
  --max_steps 10 \
  --batch_size 1 \
  --mini_batch_size 1
```

Neu on dinh moi tang:

```bash
python scripts/train_qwen_ppo.py \
  --model_path models/qwen \
  --dataset data/training/cafef_planner_workflows.jsonl \
  --output_dir models/qwen_ppo \
  --max_steps 200 \
  --batch_size 2 \
  --mini_batch_size 1 \
  --learning_rate 1e-6
```

Can theo doi:

- Reward trung binh co tang khong.
- Output co con parse duoc JSON khong.
- Model co bi lap text hoac them markdown khong.
- VRAM co on dinh khong: `watch -n 1 nvidia-smi`.

Luu y: PPO script hien la skeleton reward theo format JSON/workflow. De training thuc su hieu qua, nen nang reward len dua tren validation set: parse JSON, chon dung sequential/parallel, sub-query lien quan, va end-to-end answer tot hon baseline.

### 4. Executor training/test

Voi executor extractive QA, khong nen uu tien sinh tu do. Truoc tien test heuristic/retrieval:

```bash
python scripts/evaluate_pipeline.py \
  --qa_file data/training/cafef_extractive_qa.jsonl \
  --chunks_file data/chunks/cafef_news_chunks.json \
  --retrieval_mode sparse_only \
  --output_file data/metadata/executor_eval.json
```

Neu muon fine-tune executor, dataset phai dam bao:

- `answer` la substring cua `context`.
- Context khong qua dai.
- Cau hoi khong mo ho.
- Tach ticker/date de khong leak cung bai vao train va test.

### 5. Quy trinh khuyen nghi

Thu tu nen lam:

```text
1. Chay sparse_only + heuristic planner/executor.
2. Tao qa_eval.json nho, co ground truth ro.
3. Do baseline EM/F1.
4. Build hybrid index va so sanh recall/latency.
5. Bat reranker neu latency chap nhan duoc.
6. Bat Qwen planner 4-bit va so sanh planner JSON validity.
7. Chi training PPO sau khi baseline/eval da on.
8. Khong bat MiniMax executor neu chua co GPU/RAM du va eval baseline ro.
```

Tieu chi “hieu qua” thuc dung:

- Neu retrieval recall thap, sua chunking/index truoc, chua training LLM.
- Neu planner sai workflow, fine-tune/PPO Qwen planner.
- Neu executor bia hoac tra loi ngoai context, ep extractive prompt va loc dataset span.
- Neu latency cao, giam `top_k`, tat reranker, hoac cache index/model.

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
