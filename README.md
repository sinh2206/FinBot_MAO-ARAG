# VN Stock MAO ARAG

Hệ thống RAG/agent để hỏi đáp báo cáo tài chính chứng khoán Việt Nam.

## Luồng chạy nhanh

Nếu bạn chỉ muốn chạy ứng dụng chat:

1. Tạo môi trường Python.
2. Cài dependencies.
3. Copy `.env.example` sang `.env`.
4. Tải model local.
5. Chuẩn bị dữ liệu `raw_data -> processed_data -> chunks -> index`.
6. Chạy backend ở `backend/main.py`.
7. Mở frontend tại `http://localhost:8000` hoặc chạy frontend tĩnh riêng trên `8080`.

Backend hiện phục vụ luôn frontend tĩnh, nên mở `http://localhost:8000` là đủ trong đa số trường hợp.

## Kiến trúc

- `backend/`: FastAPI backend, cung cấp:
  - `GET /healthz`
  - `GET /api/config`
  - `POST /api/chat`
- `frontend/`: giao diện chat tĩnh.
- `rag_engine/`: các module RAG lõi.
- `scripts/`: script chuẩn bị dữ liệu, build index, train và đánh giá.
- `config/`: cấu hình model, agent và retrieval.

Luồng dữ liệu:

- `data/raw_data/` -> dữ liệu gốc.
- `data/processed_data/` -> dữ liệu đã trích xuất và làm sạch.
- `data/chunks/chunks.json` -> chunk dùng cho retrieval.
- `data/index/` -> FAISS/BM25 index.
- `data/metadata/index_summary.json` -> metadata của index.

## 1. Cài môi trường

Khuyến nghị Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu bạn chạy GPU CUDA 12.1:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

## 2. Cấu hình `.env`

Tạo file `.env` từ mẫu:

```bash
cp .env.example .env
```

Các biến quan trọng:

```env
ENABLE_LOCAL_PLANNER=true
ENABLE_LOCAL_EXECUTOR=true
ENABLE_GEMINI_FALLBACK_EXECUTOR=true
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>

LOCAL_FILES_ONLY=true
LOAD_IN_4BIT=true

PLANNER_MODEL_NAME=models/phi
EXECUTOR_MODEL_NAME=models/qwen
GEMINI_MODEL_NAME=gemini-2.5-flash

RAG_RETRIEVAL_MODE=hybrid
CORS_ALLOW_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
APP_HOST=0.0.0.0
APP_PORT=8000
```

Ghi chú:

- Nếu bạn chưa có Gemini API key, backend vẫn chạy, chỉ là phần fallback API sẽ tắt.
- Nếu model local chưa có sẵn, backend vẫn khởi động và sẽ rơi về retrieval/heuristic fallback.

## 3. Tải model local

Script tải model về `models/`:

```bash
python scripts/download_models.py
```

Model cần có:

- `models/phi/` cho planner.
- `models/qwen/` cho executor chính.
- `models/embedder/` cho dense retrieval.
- `models/cross_encoder/` cho rerank.

Nếu bạn tải thủ công:

```bash
mkdir -p models/phi models/qwen models/embedder models/cross_encoder

huggingface-cli download microsoft/Phi-4-mini-instruct \
  --local-dir models/phi \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/qwen \
  --local-dir-use-symlinks False

huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/embedder \
  --local-dir-use-symlinks False

huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --local-dir models/cross_encoder \
  --local-dir-use-symlinks False
```

## 4. Chuẩn bị dữ liệu

### 4.1. Từ raw data sang processed data

Nếu bạn đang có báo cáo gốc trong `data/raw_data/`, chạy:

```bash
python scripts/convert_raw_data.py
```

Kết quả mong đợi:

- văn bản sạch nằm trong `data/processed_data/`
- metadata chuyển đổi nằm trong `data/metadata/`

### 4.2. Build chunk và index

Chạy build index:

```bash
python scripts/build_index.py \
  --data_dir data/processed_data \
  --embedding_model models/embedder \
  --planner_model models/phi \
  --primary_executor_model models/qwen \
  --fallback_executor_model gemini-2.5-flash \
  --retrieval_mode hybrid \
  --chunk_dir data/chunks \
  --index_dir data/index \
  --embedding_dir data/embeddings \
  --metadata_dir data/metadata \
  --local_files_only
```

Nếu môi trường bị lỗi embedding/FAISS, bạn có thể tạm build sparse-only:

```bash
python scripts/build_index.py \
  --data_dir data/processed_data \
  --planner_model models/phi \
  --retrieval_mode sparse_only \
  --chunk_dir data/chunks \
  --index_dir data/index \
  --embedding_dir data/embeddings \
  --metadata_dir data/metadata \
  --local_files_only
```

## 5. Chạy backend local

Backend mới nằm ở `backend/main.py`.

```bash
python -m backend.main
```

Mở:

```text
http://localhost:8000
```

Backend sẽ:

- serve luôn `frontend/index.html`, `frontend/style.css`, `frontend/script.js`
- cung cấp API `/healthz`, `/api/config`, `/api/chat`
- cố gắng dùng model local nếu có
- nếu model local chưa sẵn sàng, backend vẫn trả về retrieval/heuristic fallback để frontend không bị lỗi trắng

## 6. Chạy frontend riêng

Nếu bạn muốn mở frontend tĩnh tách ra khỏi backend:

```bash
python -m http.server 8080 -d frontend
```

Mở:

```text
http://localhost:8080
```

Khi chạy kiểu này, `frontend/script.js` sẽ tự gọi backend ở `http://localhost:8000`.

## 7. Chạy bằng Docker

### 7.1. Build image

```bash
docker build -t vn-stock-mao-arag .
```

### 7.2. Chạy backend container

```bash
docker run --rm -p 8000:8000 --env-file .env -v "$PWD":/app vn-stock-mao-arag
```

### 7.3. Chạy bằng Docker Compose

Chạy backend:

```bash
docker compose up --build app
```

Chạy frontend tĩnh riêng trên `8080`:

```bash
docker compose --profile frontend up frontend
```

Ghi chú:

- Service `app` là backend chính và cũng serve frontend.
- Service `frontend` chỉ là server tĩnh tuỳ chọn cho trường hợp bạn muốn mở UI ở cổng `8080`.

### 7.4. Dừng container

```bash
docker compose down
```

Nếu muốn xóa luôn cache Hugging Face:

```bash
docker compose down -v
```

## 8. Kiểm tra backend

Bạn có thể kiểm tra nhanh bằng `curl`:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/api/config
```

Gửi chat thử:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Doanh thu thuần Quý III/2025 của BFC là bao nhiêu?", "history":[]}'
```

## 9. Fine-tune và đánh giá

Các bước này là tuỳ chọn nếu bạn muốn tái train hoặc chấm chất lượng.

### 9.1. Fine-tune executor Qwen

```bash
python scripts/fine-tune.py \
  --model_name_or_path models/qwen \
  --processed_dir data/processed_data \
  --train_questions data/train/questions.json \
  --train_answers data/train/reference_answers.json \
  --eval_questions data/test/questions.json \
  --eval_answers data/test/reference_answers.json \
  --output_dir models/qwen_executor_lora \
  --num_train_epochs 1 \
  --local_files_only
```

### 9.2. Train planner Phi

```bash
python scripts/ppo.py \
  --model_name_or_path models/phi \
  --processed_dir data/processed_data \
  --train_questions data/train/questions.json \
  --train_answers data/train/reference_answers.json \
  --eval_questions data/test/questions.json \
  --eval_answers data/test/reference_answers.json \
  --output_dir models/phi_planner_lora \
  --local_files_only
```

### 9.3. Đánh giá executor local-only

```bash
python scripts/evaluate_executor.py \
  --model_name_or_path models/qwen \
  --adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --predictions_file data/evaluation/qwen_executor_predictions.jsonl \
  --metrics_file data/evaluation/qwen_executor_metrics.json \
  --no-enable_gemini_fallback \
  --local_files_only
```

### 9.4. Đánh giá planner

```bash
python scripts/evaluate_planner.py \
  --model_name_or_path models/phi \
  --adapter_path models/phi_planner_lora \
  --processed_dir data/processed_data \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --predictions_file data/evaluation/phi_planner_predictions.jsonl \
  --metrics_file data/evaluation/phi_planner_metrics.json \
  --local_files_only
```

## 10. File chính cần nhớ

- `backend/app.py`: khai báo FastAPI app và các route.
- `backend/main.py`: entrypoint chạy backend.
- `frontend/index.html`: giao diện chat.
- `frontend/script.js`: logic gọi API.
- `Dockerfile`: image cho backend.
- `docker-compose.yml`: chạy local bằng compose.

## 11. Lỗi thường gặp

### Không thấy dữ liệu trả về

- Kiểm tra `data/processed_data/` và `data/chunks/chunks.json`.
- Nếu chưa có, chạy lại `scripts/convert_raw_data.py` và `scripts/build_index.py`.

### Backend chạy nhưng không gọi được model local

- Kiểm tra `models/phi/` và `models/qwen/`.
- Nếu folder chưa tồn tại, backend sẽ tự fallback sang retrieval/heuristic.

### Frontend ở cổng 8080 báo lỗi CORS

- Đảm bảo `CORS_ALLOW_ORIGINS` có `http://localhost:8080` và `http://127.0.0.1:8080`.
- Hoặc mở frontend trực tiếp từ `http://localhost:8000` để cùng origin.

### Gemini fallback không hoạt động

- Kiểm tra `GEMINI_API_KEY`.
- Đảm bảo `ENABLE_GEMINI_FALLBACK_EXECUTOR=true`.

