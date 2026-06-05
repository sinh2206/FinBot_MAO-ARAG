# VN Stock MAO ARAG

Dự án RAG/agent cho hỏi đáp báo cáo tài chính chứng khoán Việt Nam.

Kiến trúc hiện tại:

- **Microsoft `Phi-4-mini-instruct`** làm agent planner/coordinator local: đọc câu hỏi, chọn tài liệu nguồn, sinh JSON plan.
- **Qwen/Qwen2.5-7B-Instruct** làm agent executor chính local: đọc context và sinh câu trả lời cuối cùng.
- **Gemini `gemini-2.5-flash`** làm agent executor dự phòng: chỉ dùng khi Qwen trả về rỗng hoặc `KHÔNG TÌM THẤY`.
- **SentenceTransformers + FAISS/BM25** làm tầng truy xuất offline.
- `frontend/` là giao diện web chat tĩnh, đơn giản kiểu ChatGPT.

## 1. Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng GPU CUDA 12.1 cho Qwen/Phi:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

## 2. Cấu hình môi trường

Tạo `.env`:

```bash
cp .env.example .env
```

Sửa các biến chính:

```env
ENABLE_LOCAL_PLANNER=true
PLANNER_PROVIDER=local
PLANNER_MODEL_NAME=models/phi
PHI_MODEL_NAME=models/phi

ENABLE_LOCAL_EXECUTOR=true
EXECUTOR_PROVIDER=local
EXECUTOR_MODEL_NAME=models/qwen
QWEN_MODEL_NAME=models/qwen

ENABLE_GEMINI_FALLBACK_EXECUTOR=true
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
GEMINI_MODEL_NAME=gemini-2.5-flash

EMBEDDING_MODEL_NAME=models/embedder
RERANKER_MODEL_NAME=models/cross_encoder
LOCAL_FILES_ONLY=true
LOAD_IN_4BIT=true
```

## 3. Tải model local

Cần tải 4 nhóm model local:

- `models/phi/` cho planner
- `models/qwen/` cho executor chính
- `models/embedder/` cho dense retrieval
- `models/cross_encoder/` cho rerank

```bash
python scripts/download_models.py
```

Hoặc tải thủ công:

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

## 4. Build index

`scripts/build_index.py` chỉ build retrieval index. Script ghi metadata planner là `models/phi` và executor fallback là `gemini-2.5-flash` vào `data/metadata/index_summary.json`.

Hybrid retrieval:

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

Nếu môi trường bị lỗi `torch.float8_e8m0fnu` khi load `sentence-transformers`, build tạm BM25-only:

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

## 5. Fine-tune Qwen executor

Kiểm tra dữ liệu:

```bash
python scripts/fine-tune.py \
  --model_name_or_path models/qwen \
  --processed_dir data/processed_data \
  --train_questions data/train/questions.json \
  --train_answers data/train/reference_answers.json \
  --eval_questions data/test/questions.json \
  --eval_answers data/test/reference_answers.json \
  --output_dir models/qwen_executor_lora \
  --dry_run
```

Train mặc định cho máy còn trống khoảng 20 GB VRAM:

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

## 6. Fine-tune / đánh giá Phi planner

`scripts/ppo.py` giờ là entrypoint cho planner Phi local. Script này train LoRA/SFT trên dữ liệu train, rồi có thể chấm trên dữ liệu test.

Chạy khô:

```bash
python scripts/ppo.py \
  --model_name_or_path models/phi \
  --processed_dir data/processed_data \
  --train_questions data/train/questions.json \
  --train_answers data/train/reference_answers.json \
  --eval_questions data/test/questions.json \
  --eval_answers data/test/reference_answers.json \
  --output_dir models/phi_planner_lora \
  --prepare_only
```

Train và evaluate với preset mạnh mặc định:

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

Đánh giá riêng planner sau khi đã có adapter:

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

## 7. Đánh giá Qwen executor

`scripts/evaluate_executor.py` luôn chạy Qwen trước. Nếu bật fallback và có `GEMINI_API_KEY`, Gemini chỉ được gọi khi Qwen trả về rỗng hoặc `KHÔNG TÌM THẤY`.

Chấm khách quan local-only:

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

Chấm độ bền có fallback:

```bash
python scripts/evaluate_executor.py \
  --model_name_or_path models/qwen \
  --adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --predictions_file data/evaluation/qwen_executor_with_fallback_predictions.jsonl \
  --metrics_file data/evaluation/qwen_executor_with_fallback_metrics.json \
  --enable_gemini_fallback \
  --local_files_only
```

## 8. Kiểm thử toàn bộ pipeline 3 model

Mục này dùng để kiểm tra sự gắn kết của:

- planner `Phi`
- executor chính `Qwen`
- executor dự phòng `Gemini`

Nguyên tắc nên dùng:

- `data/train/`: chạy smoke test và integration test, vì đây là bộ dữ liệu model đã nhìn thấy khi fine-tune.
- `data/test/`: chạy chấm cuối để xem khả năng tổng quát hóa.
- muốn đo chất lượng local thật sự thì tắt Gemini;
- muốn kiểm tra cơ chế fallback 3 model thì bật Gemini và xem cột `executor_used`.

### 8.1. Smoke test planner trên train

```bash
python scripts/evaluate_planner.py \
  --model_name_or_path models/phi \
  --adapter_path models/phi_planner_lora \
  --processed_dir data/processed_data \
  --questions data/train/questions.json \
  --answers data/train/reference_answers.json \
  --predictions_file data/evaluation/train_phi_planner_predictions.jsonl \
  --metrics_file data/evaluation/train_phi_planner_metrics.json \
  --local_files_only
```

### 8.2. Kiểm tra riêng câu hỏi `multi_hop`

Với `multi_hop`, planner phải sinh được câu hỏi thành phần và phải gắn với đúng `source_file` trong `data/processed_data/`. Kiểm tra nhanh:

```bash
python - <<'PY'
import json
from pathlib import Path

pred_path = Path("data/evaluation/train_phi_planner_predictions.jsonl")
rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
processed_dir = Path("data/processed_data")
issues = []

for row in rows:
    if row.get("qa_type") != "multi_hop":
        continue
    plan = row.get("predicted_plan") or {}
    sub_queries = plan.get("sub_queries") or []
    selected_sources = plan.get("selected_sources") or []
    source_file = row["source_file"]
    source_exists = (processed_dir / source_file).exists() or (processed_dir / source_file.replace(".ocr_text.txt", ".txt")).exists()
    ok = len(sub_queries) >= 2 and source_file in selected_sources and source_exists
    if not ok:
        issues.append(
            {
                "id": row["id"],
                "source_file": source_file,
                "selected_sources": selected_sources,
                "sub_query_count": len(sub_queries),
            }
        )

print("multi_hop_total =", sum(1 for row in rows if row.get("qa_type") == "multi_hop"))
print("multi_hop_issues =", len(issues))
for item in issues[:20]:
    print(json.dumps(item, ensure_ascii=False))
PY
```

Nếu muốn xem planner đã tách câu hỏi thành phần như thế nào:

```bash
python - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("data/evaluation/train_phi_planner_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    if row.get("qa_type") != "multi_hop":
        continue
    plan = row.get("predicted_plan") or {}
    print("=" * 80)
    print(row["id"], "-", row["query"])
    for item in plan.get("sub_queries") or []:
        print("-", item.get("id"), ":", item.get("query"))
PY
```

### 8.3. Chấm Qwen local-only trên train

Ở bước này, đáp án cuối cùng được so với `reference_answers.json`.

```bash
python scripts/evaluate_executor.py \
  --model_name_or_path models/qwen \
  --adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --questions data/train/questions.json \
  --answers data/train/reference_answers.json \
  --predictions_file data/evaluation/train_qwen_executor_predictions.jsonl \
  --metrics_file data/evaluation/train_qwen_executor_metrics.json \
  --no-enable_gemini_fallback \
  --local_files_only
```

Lọc riêng các câu `multi_hop` đang sai nhiều nhất:

```bash
python - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("data/evaluation/train_qwen_executor_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
multi = [row for row in rows if row.get("qa_type") == "multi_hop"]
bad = sorted(multi, key=lambda row: row.get("executor_score", 0.0))

print("multi_hop_total =", len(multi))
print("multi_hop_wrong =", sum(1 for row in multi if (row.get("numeric_accuracy") or 0.0) < 1.0 or row.get("em", 0) == 0))
for row in bad[:10]:
    print(json.dumps({
        "id": row["id"],
        "prediction": row["prediction"],
        "answer": row["answer"],
        "numeric_accuracy": row["numeric_accuracy"],
        "executor_score": row["executor_score"],
    }, ensure_ascii=False))
PY
```

### 8.4. Kiểm tra fallback Gemini trên train

Bước này không dùng để lấy benchmark local. Nó chỉ dùng để xem khi Qwen không trả lời được thì Gemini có vào đúng lúc hay không.

```bash
python scripts/evaluate_executor.py \
  --model_name_or_path models/qwen \
  --adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --questions data/train/questions.json \
  --answers data/train/reference_answers.json \
  --predictions_file data/evaluation/train_qwen_executor_with_fallback_predictions.jsonl \
  --metrics_file data/evaluation/train_qwen_executor_with_fallback_metrics.json \
  --enable_gemini_fallback \
  --local_files_only
```

Đếm số câu Qwen tự trả lời và số câu phải rơi sang Gemini:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in Path("data/evaluation/train_qwen_executor_with_fallback_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
print(Counter(row.get("executor_used", "unknown") for row in rows))
PY
```

### 8.5. Chạy lại toàn bộ trên test

Sau khi smoke test trên `train` ổn, lặp lại đúng flow trên cho `data/test/`:

1. `evaluate_planner.py` với:
   - `--questions data/test/questions.json`
   - `--answers data/test/reference_answers.json`
2. kiểm tra riêng `multi_hop` từ file `data/evaluation/test_phi_planner_predictions.jsonl`
3. `evaluate_executor.py --no-enable_gemini_fallback` để lấy benchmark local thật sự
4. `evaluate_executor.py --enable_gemini_fallback` để xem độ bền của pipeline 3 model

Ví dụ cho planner trên test:

```bash
python scripts/evaluate_planner.py \
  --model_name_or_path models/phi \
  --adapter_path models/phi_planner_lora \
  --processed_dir data/processed_data \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --predictions_file data/evaluation/test_phi_planner_predictions.jsonl \
  --metrics_file data/evaluation/test_phi_planner_metrics.json \
  --local_files_only
```

Ví dụ cho executor local-only trên test:

```bash
python scripts/evaluate_executor.py \
  --model_name_or_path models/qwen \
  --adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --predictions_file data/evaluation/test_qwen_executor_predictions.jsonl \
  --metrics_file data/evaluation/test_qwen_executor_metrics.json \
  --no-enable_gemini_fallback \
  --local_files_only
```

Ví dụ cho executor có fallback trên test:

```bash
python scripts/evaluate_executor.py \
  --model_name_or_path models/qwen \
  --adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --predictions_file data/evaluation/test_qwen_executor_with_fallback_predictions.jsonl \
  --metrics_file data/evaluation/test_qwen_executor_with_fallback_metrics.json \
  --enable_gemini_fallback \
  --local_files_only
```

### 8.6. Chạy end-to-end bằng `scripts/evaluate.py`

Nếu muốn chấm toàn bộ kiến trúc hiện tại trong một lệnh, dùng `scripts/evaluate.py`.

Script này sẽ chạy theo đúng luồng:

- `Phi` sinh plan
- plan phải có `selected_sources` và `sub_queries`
- với `multi_hop`, các `sub_queries` retrieval phải tìm được bằng chứng trong `data/processed_data`
- `Qwen` trả lời trước
- `Gemini` chỉ được gọi khi Qwen trả về rỗng hoặc `KHÔNG TÌM THẤY`
- đáp án cuối cùng được chấm với `reference_answers.json`

Chạy toàn bộ trên `data/test`:

```bash
python scripts/evaluate.py \
  --planner_model_name_or_path models/phi \
  --planner_adapter_path models/phi_planner_lora \
  --executor_model_name_or_path models/qwen \
  --executor_adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --chunks_file data/chunks/chunks.json \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --plans_file data/evaluation/system_plans.jsonl \
  --predictions_file data/evaluation/system_predictions.jsonl \
  --metrics_file data/evaluation/system_metrics.json \
  --output_file output/system_output1.json \
  --enable_gemini_fallback \
  --local_files_only
```

Chỉ chấm `multi_hop`:

```bash
python scripts/evaluate.py \
  --planner_model_name_or_path models/phi \
  --planner_adapter_path models/phi_planner_lora \
  --executor_model_name_or_path models/qwen \
  --executor_adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --chunks_file data/chunks/chunks.json \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --qa_type_filter multi_hop \
  --plans_file data/evaluation/system_multi_hop_plans.jsonl \
  --predictions_file data/evaluation/system_multi_hop_predictions.jsonl \
  --metrics_file data/evaluation/system_multi_hop_metrics.json \
  --output_file output/system_output_multi_hop.json \
  --enable_gemini_fallback \
  --local_files_only
```

Kiểm tra trước khi load model:

```bash
python scripts/evaluate.py \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --qa_type_filter multi_hop \
  --prepare_only
```

Các file đầu ra chính:

- `data/evaluation/system_plans.jsonl`: plan do Phi sinh
- `data/evaluation/system_predictions.jsonl`: kết quả chấm từng câu
- `data/evaluation/system_metrics.json`: metric tổng hợp
- `output/system_output1.json`: báo cáo cuối cùng

Trong `system_predictions.jsonl`, với `multi_hop` cần chú ý thêm các trường:

- `component_count_ok`: planner có sinh đủ retrieval sub-query hay không
- `component_support_rate`: tỷ lệ sub-query retrieval tìm được evidence trong `processed_data`
- `evidence_number_coverage`: độ phủ các con số thành phần so với `ground_truth_context`
- `retrieval_component_score`: điểm phần retrieval/component trước khi chấm đáp án cuối
- `executor_used`: `qwen` hay `gemini`

Kết luận nhanh:

- `phi_planner_metrics.json`: xem planner có chọn đúng nguồn và sinh được `sub_queries` hay không;
- `*_qwen_executor_metrics.json`: xem Qwen local trả lời tốt đến đâu;
- `*_with_fallback_predictions.jsonl`: xem Gemini có chỉ được gọi khi Qwen thất bại hay không, bằng trường `executor_used`;
- với `multi_hop`, phải xem cả 2 tầng:
  - planner có tách được câu hỏi thành phần, gắn đúng `source_file`;
  - executor có trả lời đúng đáp án cuối khi so với `reference_answers.json`.

## 9. Chạy bằng Docker

Repo hiện tại phù hợp với Docker theo 2 service:

- `runner`: container để chạy các script như build index, fine-tune, evaluate
- `frontend`: container serve `frontend/` tĩnh qua `python -m http.server`

### 9.1. Build image

```bash
docker compose build
```

### 9.2. Khởi động container runner

```bash
docker compose up -d runner
```

Vào shell trong container:

```bash
docker compose exec runner bash
```

Từ đây có thể chạy toàn bộ script của dự án, ví dụ:

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

Hoặc:

```bash
python scripts/evaluate.py \
  --planner_model_name_or_path models/phi \
  --planner_adapter_path models/phi_planner_lora \
  --executor_model_name_or_path models/qwen \
  --executor_adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --chunks_file data/chunks/chunks.json \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --qa_type_filter multi_hop \
  --enable_gemini_fallback \
  --local_files_only
```

### 9.3. Chạy lệnh một lần không cần vào shell

Ví dụ build index:

```bash
docker compose run --rm runner \
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

Ví dụ evaluate end-to-end:

```bash
docker compose run --rm runner \
  python scripts/evaluate.py \
  --planner_model_name_or_path models/phi \
  --planner_adapter_path models/phi_planner_lora \
  --executor_model_name_or_path models/qwen \
  --executor_adapter_path models/qwen_executor_lora \
  --processed_dir data/processed_data \
  --chunks_file data/chunks/chunks.json \
  --questions data/test/questions.json \
  --answers data/test/reference_answers.json \
  --qa_type_filter multi_hop \
  --enable_gemini_fallback \
  --local_files_only
```

### 9.4. Chạy backend chat bằng Docker

Backend hiện nằm trong `main.py`, tự serve luôn `frontend/` và các API:

- `GET /healthz`
- `GET /api/config`
- `POST /api/chat`

Chạy service backend:

```bash
docker compose up -d app
```

Mở:

```text
http://localhost:8000
```

### 9.5. Chạy frontend tĩnh bằng Docker

```bash
docker compose --profile frontend up -d frontend
```

Mở:

```text
http://localhost:8080
```

Khi mở bản tĩnh ở cổng `8080`, `frontend/script.js` sẽ tự gọi backend ở `http://localhost:8000`.

### 9.6. Dừng container

```bash
docker compose down
```

Nếu muốn xóa luôn volume cache Hugging Face:

```bash
docker compose down -v
```

### 9.7. Ghi chú vận hành

- `app` là service backend chính để kết nối frontend với pipeline `Phi -> Qwen -> Gemini fallback`.
- `runner` mount toàn bộ repo vào `/app`, nên file sinh ra trong container sẽ hiện ngay ở máy host.
- cache Hugging Face được giữ trong volume `huggingface-cache`, giúp không phải tải lại model mỗi lần.
- nếu host có NVIDIA Container Toolkit và muốn dùng GPU cho model local, mở comment phần `deploy.resources.reservations.devices` trong `docker-compose.yml`.
- Dockerfile hiện không chạy `streamlit`; backend chat dùng `main.py` và tự serve luôn `frontend/`.

## 10. Frontend chat

Frontend nằm ở:

- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`
- `main.py`

Chạy local đầy đủ:

```bash
python main.py
```

Mở:

```text
http://localhost:8000
```

`main.py` sẽ serve luôn `index.html`, `style.css`, `script.js` và API `/api/chat`. Nếu backend của bạn chạy endpoint khác, sửa `API_ENDPOINT` trong `frontend/script.js`.

## 11. File này dùng để làm gì?

- `tools/evaluation.py`: thư viện metric dùng chung, gồm `normalize_answer`, `exact_match`, `f1_score`.
- `scripts/fine-tune.py`: fine-tune Qwen executor local.
- `scripts/ppo.py`: fine-tune/evaluate Phi planner local.
- `scripts/evaluate_executor.py`: đánh giá executor Qwen, có Gemini fallback khi Qwen không trả lời.
- `scripts/evaluate_planner.py`: đánh giá riêng planner Phi.
- `scripts/evaluate.py`: pipeline end-to-end chính cho kiến trúc `Phi -> Qwen -> Gemini fallback`.

## 12. Lỗi thường gặp

### Gemini API key lỗi

Kiểm tra `.env`:

```bash
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
GEMINI_MODEL_NAME=gemini-2.5-flash
ENABLE_GEMINI_FALLBACK_EXECUTOR=true
```

Nếu chạy script từ terminal, export trực tiếp cũng được:

```bash
export GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
```

### CUDA OOM khi fine-tune Qwen

- preset mặc định hiện dành cho khoảng `20GiB` VRAM và ưu tiên chất lượng hơn tốc độ;
- nếu vẫn OOM, giảm lần lượt `--per_device_train_batch_size 1`, `--max_seq_length 2048`, rồi `--target_modules q_proj,k_proj,v_proj,o_proj`;
- giữ `--optim paged_adamw_8bit` và `--prepare_kbit_mode minimal`;
- dừng tiến trình khác đang chiếm VRAM.

### CUDA OOM khi train Phi planner

- preset mặc định hiện dành cho khoảng `20GiB` VRAM;
- nếu vẫn OOM, giảm lần lượt `--per_device_train_batch_size 1`, `--max_seq_length 2048`, rồi `--target_modules q_proj,k_proj,v_proj,o_proj`;
- giữ `--prepare_kbit_mode minimal`;
- nếu cần, chạy `--prepare_only` trước để kiểm tra dữ liệu.

### `torch.float8_e8m0fnu` khi build index

Lỗi này do `transformers` quá mới so với `torch` hiện tại. `sentence-transformers` import `transformers`, rồi `transformers` đòi dtype FP8 mà bản Torch của bạn chưa có.

Cách nhanh nhất là build index không dùng dense embedding:

```bash
python scripts/build_index.py \
  --data_dir data/processed_data \
  --planner_model models/phi \
  --retrieval_mode sparse_only \
  --local_files_only
```

Hoặc ghim dependency lại:

```bash
pip install 'transformers<5' 'sentence-transformers<5'
```
