# VN Stock MAO ARAG

Dự án minh họa kiến trúc multi-agent RAG cho truy vấn chứng khoán Việt Nam. Luồng chính:

1. `core/orchestrator.py` nhận câu hỏi từ UI.
2. `agents/planner_agent.py` dùng Qwen2.5-7B-Instruct local hoặc heuristic fallback để lập workflow.
3. `agents/retriever_agent.py` truy xuất tài liệu bằng FAISS + BM25.
4. `agents/reranker_agent.py` sắp xếp lại đoạn văn bằng cross-encoder.
5. `agents/executor_agent.py` dùng LFM2-1.2B-RAG local hoặc heuristic fallback để trả lời extractive QA.
6. `agents/aggregator_agent.py` tổng hợp các câu trả lời con.

## Vì sao đổi executor sang LFM2-1.2B-RAG

Executor của dự án cần trả lời bám sát các đoạn đã retrieve/rerank, không cần một model quá lớn để suy luận mở rộng. MiniMax-M2.1 quá nặng cho T4/16GB và nhiều server thử nghiệm, dễ gây OOM hoặc làm chậm vòng lặp đánh giá. Vì vậy executor mặc định được đổi sang:

```text
LiquidAI/LFM2-1.2B-RAG
```

LFM2-1.2B-RAG phù hợp hơn cho vai trò này vì:

- Model khoảng 1.2B tham số, nhẹ hơn đáng kể cho inference local.
- Được tối ưu cho Retrieval-Augmented Generation và trả lời dựa trên tài liệu đầu vào.
- Khuyến nghị chạy greedy decoding với `temperature=0`, đúng với yêu cầu extractive QA.
- Context dài 32K token, hữu ích khi cần đưa nhiều đoạn tin tài chính vào executor.

Tài liệu tham khảo:

- Hugging Face: <https://huggingface.co/LiquidAI/LFM2-1.2B-RAG>
- Liquid Docs: <https://docs.liquid.ai/lfm/models/lfm2-1.2b-rag>

## Cài đặt

Yêu cầu khuyến nghị:

- Python 3.10 hoặc 3.11.
- Git và quyền ghi vào thư mục dự án.
- GPU NVIDIA nếu muốn chạy Qwen/LFM2 local. CPU vẫn chạy được chế độ demo hoặc `sparse_only`, nhưng không phù hợp để thử LLM local nghiêm túc.
- Tài khoản Hugging Face và `HF_TOKEN` nếu model yêu cầu xác thực.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Trên Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Kiểm tra nhanh:

```bash
python -m compileall main.py core agents rag_engine tools ui scripts tests
python scripts/download_models.py --dry_run --only embedder cross_encoder lfm2_rag qwen
```

## Chạy từ đầu trên Linux server

Hiện dự án chưa có frontend tách riêng. Màn hình thao tác duy nhất là Streamlit chạy trong service app và mở qua port `8501`. Cách khuyến nghị trên server Linux là dùng Docker Compose để đóng gói môi trường Python, còn dữ liệu/model vẫn mount từ thư mục dự án trên server.

### Cách A. Chạy bằng Docker Compose

1. Kiểm tra server có Docker:

```bash
docker --version
docker compose version
nvidia-smi || true
```

Nếu chưa có Docker, cài Docker Engine và Docker Compose plugin theo tài liệu Linux của Docker. Nếu muốn chạy LLM local bằng GPU trong container, server cần NVIDIA driver và NVIDIA Container Toolkit.

2. Clone repo:

```bash
cd ~
git clone https://github.com/sinh2206/vn_stock_mao_arag.git
cd vn_stock_mao_arag
```

Nếu repo đã có:

```bash
cd ~/vn_stock_mao_arag
git pull --ff-only
```

3. Tạo `.env` cho runtime:

```bash
cp .env.example .env
```

Cấu hình nhẹ để chạy ngay trên server:

```env
ENABLE_GEMINI_API=false
ENABLE_LOCAL_PLANNER=false
ENABLE_LOCAL_EXECUTOR=false
ENABLE_RERANKER=false
LOCAL_FILES_ONLY=true
LOAD_IN_4BIT=false
RAG_RETRIEVAL_MODE=sparse_only
RAG_DOCUMENT_PATH=data/chunks/cafef_news_chunks.json
RAG_INDEX_PATH=data/index
```

4. Build image:

```bash
docker compose build
```

5. Nếu cần xử lý lại raw CafeF trước khi chạy app:

```bash
docker compose run --rm app python scripts/process_cafef_news.py \
  --input_dir cafef_news_ \
  --storage_dir storage_rag/cafef_news \
  --documents_out data/documents/cafef_news.jsonl \
  --chunks_out data/chunks/cafef_news_chunks.json \
  --training_dir data/training \
  --metadata_dir data/metadata
```

Nếu dữ liệu hiện tại vẫn còn trong `data/`, bước này có thể bỏ qua.

6. Chạy app Streamlit trên server:

```bash
docker compose up -d
```

Mở từ máy cá nhân:

```text
http://server_ip:8501
```

Hoặc tunnel qua SSH:

```bash
ssh -L 8501:localhost:8501 user@server_ip
```

Sau đó mở:

```text
http://localhost:8501
```

7. Xem log và dừng app:

```bash
docker compose logs -f app
docker compose down
```

8. Chạy test trong container:

```bash
docker compose run --rm app python -m pytest -q
```

9. Tải model local trong container nếu muốn bật Qwen/LFM2:

```bash
docker compose run --rm app python scripts/download_models.py --only embedder cross_encoder lfm2_rag qwen
```

Sau khi tải xong, sửa `.env`:

```env
ENABLE_LOCAL_PLANNER=true
ENABLE_LOCAL_EXECUTOR=true
LOAD_IN_4BIT=true
QWEN_MODEL_NAME=models/qwen
EXECUTOR_MODEL_NAME=models/lfm2_rag
EMBEDDING_MODEL_NAME=models/embedder
RERANKER_MODEL_NAME=models/cross_encoder
```

Rồi restart:

```bash
docker compose up -d --force-recreate
```

Các volume trong `docker-compose.yml`:

- `./data:/app/data`: documents, chunks, index, metadata, training files.
- `./storage_rag:/app/storage_rag`: processed RAG artifacts.
- `./models:/app/models`: Hugging Face model files và adapter.
- `./reports:/app/reports`: báo cáo/evaluation output.
- `./cafef_news_:/app/cafef_news_:ro`: raw CafeF, mount read-only.

### Cách B. Chạy trực tiếp bằng Python

### 1. Kiểm tra server

```bash
pwd
python3 --version
git --version
nvidia-smi || true
```

Khuyến nghị:

- GPU 8GB+ VRAM có thể thử LFM2 executor.
- GPU 16GB+ VRAM phù hợp hơn khi bật Qwen2.5-7B-Instruct 4-bit.
- Dùng `tmux` hoặc `screen` khi tải model, build index hoặc training để tránh mất session SSH.

### 2. Clone repo

```bash
cd ~
git clone https://github.com/sinh2206/vn_stock_mao_arag.git
cd vn_stock_mao_arag
```

Nếu repo đã clone:

```bash
cd ~/vn_stock_mao_arag
git pull --ff-only
```

### 3. Tạo môi trường Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Nếu cần cài PyTorch đúng CUDA, cài PyTorch trước rồi mới cài requirements. Ví dụ CUDA 12.1:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

Kiểm tra CUDA:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

### 4. Cấu hình runtime

Tạo `.env`:

```bash
cp .env.example .env
```

Chế độ nhẹ để test pipeline trước:

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

Sau khi đã tải model local và muốn bật Qwen planner + LFM2 executor:

```env
ENABLE_LOCAL_PLANNER=true
ENABLE_LOCAL_EXECUTOR=true
ENABLE_RERANKER=false
LOCAL_FILES_ONLY=true
LOAD_IN_4BIT=true

QWEN_MODEL_NAME=models/qwen
EXECUTOR_MODEL_NAME=models/lfm2_rag
EMBEDDING_MODEL_NAME=models/embedder
RERANKER_MODEL_NAME=models/cross_encoder

RAG_RETRIEVAL_MODE=sparse_only
RAG_DOCUMENT_PATH=data/chunks/chunks.json
RAG_INDEX_PATH=data/index
```

Ghi chú tương thích: code vẫn đọc `MINIMAX_MODEL_NAME` nếu `.env` cũ còn biến này, nhưng cấu hình mới nên dùng `EXECUTOR_MODEL_NAME`.

### 5. Tải model

Nếu Hugging Face yêu cầu token:

```bash
export HF_TOKEN=hf_xxx
```

Tải model nhẹ cho retrieval/index:

```bash
python scripts/download_models.py --only embedder cross_encoder
```

Tải Qwen planner:

```bash
python scripts/download_models.py --only qwen
```

Tải LFM2 executor:

```bash
python scripts/download_models.py --only lfm2_rag
```

Tải tất cả model mặc định:

```bash
python scripts/download_models.py
```

Kiểm tra file model:

```bash
find models -maxdepth 2 -type f | head -50
```

## Chuẩn bị dữ liệu

Trong repo này có hai nhóm dữ liệu cần phân biệt rõ:

- `cafef_news_/`: raw data CafeF, gồm nhiều CSV tin tức, sentiment, giá và feature theo mã cổ phiếu. Đây là nguồn đầu vào để chuẩn hóa lại.
- `storage_rag/`: processed data đã có dạng artifact RAG theo từng ticker, gồm `docstore.json`, `default__vector_store.json`, `image__vector_store.json`, `graph_store.json`, `index_store.json`. Đây là dữ liệu đã xử lý, dùng để đối chiếu, kiểm tra lại chunk/context hoặc làm nguồn phục hồi corpus.

### Trạng thái dữ liệu hiện tại

Tính theo artifact đang có trong workspace, dữ liệu đã được xử lý:

- `cafef_news_/` vẫn là thư mục raw chính, hiện có các CSV tin tức, sentiment, feature giá và một số artifact phụ từ các thử nghiệm cũ.
- `data/metadata/cafef_processing_summary.json` ghi `input_dir` là `cafef_news_/cafef_news`, nghĩa là raw folder này đã được chạy qua pipeline chuẩn hóa.
- Kết quả đã sinh `34.440` documents vào `data/documents/cafef_news.jsonl`.
- Kết quả đã sinh `44.715` chunks vào `data/chunks/cafef_news_chunks.json`.
- Kết quả đã sinh `20.000` dòng QA extractive vào `data/training/cafef_extractive_qa.jsonl`.
- Kết quả đã sinh `30` workflow planner vào `data/training/cafef_planner_workflows.jsonl`.
- `storage_rag/cafef_news/` hiện có processed artifact cho `38` ticker từ CafeF.
- `storage_rag/` cấp gốc cũng có nhiều thư mục ticker đã xử lý sẵn như `FPT`, `HPG`, `VCB`, `SSI`, dùng tốt cho audit context/RAG nhưng nên đồng bộ lại từ raw nếu muốn training sạch.

Vì vậy không cần xử lý lại chỉ để chạy demo. Chỉ xử lý lại khi muốn tái tạo sạch dataset, đổi chunk size, đổi rule lọc dữ liệu, hoặc chuẩn bị một lần huấn luyện mới.

### Xử lý raw `cafef_news_/`

Chạy pipeline chuẩn hóa raw CSV:

```bash
python scripts/process_cafef_news.py \
  --input_dir cafef_news_ \
  --storage_dir storage_rag/cafef_news \
  --documents_out data/documents/cafef_news.jsonl \
  --chunks_out data/chunks/cafef_news_chunks.json \
  --training_dir data/training \
  --metadata_dir data/metadata
```

Script sẽ:

- Đọc CSV bằng nhiều encoding (`utf-8-sig`, `utf-8`, `cp1258`, `latin1`).
- Nhận diện dòng `news`, `daily_sentiment`, `price_feature` hoặc bảng thường.
- Chuẩn hóa mỗi dòng thành `Document` có `ticker`, `target_ticker`, `date`, `title`, `url`, `source`.
- Sinh chunk RAG vào `data/chunks/cafef_news_chunks.json`.
- Sinh QA extractive cho executor vào `data/training/cafef_extractive_qa.jsonl`.
- Sinh workflow mẫu cho planner vào `data/training/cafef_planner_workflows.jsonl`.
- Ghi summary vào `data/metadata/cafef_processing_summary.json`.
- Ghi artifact dạng `storage_rag/cafef_news/<TICKER>/`.

Các file `.pt`, `.pth`, `.joblib`, `.png` trong raw folder không dùng làm corpus text cho LFM2. Chỉ đưa vào training những dòng có text đủ dài, metadata rõ và không trùng fingerprint.

### Dùng processed `storage_rag/`

`storage_rag/` hiện đã có processed data theo ticker như `FPT`, `HPG`, `VCB`, `SSI`. Nếu cần kiểm tra hoặc tái tạo corpus từ processed data, ưu tiên đọc `docstore.json` vì text nằm ở:

```text
docstore/data/*/__data__/text
```

Quy tắc sử dụng:

- Dùng `storage_rag/<TICKER>/docstore.json` để audit context, kiểm tra câu trả lời LFM2 có bám nguồn không.
- Không fine-tune trực tiếp từ vector store hoặc embedding; chỉ fine-tune từ text/context và QA đã kiểm định.
- Nếu cần đồng bộ lại format mới, chạy lại `scripts/process_cafef_news.py` từ raw `cafef_news_/` để sinh `data/*` nhất quán.
- Không đưa toàn bộ báo cáo dài vào một sample SFT; phải chunk, chọn context vừa đủ, rồi tạo answer ngắn có căn cứ.

Nếu dùng tài liệu riêng, đặt vào `data/documents/`, rồi build index:

```bash
mkdir -p data/documents
cp /path/to/files/* data/documents/
python scripts/build_index.py --data_dir data/documents --local_files_only
```

Nếu chỉ muốn test nhanh bằng chunks CafeF đã xử lý:

```env
RAG_DOCUMENT_PATH=data/chunks/cafef_news_chunks.json
RAG_RETRIEVAL_MODE=sparse_only
```

## Gemini API ẩn

Dự án có thể cài sẵn Gemini API để phục vụ soạn dữ liệu phụ trợ hoặc audit thủ công sau này, nhưng mặc định phải ẩn và không tham gia pipeline đánh giá local models.

Cài SDK chính thức:

```bash
pip install google-genai
```

Trong `.env`, giữ trạng thái tắt:

```env
ENABLE_GEMINI_API=false
GEMINI_API_KEY=
GEMINI_MODEL_NAME=gemini-2.5-flash
```

Khi cần dùng riêng ngoài benchmark, đặt key trong `.env` local hoặc biến môi trường:

```bash
export GEMINI_API_KEY=your_key_here
```

Không commit `.env`; `.gitignore` đã loại `.env` và `.env.*`. Theo tài liệu Google AI, SDK Python hiện dùng package `google-genai`, client tự đọc `GEMINI_API_KEY` hoặc `GOOGLE_API_KEY` từ môi trường. Nguồn tham khảo: <https://ai.google.dev/gemini-api/docs/quickstart> và <https://ai.google.dev/gemini-api/docs/api-key>.

Quy tắc hiện tại:

- Không gọi Gemini trong `core/orchestrator.py`.
- Không gọi Gemini trong `scripts/evaluate_pipeline.py`.
- Không dùng Gemini làm judge khi so sánh Qwen planner và LFM2 executor.
- Không dùng output Gemini làm ground truth nếu chưa audit thủ công.
- Nếu sau này bật Gemini để tạo candidate QA, file sinh ra phải để riêng, ví dụ `data/training/gemini_candidates.jsonl`, rồi lọc thủ công trước khi nhập vào `lfm2_rag_sft.jsonl`.

## Chạy test và UI

Chạy unit test:

```bash
PYTHONPATH=. python -m pytest -q
```

Nếu chưa cài pytest:

```bash
python -m pip install pytest
PYTHONPATH=. python -m pytest -q
```

Smoke test pipeline bằng Python:

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

Chạy UI:

```bash
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

Nếu chạy qua SSH:

```bash
ssh -L 8501:localhost:8501 user@server_ip
```

Sau đó mở:

```text
http://localhost:8501
```

## Đánh giá trước khi training

Cần tách ba việc:

1. RAG indexing: không phải training LLM, chỉ biến documents/chunks thành index để truy xuất.
2. Planner training: huấn luyện Qwen2.5-7B-Instruct tạo workflow JSON đúng.
3. Executor training: fine-tune LFM2-1.2B-RAG để trả lời bám context, ưu tiên trích xuất đúng span và tránh hallucination.

Luôn đo baseline trước:

```bash
python scripts/evaluate_pipeline.py \
  --qa_file data/metadata/qa_eval.json \
  --chunks_file data/chunks/cafef_news_chunks.json \
  --retrieval_mode sparse_only \
  --output_file data/metadata/evaluation_sparse.json
```

So sánh với hybrid:

```bash
python scripts/evaluate_pipeline.py \
  --qa_file data/metadata/qa_eval.json \
  --chunks_file data/chunks/cafef_news_chunks.json \
  --retrieval_mode hybrid \
  --output_file data/metadata/evaluation_hybrid.json
```

Metric nên theo dõi:

- Retrieval: recall@5, recall@10 nếu có expected doc ids.
- Executor LFM2: Exact Match, token F1, tỷ lệ câu trả lời là substring hoặc được chứng minh trực tiếp từ context.
- Planner Qwen: tỷ lệ JSON parse được, đúng strategy, đúng số sub-query, đúng tool.
- End-to-end: EM/F1, groundedness, latency và VRAM.

## Fine-tune executor LFM2-1.2B-RAG

### 1. Mục tiêu fine-tune

Không fine-tune LFM2 để "biết thêm" dữ liệu chứng khoán. Dữ liệu mới phải nằm trong RAG index. Fine-tune executor chỉ nên nhằm:

- Tuân thủ prompt tiếng Việt.
- Trả lời ngắn, có căn cứ, không bịa ngoài context.
- Biết trả lời `KHÔNG TÌM THẤY` khi context không đủ.
- Trích xuất đúng số liệu, mã cổ phiếu, ngày, tiêu đề, nguồn tin.

### 2. Format dataset SFT

File khuyến nghị:

```text
data/training/lfm2_rag_sft.jsonl
```

Mỗi dòng nên là chat messages:

```json
{"messages":[{"role":"system","content":"Bạn là executor_agent cho RAG chứng khoán Việt Nam. Chỉ trả lời dựa trên tài liệu được cung cấp. Nếu không có căn cứ, trả lời KHÔNG TÌM THẤY."},{"role":"user","content":"Tài liệu:\n<document1>FPT công bố lợi nhuận sau thuế quý II đạt 2.100 tỷ đồng...</document1>\n\nCâu hỏi: Lợi nhuận sau thuế quý II của FPT là bao nhiêu?"},{"role":"assistant","content":"2.100 tỷ đồng"}]}
```

Nguyên tắc dữ liệu:

- `assistant.content` phải có căn cứ trực tiếp trong context.
- Thêm mẫu phủ định, trong đó context không chứa đáp án và output là `KHÔNG TÌM THẤY`.
- Tách train/validation/test theo thời gian để tránh leakage, ví dụ train trước `2025-08-01`, validation trong tháng `2025-08`, test từ `2025-09-01`.
- Giữ metadata ngoài prompt nếu cần debug: `ticker`, `date`, `source`, `url`, `doc_id`.
- Không đưa báo cáo do model sinh vào train nếu chưa audit thủ công.

### 3. Chuyển CafeF QA sang SFT cho LFM2

Sau khi chạy `scripts/process_cafef_news.py`, file gần nhất với SFT executor là:

```text
data/training/cafef_extractive_qa.jsonl
```

File này có dạng:

```json
{"question":"Tin CafeF về FPT ngày ... có tiêu đề gì?","context":"...","answer":"...","metadata":{"ticker":"FPT","date":"...","source":"..."}}
```

Để fine-tune LFM2, chuyển mỗi row sang chat format:

```json
{"messages":[{"role":"system","content":"Bạn là executor_agent cho RAG chứng khoán Việt Nam. Chỉ trả lời dựa trên tài liệu được cung cấp. Nếu không có căn cứ, trả lời KHÔNG TÌM THẤY."},{"role":"user","content":"Tài liệu:\n<document1>...</document1>\n\nCâu hỏi: ..."},{"role":"assistant","content":"..."}],"metadata":{"ticker":"FPT","date":"...","source":"..."}}
```

Quy trình lọc trước khi train:

- Giữ row có `answer` nằm nguyên văn trong `context`.
- Loại row quá dài hoặc context nhiễu bảng trống.
- Loại row thiếu `ticker`, `source` hoặc mốc thời gian nếu cần đánh giá theo thời gian.
- Tạo thêm negative samples bằng cách ghép câu hỏi với context sai ticker/ngày và đặt answer là `KHÔNG TÌM THẤY`.
- Chia train/valid/test theo `date`, không random theo dòng vì tin cùng ngày rất dễ leak.

Lệnh gợi ý cho bước chuyển format nếu viết script riêng:

```bash
python scripts/prepare_lfm2_sft.py \
  --input data/training/cafef_extractive_qa.jsonl \
  --train_out data/training/lfm2_rag_sft.jsonl \
  --valid_out data/training/lfm2_rag_valid.jsonl \
  --test_out data/training/lfm2_rag_test.jsonl
```

Nếu dùng `storage_rag/` làm nguồn kiểm tra, chỉ đọc text từ `docstore.json`, sau đó tạo QA cùng format trên. Không dùng embedding hoặc vector id làm target training.

### 4. SFT bằng LoRA/QLoRA với TRL

Cài thêm dependency:

```bash
pip install "transformers>=4.55" accelerate peft trl bitsandbytes datasets
```

Lệnh mẫu nếu có script SFT riêng:

```bash
python scripts/train_lfm2_sft.py \
  --model_name_or_path models/lfm2_rag \
  --train_file data/training/lfm2_rag_sft.jsonl \
  --eval_file data/training/lfm2_rag_valid.jsonl \
  --output_dir models/lfm2_rag_lora \
  --load_in_4bit \
  --learning_rate 2e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 8192
```

Nếu chưa có `scripts/train_lfm2_sft.py`, tạo script này trước theo `trl.SFTTrainer` + `peft.LoraConfig`. Không ghi đè `models/lfm2_rag`; chỉ lưu adapter vào `models/lfm2_rag_lora`.

## PPO cho Qwen planner khi train lại

Nguồn planner:

```text
data/training/cafef_planner_workflows.jsonl
```

Chạy dry-run:

```bash
python scripts/train_qwen_ppo.py --dry_run
```

Chạy PPO nhỏ:

```bash
python scripts/train_qwen_ppo.py \
  --model_path models/qwen \
  --dataset data/training/cafef_planner_workflows.jsonl \
  --output_dir models/qwen_ppo \
  --max_steps 10 \
  --batch_size 1 \
  --mini_batch_size 1
```

Khi ổn định mới tăng `max_steps`. PPO hiện tối ưu workflow JSON cho planner; không dùng Gemini làm judge.

## Đánh giá và promote model sau train lại

Sau khi có adapter/checkpoint mới, chạy lại benchmark cùng một QA file:

```bash
python scripts/evaluate_pipeline.py \
  --qa_file data/training/cafef_extractive_qa.jsonl \
  --chunks_file data/chunks/cafef_news_chunks.json \
  --retrieval_mode sparse_only \
  --output_file data/metadata/evaluation_after_training.json
```

Chỉ promote model nếu:

- Retrieval recall không giảm.
- LFM2 giảm hallucination và tăng EM/F1 hoặc groundedness.
- Qwen planner tăng JSON validity và workflow đúng hơn.
- Latency/VRAM vẫn phù hợp runtime mục tiêu.
- Gemini vẫn tắt trong toàn bộ quá trình đánh giá hai model local.


## Các bước huấn luyện lại từ đầu

Quy trình train lại sạch cho dự án là tái tạo data từ raw `cafef_news_/`, build lại RAG artifacts, fine-tune adapter LFM2 executor và PPO checkpoint Qwen planner. Không pretrain LFM2 từ random init.

1. Dọn artifact sinh lại, nhưng giữ nguyên `cafef_news_/`, `storage_rag/`, `.env` và base models nếu đã tải:

```bash
rm -f data/documents/cafef_news.jsonl data/chunks/cafef_news_chunks.json
rm -f data/training/cafef_extractive_qa.jsonl data/training/cafef_planner_workflows.jsonl
rm -f data/metadata/cafef_processing_summary.json
rm -rf data/index/* data/embeddings/* storage_rag/cafef_news
rm -rf models/lfm2_rag_lora models/qwen_ppo
```

2. Cài dependencies và tải lại base models nếu cần:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py --only embedder cross_encoder lfm2_rag qwen
```

3. Xử lý lại raw CafeF:

```bash
python scripts/process_cafef_news.py \
  --input_dir cafef_news_ \
  --storage_dir storage_rag/cafef_news \
  --documents_out data/documents/cafef_news.jsonl \
  --chunks_out data/chunks/cafef_news_chunks.json \
  --training_dir data/training \
  --metadata_dir data/metadata \
  --chunk_size 384 \
  --chunk_overlap_ratio 0.2 \
  --max_training_rows 20000
```

4. Kiểm tra số lượng output:

```bash
wc -l data/documents/cafef_news.jsonl data/training/cafef_extractive_qa.jsonl data/training/cafef_planner_workflows.jsonl
python -c 'import json; print(json.load(open("data/metadata/cafef_processing_summary.json", encoding="utf-8")))'
```

5. Build lại index hoặc dùng chunks trực tiếp. Với chế độ nhẹ, đặt `.env`:

```env
RAG_DOCUMENT_PATH=data/chunks/cafef_news_chunks.json
RAG_RETRIEVAL_MODE=sparse_only
```

Với hybrid/dense index:

```bash
python scripts/build_index.py --data_dir data/documents --local_files_only
```

6. Đo baseline trước training:

```bash
python scripts/evaluate_pipeline.py \
  --qa_file data/training/cafef_extractive_qa.jsonl \
  --chunks_file data/chunks/cafef_news_chunks.json \
  --retrieval_mode sparse_only \
  --output_file data/metadata/evaluation_baseline_sparse.json
```

7. Chuẩn bị `lfm2_rag_sft.jsonl` từ `cafef_extractive_qa.jsonl`, lọc row mà answer không có căn cứ trong context, thêm negative samples `KHÔNG TÌM THẤY`, rồi tách train/valid/test theo `date`.

8. Fine-tune LFM2 bằng LoRA/QLoRA vào `models/lfm2_rag_lora`; không ghi đè `models/lfm2_rag`.

9. Train/PPO Qwen planner từ `data/training/cafef_planner_workflows.jsonl` vào `models/qwen_ppo`.

10. Đánh giá lại bằng cùng QA file, promote model chỉ khi retrieval recall không giảm, LFM2 ít hallucination hơn, Qwen JSON validity tốt hơn và Gemini vẫn tắt trong benchmark.

## Cấu trúc model local

Model được tải về thư mục `models/`:

```text
models/
  qwen/           # Qwen/Qwen2.5-7B-Instruct, dùng cho planner_agent
  lfm2_rag/       # LiquidAI/LFM2-1.2B-RAG, dùng cho executor_agent
  embedder/       # sentence-transformers/all-MiniLM-L6-v2
  cross_encoder/  # cross-encoder/ms-marco-MiniLM-L-6-v2
```

Ý nghĩa các file cấu hình:

- `config/agent_config.yaml`: tham số cho planner, executor, retriever, reranker.
- `config/model_config.yaml`: repo Hugging Face và local folder cho từng model.
- `config/retrieval_config.yaml`: `chunk_size`, overlap, `top_k`, dense/sparse/hybrid weight.
- `.env`: cấu hình runtime ưu tiên cao nhất cho UI và scripts local.

## Google Colab T4

Mở [main.ipynb](main.ipynb) trên Colab, chọn runtime GPU T4 rồi chạy các cell theo thứ tự. Cấu hình khuyến nghị trên T4:

```env
ENABLE_LOCAL_PLANNER=true
ENABLE_LOCAL_EXECUTOR=true
ENABLE_RERANKER=false
LOAD_IN_4BIT=true
QWEN_MODEL_NAME=models/qwen
EXECUTOR_MODEL_NAME=models/lfm2_rag
RAG_RETRIEVAL_MODE=sparse_only
```

Nếu T4 bị thiếu VRAM khi bật cả Qwen và LFM2, tắt planner local trước để kiểm tra executor:

```env
ENABLE_LOCAL_PLANNER=false
ENABLE_LOCAL_EXECUTOR=true
```

Hoặc tắt executor local để kiểm tra planner:

```env
ENABLE_LOCAL_PLANNER=true
ENABLE_LOCAL_EXECUTOR=false
```

## Lệnh vận hành nhanh

```bash
python scripts/download_models.py --only embedder cross_encoder lfm2_rag
python scripts/build_index.py --data_dir data/documents
streamlit run main.py
```

Chạy dry-run PPO cho Qwen planner:

```bash
python scripts/train_qwen_ppo.py --dry_run
```

Chạy PPO thật cần GPU, `models/qwen` đã tải về, và dependency `trl`, `peft`, `torch`:

```bash
python scripts/train_qwen_ppo.py \
  --model_path models/qwen \
  --dataset data/training/cafef_planner_workflows.jsonl \
  --output_dir models/qwen_ppo
```

## Checklist trước khi đánh giá nghiêm túc

- Dependency cài xong và `python -m compileall ...` chạy thành công.
- Đã tải ít nhất `embedder` để build vector index.
- Đã build index từ `data/documents` hoặc cấu hình đ
