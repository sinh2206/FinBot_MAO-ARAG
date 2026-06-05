# Local Models

Thư mục này chứa các model local của dự án.

## Cấu trúc mong đợi

- `models/phi/`: `microsoft/Phi-4-mini-instruct`, dùng làm planner/coordinator.
- `models/qwen/`: `Qwen/Qwen2.5-7B-Instruct`, dùng làm executor chính.
- `models/qwen_executor_lora/`: adapter LoRA sau fine-tune cho Qwen executor.
- `models/embedder/`: `sentence-transformers/all-MiniLM-L6-v2`, dùng cho dense retrieval.
- `models/cross_encoder/`: `cross-encoder/ms-marco-MiniLM-L-6-v2`, dùng cho rerank.

## Ghi chú

- Gemini là executor dự phòng qua API, nên không có thư mục local riêng.
- Không nên commit file trọng số lớn vào git. Thư mục này chỉ nên giữ metadata, README hoặc `.gitkeep`.
