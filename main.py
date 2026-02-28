"""
main.py - Điểm khởi chạy FastAPI server.
- Cấu hình CORS
- Mount static files (frontend và charts)
- Include router từ app.api.routes
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes import router as api_router

# Khởi tạo app
app = FastAPI(title="MAO-ARAG")

# Cấu hình CORS - cho phép tất cả các origin khi dev (có thể giới hạn sau)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router API (các endpoint bắt đầu bằng /api)
app.include_router(api_router, prefix="/api")

# Mount thư mục frontend (chứa index.html, CSS, JS) tại đường dẫn gốc "/"
# FastAPI sẽ tự động tìm file index.html trong thư mục này
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
else:
    # Fallback: in ra warning (khi chạy trong container có thể frontend đã được copy)
    import logging
    logging.warning("Thư mục frontend không tồn tại, static files sẽ không được serve.")

# Mount thư mục storage/charts để phục vụ ảnh biểu đồ qua đường dẫn /charts
charts_path = Path(__file__).parent / "storage" / "charts"
charts_path.mkdir(parents=True, exist_ok=True)  # Tạo nếu chưa có
app.mount("/charts", StaticFiles(directory=str(charts_path)), name="charts")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)