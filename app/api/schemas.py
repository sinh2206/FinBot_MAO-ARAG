"""
Pydantic schemas cho API requests/responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    """Yêu cầu chat từ người dùng."""
    query: str = Field(..., min_length=1, max_length=1000, description="Câu hỏi của người dùng")
    user_id: Optional[str] = Field("anonymous", description="ID người dùng (mặc định anonymous)")
    session_id: Optional[str] = Field(None, description="ID phiên (nếu có)")

    @field_validator('query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query không được để trống")
        return v.strip()


class ChatResponse(BaseModel):
    """Phản hồi chat."""
    answer: str = Field(..., description="Câu trả lời dạng Markdown")
    cost: float = Field(0.0, description="Chi phí USD của request")
    workflow: List[str] = Field(default_factory=list, description="Các bước workflow đã thực thi")
    latency: float = Field(0.0, description="Tổng thời gian xử lý (giây)")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Số token sử dụng")


class ChatHistoryItem(BaseModel):
    """Một mục trong lịch sử chat."""
    id: int
    user_id: str
    query: str
    response: str
    tokens: Optional[int] = None
    cost: float
    timestamp: str  # ISO format


class ChatHistoryResponse(BaseModel):
    """Danh sách lịch sử chat."""
    history: List[ChatHistoryItem]