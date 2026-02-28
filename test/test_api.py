"""
Unit tests cho API endpoints, sử dụng mock để không gọi Gemini thật.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from main import app  # giả sử main.py tạo FastAPI app và include router

client = TestClient(app)


@pytest.fixture
def mock_planner():
    """Mock PlannerAgent trả về workflow cố định."""
    with patch('app.api.routes.planner') as mock:
        mock.plan = AsyncMock(return_value=MagicMock(steps=["financial_agent", "generator"]))
        yield mock


@pytest.fixture
def mock_execution_engine():
    """Mock ExecutionEngine trả về context có generator."""
    with patch('app.api.routes.execution_engine') as mock:
        mock.execute_plan = AsyncMock(return_value={"generator": "Giá HPG hôm nay là 27,500 VNĐ."})
        yield mock


@pytest.fixture
def mock_chat_db():
    """Mock ChatHistoryDB để không ghi DB thật."""
    with patch('app.api.routes.chat_db') as mock:
        mock.save_chat = MagicMock()
        yield mock


def test_chat_endpoint_success(mock_planner, mock_execution_engine, mock_chat_db):
    """Test /chat trả về response đúng."""
    response = client.post("/api/chat", json={"query": "Giá HPG hôm nay?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Giá HPG hôm nay là 27,500 VNĐ."
    assert data["workflow"] == ["financial_agent", "generator"]
    assert data["cost"] >= 0


def test_chat_endpoint_empty_query():
    """Test query rỗng bị lỗi validation."""
    response = client.post("/api/chat", json={"query": ""})
    assert response.status_code == 422  # Unprocessable Entity


def test_history_endpoint(mock_chat_db):
    """Test lấy lịch sử chat."""
    mock_chat_db.get_history = MagicMock(return_value=[
        {"id": 1, "user_id": "test", "query": "test", "response": "test",
         "tokens_input": 10, "tokens_output": 20, "cost": 0.001, "workflow": "[]", "latency": 0.5, "timestamp": "2025-01-01"}
    ])
    response = client.get("/api/history/test")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert len(data["history"]) == 1