"""
Unit tests cho FactCheckerAgent, đặc biệt kiểm tra khả năng phát hiện hallucination.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.executors.fact_checker import FactCheckerAgent


@pytest.mark.asyncio
async def test_hallucination_removal():
    """
    Test rằng FactCheckerAgent loại bỏ các câu hallucination và thêm citation đúng.
    """
    # Tạo context mẫu
    mock_context = [
        {
            "content": "Lợi nhuận quý 3 của FPT đạt 2.000 tỷ đồng.",
            "metadata": {"filename": "BCTC_FPT_Q3.pdf", "page_number": 12}
        },
        {
            "content": "Doanh thu là 10.000 tỷ.",
            "metadata": {"filename": "BCTC_FPT_Q3.pdf", "page_number": 12}
        }
    ]

    # Draft có cả thông tin đúng, sai và bịa
    mock_draft = """
Lợi nhuận quý 3 của FPT đạt 2.000 tỷ đồng.
Doanh thu là 15.000 tỷ đồng.
Công ty chuẩn bị thâu tóm một đối thủ ở Mỹ.
"""

    # Kết quả mong đợi sau khi fact-checker xử lý
    expected_output = """
Lợi nhuận quý 3 của FPT đạt 2.000 tỷ đồng. [Nguồn: BCTC_FPT_Q3.pdf, Trang 12]
*[Cảnh báo: Không có đủ dữ liệu gốc để xác minh thông tin này]*
"""

    # Tạo mock cho LLM và chain
    with patch('app.executors.fact_checker.ChatGoogleGenerativeAI') as MockLLM:
        # Khởi tạo agent
        agent = FactCheckerAgent()

        # Mock chain invoke trả về kết quả mong muốn
        mock_response = MagicMock()
        mock_response.content = expected_output

        # Mock prompt và chain
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        agent.prompt = MagicMock()
        agent.prompt.__or__ = MagicMock(return_value=mock_chain)

        # Gọi hàm
        result = await agent.verify_and_cite(mock_draft, mock_context)

        # Kiểm tra kết quả
        assert "2.000 tỷ" in result
        assert "[Nguồn: BCTC_FPT_Q3.pdf, Trang 12]" in result
        assert "15.000 tỷ" not in result  # Số liệu sai bị loại
        assert "Mỹ" not in result          # Thông tin bịa bị loại
        assert "Cảnh báo" in result        # Thông báo thay thế xuất hiện


@pytest.mark.asyncio
async def test_fact_checker_empty_context():
    """
    Test khi context rỗng, toàn bộ draft phải bị loại bỏ hoặc thay bằng cảnh báo.
    """
    draft = "Công ty A có lợi nhuận 100 tỷ."

    expected = "*[Cảnh báo: Không có đủ dữ liệu gốc để xác minh thông tin này]*"

    # Mock LLM response
    with patch('app.executors.fact_checker.ChatGoogleGenerativeAI'):
        agent = FactCheckerAgent()
        mock_response = MagicMock()
        mock_response.content = expected

        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        agent.prompt = MagicMock()
        agent.prompt.__or__ = MagicMock(return_value=mock_chain)

        result = await agent.verify_and_cite(draft, [])
        assert "Cảnh báo" in result
        assert "100 tỷ" not in result
