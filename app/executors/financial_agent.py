"""
Financial Agent: Lấy dữ liệu thị trường realtime (giá, PE, PB) từ API giả lập.
"""

import logging
import random
from typing import Dict, Any

from app.mao_core.cost_manager import track_latency

logger = logging.getLogger(__name__)


class FinancialAgent:
    """
    Lấy thông tin tài chính realtime cho một mã chứng khoán.
    (Giả lập dữ liệu)
    """

    @track_latency
    async def get_stock_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trả về dict chứa thông tin giá, PE, PB.

        Args:
            context: Chứa 'query' hoặc 'ticker' (có thể trích xuất từ query).
        """
        query = context.get("query", "")
        # Trích xuất ticker đơn giản (giả sử query có mã)
        ticker = self._extract_ticker(query) or "HPG"
        logger.info(f"Fetching financial data for {ticker}")

        # Giả lập dữ liệu
        data = {
            "ticker": ticker,
            "price": round(random.uniform(20, 150), 2),
            "pe": round(random.uniform(5, 25), 2),
            "pb": round(random.uniform(1, 5), 2),
            "volume": random.randint(1000000, 10000000),
            "time": "2025-02-27 10:30:00"  # Giả định
        }
        logger.info(f"Financial data: {data}")
        return data

    def _extract_ticker(self, query: str) -> str | None:
        """Trích xuất mã cổ phiếu từ query (đơn giản)."""
        words = query.split()
        for w in words:
            if w.isupper() and 2 <= len(w) <= 5:
                return w
        return None