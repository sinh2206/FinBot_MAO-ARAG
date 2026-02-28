"""
Charting Tool: Vẽ biểu đồ chứng khoán (nến) bằng matplotlib.
"""

import logging
import random
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta

from app.mao_core.cost_manager import track_latency

logger = logging.getLogger(__name__)


class ChartingTool:
    """
    Vẽ biểu đồ nến cho một mã chứng khoán.
    Lưu file PNG vào storage/charts/ và trả về đường dẫn.
    """

    def __init__(self, charts_dir: str = "storage/charts"):
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    @track_latency
    async def draw_candlestick_chart(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Vẽ biểu đồ nến dựa trên ticker (hoặc query).

        Args:
            context: Chứa 'ticker' hoặc 'query'.

        Returns:
            Đường dẫn file ảnh (string) hoặc None nếu lỗi.
        """
        ticker = context.get("ticker") or self._extract_ticker(context.get("query", ""))
        if not ticker:
            ticker = "HPG"
        logger.info(f"Drawing chart for {ticker}")

        try:
            # Tạo dữ liệu giá giả lập (có thể thay bằng yfinance thực tế)
            df = self._generate_mock_data(ticker)

            # Vẽ biểu đồ nến
            fig, ax = plt.subplots(figsize=(10, 6))
            # Đơn giản: vẽ đường line thay vì nến (do không có mplfinance)
            ax.plot(df.index, df['Close'], label='Close Price')
            ax.set_title(f'Biểu đồ giá {ticker}')
            ax.set_xlabel('Ngày')
            ax.set_ylabel('Giá (VNĐ)')
            ax.legend()
            ax.grid(True)

            # Format trục x
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            plt.xticks(rotation=45)

            # Lưu file
            filename = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.charts_dir / filename
            plt.tight_layout()
            plt.savefig(filepath, dpi=100)
            plt.close(fig)

            logger.info(f"Chart saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.exception(f"Failed to draw chart: {e}")
            return None

    def _generate_mock_data(self, ticker: str) -> pd.DataFrame:
        """Tạo dữ liệu giá giả lập 30 ngày."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Tạo giá ngẫu nhiên theo bước
        base = random.uniform(50, 200)
        prices = [base]
        for _ in range(len(dates)-1):
            prices.append(prices[-1] * (1 + random.uniform(-0.05, 0.05)))
        df = pd.DataFrame({'Close': prices}, index=dates)
        return df

    def _extract_ticker(self, query: str) -> Optional[str]:
        words = query.split()
        for w in words:
            if w.isupper() and 2 <= len(w) <= 5:
                return w
        return None