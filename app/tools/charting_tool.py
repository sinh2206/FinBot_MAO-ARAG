import logging
import random
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from app.mao_core.cost_manager import track_latency

try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except Exception:
    Vnstock = None
    VNSTOCK_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChartingTool:
    """Draw a price chart for a stock ticker."""

    _STOPWORDS = {
        "VE", "BIEU", "DO", "DO THI", "CO", "PHIEU", "CAU", "GIA", "NGAY", "QUA",
        "HOM", "NAY", "TUAN", "THANG", "NAM", "CHUNG", "KHOAN",
    }

    def __init__(self, charts_dir: str = "storage/charts"):
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.preferred_sources = ["VCI", "TCBS"]

    @track_latency
    async def draw_candlestick_chart(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Draw a chart using trained data if available, else vnstock realtime, else mock.
        """
        query = str(context.get("query", ""))
        ticker = (context.get("ticker") or self._extract_ticker(query) or "FPT").upper()
        days = self._extract_days(query) or 30
        logger.info("Drawing chart for %s over %s days", ticker, days)

        df: Optional[pd.DataFrame] = None
        source_label = ""

        # 1) Try trained RAG data first.
        df = self._build_df_from_trained_data(context=context, ticker=ticker, days=days)
        if df is not None and not df.empty:
            source_label = "Trained RAG data"
        else:
            # 2) Fallback to vnstock realtime.
            df, source_label = self._build_df_from_vnstock(ticker=ticker, days=days)

        # 3) Final fallback mock for system resilience.
        if df is None or df.empty:
            df = self._generate_mock_data(ticker=ticker, days=days)
            source_label = "Mock Data (simulated)"

        return self._plot_and_save(df=df, ticker=ticker, days=days, source_label=source_label)

    def _build_df_from_trained_data(self, context: Dict[str, Any], ticker: str, days: int) -> Optional[pd.DataFrame]:
        docs = []
        for key in ("doc_ranker", "retriever"):
            items = context.get(key)
            if isinstance(items, list) and items:
                docs.extend(items)

        if not docs:
            return None

        points: List[Tuple[datetime, float]] = []
        for item in docs:
            content = ""
            if isinstance(item, dict):
                content = str(item.get("content", ""))
            elif isinstance(item, str):
                content = item
            if not content:
                continue

            points.extend(self._extract_points_from_text(content, ticker))

        if not points:
            return None

        points.sort(key=lambda x: x[0])
        dedup: Dict[datetime, float] = {}
        for d, p in points:
            dedup[d] = p

        rows = [{"Date": d, "Close": p} for d, p in dedup.items()]
        df = pd.DataFrame(rows)
        if df.empty:
            return None

        cutoff = datetime.now() - timedelta(days=days * 2)
        df = df[df["Date"] >= cutoff]
        if df.empty:
            return None

        df = df.sort_values("Date").tail(days)
        return df.set_index("Date")

    def _extract_points_from_text(self, text: str, ticker: str) -> List[Tuple[datetime, float]]:
        # Expected patterns in trained docs:
        # - 2026-03-04 ... 84.9
        # - 04/03/2026 ... 84.9
        points: List[Tuple[datetime, float]] = []

        # YYYY-MM-DD ... price
        p1 = re.findall(r"(\d{4}-\d{2}-\d{2}).{0,40}?([0-9]{1,3}(?:\.[0-9]{1,3})?)", text)
        for date_str, price_str in p1:
            dt = self._parse_date(date_str)
            price = self._to_float(price_str)
            if dt and price > 0:
                points.append((dt, price))

        # DD/MM/YYYY ... price
        p2 = re.findall(r"(\d{2}/\d{2}/\d{4}).{0,40}?([0-9]{1,3}(?:\.[0-9]{1,3})?)", text)
        for date_str, price_str in p2:
            dt = self._parse_date(date_str)
            price = self._to_float(price_str)
            if dt and price > 0:
                points.append((dt, price))

        return points

    def _build_df_from_vnstock(self, ticker: str, days: int) -> Tuple[Optional[pd.DataFrame], str]:
        if not VNSTOCK_AVAILABLE:
            return None, ""

        end_date = datetime.now().date()
        # Fetch wider window to skip weekends and keep enough points.
        start_date = end_date - timedelta(days=max(days * 3, 10))

        for source in self.preferred_sources:
            try:
                stock = Vnstock().stock(symbol=ticker, source=source)
                hist = stock.quote.history(
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    interval="1D",
                )
                if hist is None or len(hist) == 0:
                    continue

                time_col = self._find_column(hist, ["time", "date", "ngay"])
                close_col = self._find_column(hist, ["close", "dong cua", "gia dong cua"])
                if close_col is None:
                    continue

                if time_col is not None:
                    dates = pd.to_datetime(hist[time_col], errors="coerce")
                else:
                    dates = pd.to_datetime(hist.index, errors="coerce")

                closes = pd.to_numeric(hist[close_col], errors="coerce")
                df = pd.DataFrame({"Date": dates, "Close": closes}).dropna()
                if df.empty:
                    continue

                df = df.sort_values("Date").tail(days)
                if df.empty:
                    continue

                return df.set_index("Date"), f"VnStock API ({source})"
            except Exception as e:
                logger.warning("vnstock chart fetch failed for %s via %s: %s", ticker, source, e)

        return None, ""

    def _plot_and_save(self, df: pd.DataFrame, ticker: str, days: int, source_label: str) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df["Close"], label="Close", linewidth=2)
            ax.set_title(f"{ticker} - {days} ngay qua")
            ax.set_xlabel("Ngay")
            ax.set_ylabel("Gia")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")

            if source_label:
                ax.text(
                    0.01,
                    0.01,
                    f"Nguon: {source_label}",
                    transform=ax.transAxes,
                    fontsize=9,
                    alpha=0.8,
                    bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none"},
                )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            plt.xticks(rotation=45)

            filename = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.charts_dir / filename
            plt.tight_layout()
            plt.savefig(filepath, dpi=120)
            plt.close(fig)
            logger.info("Chart saved to %s", filepath)
            return str(filepath)
        except Exception as e:
            logger.exception("Failed to draw chart for %s: %s", ticker, e)
            return None

    def _generate_mock_data(self, ticker: str, days: int = 30) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=max(days, 5))
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        base = random.uniform(50, 200)
        prices = [base]
        for _ in range(len(dates) - 1):
            prices.append(prices[-1] * (1 + random.uniform(-0.04, 0.04)))

        df = pd.DataFrame({"Close": prices}, index=dates)
        return df.tail(days)

    def _extract_days(self, query: str) -> Optional[int]:
        norm = self._normalize_text(query)
        m = re.search(r"\b(\d{1,3})\s*(ngay|day|d)\b", norm)
        if m:
            value = int(m.group(1))
            return min(max(value, 1), 365)
        if "1 tuan" in norm or "mot tuan" in norm or "7 ngay" in norm:
            return 7
        if "1 thang" in norm or "mot thang" in norm:
            return 30
        return None

    def _extract_ticker(self, query: str) -> Optional[str]:
        if not query:
            return None
        tokens = re.findall(r"[A-Za-z]{2,5}", query.upper())
        candidates = [t for t in tokens if t not in self._STOPWORDS]
        if not candidates:
            return None
        three = [t for t in candidates if len(t) == 3]
        return three[-1] if three else candidates[-1]

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _find_column(self, df: Any, tokens: List[str]) -> Optional[Any]:
        if not hasattr(df, "columns"):
            return None

        low_tokens = [self._normalize_text(t) for t in tokens]
        for col in df.columns:
            key = self._normalize_text(self._col_to_text(col))
            if any(t in key for t in low_tokens):
                return col
        return None

    def _col_to_text(self, col: Any) -> str:
        if isinstance(col, tuple):
            return " ".join(str(p) for p in col if p is not None)
        return str(col)

    def _parse_date(self, value: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _to_float(self, value: Any) -> float:
        try:
            if value is None:
                return 0.0
            if isinstance(value, str):
                value = value.replace(",", "").strip()
            return float(value)
        except Exception:
            return 0.0

