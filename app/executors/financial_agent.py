"""
Financial Agent: fetch realtime market data from vnstock with safe fallback.
"""

import logging
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except Exception as e:  # pragma: no cover - import failure is environment-specific
    Vnstock = None
    VNSTOCK_AVAILABLE = False
    logging.warning(f"vnstock import failed: {e}. FinancialAgent will use mock data.")

from app.mao_core.cost_manager import track_latency

logger = logging.getLogger(__name__)


class FinancialAgent:
    """
    Fetch realtime ticker snapshot (price, volume, PE, PB).
    """

    _TICKER_STOPWORDS = {
        "PHAN", "TICH", "CO", "PHIEU", "GIA", "MA", "CP", "VA", "XEM", "CUA",
        "DU", "DOAN", "HOM", "NAY", "LA", "NAO", "TINH", "HINH", "CHO", "TOI",
    }

    def __init__(self):
        self.preferred_sources = ["VCI", "TCBS"]
        self.source_name = "VnStock API" if VNSTOCK_AVAILABLE else "Mock Data (simulated)"

    @track_latency
    async def get_stock_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return ticker data with source metadata.
        """
        query = str(context.get("query", ""))
        ticker = (context.get("ticker") or self._extract_ticker(query) or "FPT").upper()
        logger.info(f"Fetching financial data for {ticker} from {self.source_name}")

        if VNSTOCK_AVAILABLE:
            for source in self.preferred_sources:
                try:
                    return self._fetch_from_vnstock(ticker=ticker, source=source)
                except Exception as e:
                    logger.warning(
                        "vnstock fetch failed for %s via %s: %s",
                        ticker,
                        source,
                        e,
                    )
            logger.exception(
                "All vnstock sources failed for %s. Falling back to simulated data.",
                ticker,
            )

        return self._get_mock_data(ticker, is_simulated=True)

    def _fetch_from_vnstock(self, ticker: str, source: str) -> Dict[str, Any]:
        stock = Vnstock().stock(symbol=ticker, source=source)

        price, volume = self._fetch_price_volume(stock)
        pe, pb = self._fetch_pe_pb(stock)

        data = {
            "ticker": ticker,
            "price": price,
            "pe": pe,
            "pb": pb,
            "volume": volume,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": f"VnStock API ({source})",
            "is_simulated": False,
        }
        logger.info("Realtime data for %s: %s", ticker, data)
        return data

    def _fetch_price_volume(self, stock: Any) -> Tuple[float, int]:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=14)

        history = stock.quote.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval="1D",
        )
        if history is None or len(history) == 0:
            raise ValueError("Empty price history from vnstock")

        close_col = self._find_column(history, ["close", "giá đóng cửa", "gia dong cua"])
        volume_col = self._find_column(history, ["volume", "kl", "khoi luong"])
        if close_col is None:
            raise ValueError("Cannot find close price column in history")

        row = history.iloc[-1]
        price = self._to_float(row.get(close_col), default=0.0)
        volume = int(self._to_float(row.get(volume_col), default=0.0)) if volume_col else 0
        if price <= 0:
            raise ValueError("Invalid close price from vnstock history")

        return price, volume

    def _fetch_pe_pb(self, stock: Any) -> Tuple[float, float]:
        pe = 0.0
        pb = 0.0

        ratio = stock.finance.ratio(period="year", lang="vi")
        if ratio is None:
            return pe, pb

        # DataFrame path
        if hasattr(ratio, "columns") and len(ratio) > 0:
            row = self._select_latest_ratio_row(ratio)
            pe_key = self._find_column(ratio, ["p/e", "pe"])
            pb_key = self._find_column(ratio, ["p/b", "pb"])

            if pe_key is not None:
                pe = self._to_float(row.get(pe_key), default=0.0)
            if pb_key is not None:
                pb = self._to_float(row.get(pb_key), default=0.0)

            return pe, pb

        # Dict path
        if isinstance(ratio, dict):
            for k, v in ratio.items():
                key = self._normalize_col_name(k)
                if ("p/e" in key or key == "pe") and pe == 0.0:
                    pe = self._to_float(v, default=0.0)
                if ("p/b" in key or key == "pb") and pb == 0.0:
                    pb = self._to_float(v, default=0.0)

        return pe, pb

    def _select_latest_ratio_row(self, ratio_df: Any) -> Any:
        year_col = self._find_column(ratio_df, ["năm", "nam", "year"])
        if year_col is not None:
            try:
                years = ratio_df[year_col]
                numeric = years.astype(str).str.extract(r"(\d{4})", expand=False)
                numeric = numeric.astype(float)
                if numeric.notna().any():
                    idx = numeric.idxmax()
                    return ratio_df.loc[idx]
            except Exception:
                pass

        # Many vnstock tables are descending by year, so prefer first row.
        return ratio_df.iloc[0]

    def _find_column(self, df: Any, tokens: list[str]) -> Optional[Any]:
        if not hasattr(df, "columns"):
            return None

        token_set = [t.lower() for t in tokens]
        for col in df.columns:
            key = self._normalize_col_name(col)
            if any(token in key for token in token_set):
                return col
        return None

    def _normalize_col_name(self, col: Any) -> str:
        if isinstance(col, tuple):
            return " ".join(str(part) for part in col if part is not None).strip().lower()
        return str(col).strip().lower()

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, str):
                value = value.replace(",", "").strip()
                if value == "":
                    return default
            return float(value)
        except Exception:
            return default

    def _get_mock_data(self, ticker: str, is_simulated: bool = True) -> Dict[str, Any]:
        """Simulated data fallback with explicit marker."""
        data = {
            "ticker": ticker,
            "price": round(random.uniform(20, 150), 2),
            "pe": round(random.uniform(5, 25), 2),
            "pb": round(random.uniform(1, 5), 2),
            "volume": random.randint(1_000_000, 10_000_000),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Mock Data (simulated)",
            "is_simulated": is_simulated,
        }
        logger.info("Mock data for %s: %s", ticker, data)
        return data

    def _extract_ticker(self, query: str) -> Optional[str]:
        """Extract probable ticker from user text."""
        if not query:
            return None

        tokens = re.findall(r"[A-Za-z]{2,5}", query.upper())
        candidates = [t for t in tokens if t not in self._TICKER_STOPWORDS]
        if not candidates:
            return None

        three_letter = [t for t in candidates if len(t) == 3]
        if three_letter:
            return three_letter[-1]
        return candidates[-1]
