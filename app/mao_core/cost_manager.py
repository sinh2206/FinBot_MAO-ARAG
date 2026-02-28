"""
Cost Manager: Theo dõi token, latency, chi phí cho mỗi request.
Cung cấp decorator để tự động đo lường.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
from contextvars import ContextVar

# Context variable để lưu thông tin cost của request hiện tại
current_cost_ctx: ContextVar[Dict] = ContextVar("current_cost", default={})

logger = logging.getLogger(__name__)


class CostManager:
    """
    Quản lý chi phí và thời gian cho mỗi request.
    Sử dụng như một singleton hoặc instance dùng chung.
    """

    def __init__(self):
        self.total_cost = 0.0
        # Pricing: gemini-flash: $0.075 / 1M input tokens, $0.3 / 1M output tokens (giá ước lượng)
        # gemini-pro: $0.5 / 1M input, $1.5 / 1M output
        self.price_per_million_input = {"flash": 0.075, "pro": 0.5}
        self.price_per_million_output = {"flash": 0.3, "pro": 1.5}

    def track_request(self, model_type: str, input_tokens: int, output_tokens: int, latency: float):
        """Ghi nhận một request và cập nhật tổng chi phí."""
        cost_input = input_tokens / 1_000_000 * self.price_per_million_input.get(model_type, 0)
        cost_output = output_tokens / 1_000_000 * self.price_per_million_output.get(model_type, 0)
        cost = cost_input + cost_output
        self.total_cost += cost

        # Lưu vào context hiện tại nếu có
        ctx = current_cost_ctx.get({})
        ctx.setdefault("requests", []).append({
            "model": model_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": latency,
            "cost": cost
        })
        ctx["total_cost"] = ctx.get("total_cost", 0) + cost
        ctx["total_latency"] = ctx.get("total_latency", 0) + latency
        current_cost_ctx.set(ctx)

        logger.debug(f"Tracked request: {model_type}, in={input_tokens}, out={output_tokens}, cost=${cost:.6f}")

    def reset(self):
        """Reset tổng chi phí (dùng cho testing)."""
        self.total_cost = 0.0


def track_latency(func: Callable) -> Callable:
    """
    Decorator đo latency của một hàm async và ghi nhận qua CostManager.
    Giả sử hàm trả về object có chứa thông tin token (hoặc ta tự truyền).
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            latency = time.perf_counter() - start
            # Ghi latency vào context (có thể không cần token)
            ctx = current_cost_ctx.get({})
            ctx.setdefault("latencies", []).append({"func": func.__name__, "latency": latency})
            current_cost_ctx.set(ctx)
            logger.debug(f"Latency for {func.__name__}: {latency:.3f}s")
    return wrapper


def get_current_cost_context() -> Dict:
    """Lấy context cost hiện tại."""
    return current_cost_ctx.get({})


def reset_cost_context():
    """Reset context cost."""
    current_cost_ctx.set({})