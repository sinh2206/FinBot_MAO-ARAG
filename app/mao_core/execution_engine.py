"""
Execution Engine: Điều phối việc thực thi các executor theo kế hoạch.
Chạy song song các executor không phụ thuộc.
"""

import asyncio
import logging
from typing import Dict, Any, Callable, Awaitable

from app.mao_core.cost_manager import CostManager, track_latency
from app.mao_core.planner_agent import WorkflowPlan

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Nhạc trưởng thực thi workflow.
    """

    def __init__(self):
        self.executors: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self.cost_manager = CostManager()

    def register_executor(self, name: str, func: Callable[..., Awaitable[Any]]):
        self.executors[name] = func
        logger.info(f"Registered executor: {name}")

    @track_latency
    async def execute_step(self, step_name: str, context: Dict[str, Any]) -> Any:
        if step_name not in self.executors:
            raise ValueError(f"Executor '{step_name}' not registered.")
        executor = self.executors[step_name]
        # Ho tro executor co chu ky tham so khac nhau.
        if step_name == "query_rewriter":
            result = await executor(context.get("query", ""))
        elif step_name == "doc_ranker":
            result = await executor(context.get("query", ""), context.get("retriever", []))
        else:
            result = await executor(context)
        return result

    async def execute_plan(self, plan: WorkflowPlan, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        context = initial_context.copy()
        steps = plan.steps

        logger.info(f"Starting execution of plan: {steps}")

        for step in steps:
            logger.info(f"Executing step: {step}")
            try:
                # Nếu step là fact_checker, chuẩn bị context đặc biệt
                if step == "fact_checker":
                    all_sources = []
                    # Lấy kết quả từ retriever
                    if "retriever" in context and isinstance(context["retriever"], list):
                        all_sources.extend(context["retriever"])
                    # Lấy kết quả từ financial_agent và chuyển thành document
                    if "financial_agent" in context and isinstance(context["financial_agent"], dict):
                        fa = context["financial_agent"]
                        doc = {
                            "content": f"{fa['ticker']}: Giá {fa['price']}, P/E {fa['pe']}, P/B {fa['pb']}, KLGD {fa['volume']}",
                            "metadata": {
                                "filename": fa.get("source", "Unknown"),
                                "page_number": None,
                                "is_simulated": fa.get("is_simulated", False)
                            }
                        }
                        all_sources.append(doc)
                    # Lấy kết quả từ search_tools
                    if "search_tools" in context and isinstance(context["search_tools"], list):
                        for news in context["search_tools"]:
                            doc = {
                                "content": f"Tiêu đề: {news.get('title')}\n{news.get('snippet')}",
                                "metadata": {
                                    "filename": news.get("link", "Unknown"),
                                    "page_number": None,
                                    "is_simulated": False  # Giả sử tin tức là thật
                                }
                            }
                            all_sources.append(doc)
                    # Lưu vào context để fact_checker dùng
                    context["fact_checker_context"] = all_sources

                result = await self.execute_step(step, context)
                context[step] = result
                logger.info(f"Step {step} completed.")
            except Exception as e:
                logger.exception(f"Step {step} failed: {e}")
                context[f"{step}_error"] = str(e)

        logger.info("Execution finished.")
        return context
