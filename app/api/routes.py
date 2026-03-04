from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional
import logging
import time

from app.mao_core.planner_agent import PlannerAgent
from app.mao_core.execution_engine import ExecutionEngine
from app.mao_core.cost_manager import get_current_cost_context, reset_cost_context
from app.memory.sqlite_db import ChatHistoryDB
from app.executors.generator import GeneratorAgent
from app.executors.query_rewriter import QueryRewriter
from app.executors.retriever import Retriever
from app.executors.doc_ranker import DocumentRanker
from app.executors.financial_agent import FinancialAgent
from app.tools.search_tools import SearchTools
from app.tools.charting_tool import ChartingTool
from app.executors.math_coder import MathCoder

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def normalize_query(self):
        # Support both old payload (`message`) and new payload (`query`).
        if self.query and self.query.strip():
            self.query = self.query.strip()
            return self
        if self.message and self.message.strip():
            self.query = self.message.strip()
            return self
        raise ValueError("Missing query/message in request body")


class ChatResponse(BaseModel):
    answer: str
    response: str
    chart_path: Optional[str] = None
    cost: float = 0.0
    workflow: List[str] = Field(default_factory=list)
    latency: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)


class HistoryItem(BaseModel):
    id: int
    user_id: str
    query: str
    response: str
    cost: float
    timestamp: str


planner = PlannerAgent()
execution_engine = ExecutionEngine()
chat_db = ChatHistoryDB()

execution_engine.register_executor("query_rewriter", QueryRewriter().rewrite)
execution_engine.register_executor("retriever", Retriever().retrieve)
execution_engine.register_executor("doc_ranker", DocumentRanker().rank)
execution_engine.register_executor("financial_agent", FinancialAgent().get_stock_data)
execution_engine.register_executor("search_tools", SearchTools().search_news)
execution_engine.register_executor("charting_tool", ChartingTool().draw_candlestick_chart)
execution_engine.register_executor("math_coder", MathCoder().execute_math)
execution_engine.register_executor("generator", GeneratorAgent().generate_report)

DEFAULT_USER_ID = "anonymous"


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Receive a query and return answer + optional chart path.
    """
    reset_cost_context()
    start_time = time.time()

    try:
        user_id = (request.user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        query = request.query or ""

        plan = await planner.plan(query)
        logger.info(f"Plan: {plan.steps}")

        context = {"query": query, "user_id": user_id}
        result_context = await execution_engine.execute_plan(plan, context)

        answer = result_context.get("generator", "Xin loi, toi chua the tra loi cau hoi nay.")

        chart_path = None
        if "charting_tool" in result_context:
            chart_path = result_context["charting_tool"]
            if chart_path and isinstance(chart_path, str):
                # Normalize Windows/Linux paths to public URL under /storage.
                normalized = chart_path.replace("\\", "/")
                if normalized.startswith("/storage/"):
                    chart_path = normalized
                elif "storage/" in normalized:
                    relative = normalized.split("storage/", 1)[1].lstrip("/")
                    chart_path = f"/storage/{relative}"
                else:
                    chart_path = f"/storage/{normalized.lstrip('/')}"

        cost_ctx = get_current_cost_context()
        total_cost = float(cost_ctx.get("total_cost", 0.0) or 0.0)
        latency = time.time() - start_time

        chat_db.save_chat(
            user_id=user_id,
            query=query,
            response=answer,
            cost=total_cost,
            latency=latency,
            workflow=plan.steps,
            tokens_input=int(cost_ctx.get("input_tokens", 0) or 0),
            tokens_output=int(cost_ctx.get("output_tokens", 0) or 0),
            session_id=request.session_id,
        )

        token_usage = {
            "input_tokens": int(cost_ctx.get("input_tokens", 0) or 0),
            "output_tokens": int(cost_ctx.get("output_tokens", 0) or 0),
        }
        return ChatResponse(
            answer=answer,
            response=answer,
            chart_path=chart_path,
            cost=total_cost,
            workflow=plan.steps or [],
            latency=latency,
            token_usage=token_usage,
        )

    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
@router.get("/history/{user_id}")
async def get_history(user_id: str = DEFAULT_USER_ID):
    """
    Return chat history for provided user_id (or default user).
    """
    try:
        history = chat_db.get_history(user_id, limit=50)
        items: List[HistoryItem] = []
        for chat in history:
            items.append(
                HistoryItem(
                    id=int(chat["id"]),
                    user_id=str(chat["user_id"]),
                    query=str(chat["query"]),
                    response=str(chat["response"]),
                    cost=float(chat.get("cost") or 0.0),
                    timestamp=str(chat["timestamp"]),
                )
            )

        return {"history": items}
    except Exception as e:
        logger.exception("History error")
        raise HTTPException(status_code=500, detail=str(e))
