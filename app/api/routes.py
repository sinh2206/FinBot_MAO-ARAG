from fastapi import APIRouter, HTTPException
from app.api.schemas import ChatRequest, ChatResponse, ChatHistoryResponse
from app.memory.sqlite_db import ChatHistoryDB
from app.mao_core.planner_agent import PlannerAgent
from app.mao_core.execution_engine import ExecutionEngine
from app.mao_core.cost_manager import get_current_cost_context, reset_cost_context
from app.executors.generator import GeneratorAgent
from app.executors.query_rewriter import QueryRewriter
from app.executors.retriever import Retriever
from app.executors.doc_ranker import DocumentRanker
from app.executors.financial_agent import FinancialAgent
from app.tools.search_tools import SearchTools
from app.tools.charting_tool import ChartingTool
from app.executors.math_coder import MathCoder
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Khởi tạo các thành phần
planner = PlannerAgent()
execution_engine = ExecutionEngine()
chat_db = ChatHistoryDB()

# Đăng ký executors
execution_engine.register_executor("query_rewriter", QueryRewriter().rewrite)
execution_engine.register_executor("retriever", Retriever().retrieve)
execution_engine.register_executor("doc_ranker", DocumentRanker().rank)
execution_engine.register_executor("financial_agent", FinancialAgent().get_stock_data)
execution_engine.register_executor("search_tools", SearchTools().search_news)
execution_engine.register_executor("charting_tool", ChartingTool().draw_candlestick_chart)
execution_engine.register_executor("math_coder", MathCoder().execute_math)
execution_engine.register_executor("generator", GeneratorAgent().generate_report)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    reset_cost_context()
    try:
        plan = await planner.plan(request.query)
        context = {"query": request.query, "user_id": request.user_id}
        result_context = await execution_engine.execute_plan(plan, context)
        answer = result_context.get("generator", "Không thể tạo câu trả lời.")
        cost_ctx = get_current_cost_context()
        total_cost = cost_ctx.get("total_cost", 0.0)
        total_latency = cost_ctx.get("total_latency", 0.0)
        token_usage = {
            "input": cost_ctx.get("input_tokens", 0),
            "output": cost_ctx.get("output_tokens", 0)
        }
        chat_db.save_chat(
            user_id=request.user_id,
            query=request.query,
            response=answer,
            tokens_input=token_usage.get("input", 0),
            tokens_output=token_usage.get("output", 0),
            cost=total_cost,
            workflow=plan.steps,
            latency=total_latency,
            session_id=request.session_id
        )
        return ChatResponse(
            answer=answer,
            cost=total_cost,
            workflow=plan.steps,
            latency=total_latency,
            token_usage=token_usage
        )
    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history/{user_id}", response_model=ChatHistoryResponse)
async def get_history(user_id: str, limit: int = 50):
    try:
        history = chat_db.get_history(user_id, limit)
        return ChatHistoryResponse(history=history)
    except Exception as e:
        logger.exception("Error fetching history")
        raise HTTPException(status_code=500, detail="Internal server error")