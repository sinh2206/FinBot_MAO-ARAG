"""
Planner Agent: use a small LLM to build a workflow plan for stock questions.
"""

import logging
import re
import unicodedata
from typing import Any, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

ALLOWED_EXECUTORS = {
    "query_rewriter",
    "retriever",
    "doc_ranker",
    "financial_agent",
    "search_tools",
    "charting_tool",
    "math_coder",
    "generator",
    "fact_checker",
}


class WorkflowPlan(BaseModel):
    """Workflow plan as an ordered list of executor names."""

    steps: List[str] = Field(
        default_factory=lambda: ["generator"],
        description="Ordered list of executor names to run.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_steps(cls, data: Any) -> dict:
        """
        Accept both input shapes:
        - {"steps": [...]} 
        - [...]
        Then sanitize and enforce generator as final step.
        """
        raw_steps: Any
        if isinstance(data, list):
            raw_steps = data
        elif isinstance(data, dict):
            raw_steps = data.get("steps", [])
        else:
            raw_steps = []

        if isinstance(raw_steps, str):
            raw_steps = [raw_steps]
        if not isinstance(raw_steps, list):
            raw_steps = []

        normalized: List[str] = []
        has_fact_checker = False
        for step in raw_steps:
            if not isinstance(step, str):
                continue
            step_name = step.strip().lower()
            if step_name not in ALLOWED_EXECUTORS:
                continue
            if step_name == "generator":
                continue
            if step_name == "fact_checker":
                has_fact_checker = True
                continue
            if step_name not in normalized:
                normalized.append(step_name)

        normalized.append("generator")
        if has_fact_checker:
            normalized.append("fact_checker")
        return {"steps": normalized}


class PlannerAgent:
    """
    PlannerAgent receives a user query and returns a WorkflowPlan.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=WorkflowPlan)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Ban la Planner AI cho bai toan chung khoan.
Muc tieu: chon cac executor can thiet de tra loi cau hoi.
Executor co san:
- query_rewriter
- retriever
- doc_ranker
- financial_agent
- search_tools
- charting_tool
- math_coder
- generator (luon la buoc cuoi)

Tra ve JSON dung schema. Khong them giai thich.
{format_instructions}
""",
                ),
                ("human", "{query}"),
            ]
        )

        self.chain = (
            self.prompt.partial(
                format_instructions=self.parser.get_format_instructions()
            )
            | self.llm
            | self.parser
        )

    async def plan(self, query: str) -> WorkflowPlan:
        logger.info(f"Planning for query: {query}")
        try:
            result = await self.chain.ainvoke({"query": query})
            steps = result.steps
        except Exception as e:
            logger.exception(f"Error during planning: {e}")
            steps = ["generator"]

        steps = self._apply_deterministic_rules(query, steps)
        if "fact_checker" not in steps:
            steps.append("fact_checker")

        logger.info(f"Plan generated: {steps}")
        return WorkflowPlan(steps=steps)

    def _apply_deterministic_rules(self, query: str, steps: List[str]) -> List[str]:
        """Handle noisy chart requests (typo/accent mismatch) with deterministic routing."""
        merged = [s for s in steps if isinstance(s, str)]
        norm = self._normalize_text(query)

        chart_markers = ["bieu do", "do thi", "ve chart", "chart", "candlestick", "nen"]
        stock_markers = ["co phieu", "cophieu", "cau phieu", "cauphieu", "chung khoan"]
        has_day_window = bool(re.search(r"\b\d+\s*(ngay|day)\b", norm))

        wants_chart = any(k in norm for k in chart_markers)
        mentions_stock = any(k in norm for k in stock_markers)

        if wants_chart or (mentions_stock and has_day_window):
            if "retriever" not in merged:
                merged.insert(0, "retriever")
            if "doc_ranker" not in merged:
                retriever_idx = merged.index("retriever")
                merged.insert(retriever_idx + 1, "doc_ranker")
            if "financial_agent" not in merged:
                merged.append("financial_agent")
            if "charting_tool" not in merged:
                merged.append("charting_tool")

        if "generator" not in merged:
            merged.append("generator")
        return merged

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

