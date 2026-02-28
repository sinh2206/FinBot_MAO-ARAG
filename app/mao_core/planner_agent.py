"""
Planner Agent: use a small LLM to build a workflow plan for stock questions.
"""

import logging
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
        for step in raw_steps:
            if not isinstance(step, str):
                continue
            step_name = step.strip().lower()
            if step_name not in ALLOWED_EXECUTORS:
                continue
            if step_name == "generator":
                continue
            if step_name not in normalized:
                normalized.append(step_name)

        normalized.append("generator")
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

        self.chain = self.prompt | self.llm | self.parser

    async def plan(self, query: str) -> WorkflowPlan:
        """Build execution plan for a user query."""
        logger.info("Planning for query: %s", query)
        try:
            result = await self.chain.ainvoke(
                {
                    "query": query,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
            logger.info("Plan generated: %s", result.steps)
            return result
        except Exception as exc:
            logger.exception("Error during planning: %s", exc)
            return WorkflowPlan(steps=["generator"])