"""
Query Rewriter: Phân tích và tách câu hỏi phức tạp thành nhiều câu hỏi con.
"""

import logging
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.mao_core.cost_manager import track_latency, get_current_cost_context

logger = logging.getLogger(__name__)


class DecomposedQueries(BaseModel):
    """Danh sách các câu hỏi con."""
    queries: List[str] = Field(description="Danh sách các câu hỏi đã được tách nhỏ.")


class QueryRewriter:
    """
    Tách câu hỏi phức tạp thành nhiều câu hỏi đơn giản.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=DecomposedQueries)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia phân tích câu hỏi chứng khoán.
Nhiệm vụ: Phân tích câu hỏi của người dùng, nếu nó chứa nhiều ý hoặc so sánh, hãy tách thành các câu hỏi nhỏ hơn, mỗi câu chỉ đề cập một chủ đề/mã chứng khoán.
Ví dụ:
- "So sánh HPG và HSG năm 2024" -> ["Tìm thông tin tài chính của HPG năm 2024", "Tìm thông tin tài chính của HSG năm 2024"]
- "Giá và tin tức của VNM hôm nay" -> ["Giá VNM hôm nay", "Tin tức VNM hôm nay"]
Nếu câu hỏi đã đơn giản, chỉ trả về một phần tử chính là câu hỏi gốc.
Chỉ trả về JSON theo schema.
"""),
            ("human", "{query}")
        ])

        self.chain = self.prompt | self.llm | self.parser

    @track_latency
    async def rewrite(self, query: str) -> List[str]:
        """
        Nhận câu hỏi và trả về danh sách câu hỏi con.
        """
        logger.info(f"Rewriting query: {query}")
        try:
            result = await self.chain.ainvoke({"query": query})
            queries = result.queries
            logger.info(f"Decomposed into {len(queries)} queries: {queries}")
            return queries
        except Exception as e:
            logger.exception(f"Error rewriting query: {e}")
            # Fallback: trả về chính câu hỏi gốc
            return [query]