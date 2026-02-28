"""
Generator Agent: Tổng hợp thông tin từ các nguồn và viết báo cáo Markdown.
"""

import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.mao_core.cost_manager import track_latency

logger = logging.getLogger(__name__)


class GeneratorAgent:
    """
    Sinh câu trả lời cuối cùng dạng Markdown, chèn bảng và ảnh nếu có.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia phân tích chứng khoán, có nhiệm vụ tổng hợp thông tin từ nhiều nguồn để trả lời câu hỏi của người dùng.
Bạn sẽ nhận được context chứa các kết quả từ các công cụ: tài liệu RAG, dữ liệu tài chính, tin tức, đường dẫn biểu đồ.
Hãy viết câu trả lời bằng Markdown, có cấu trúc rõ ràng.
Yêu cầu:
- Nếu có dữ liệu tài chính (giá, PE, PB), hãy trình bày dưới dạng bảng.
- Nếu có đường dẫn ảnh biểu đồ, hãy chèn vào với cú pháp ![alt](path).
- Trích dẫn nguồn tin tức nếu có.
- Nếu có thông tin từ tài liệu RAG, hãy sử dụng để phân tích.

Hãy trả lời bằng tiếng Việt, lịch sự, chuyên nghiệp.
"""),
            ("human", """
Câu hỏi: {query}

Context:
{context}

Hãy viết câu trả lời hoàn chỉnh.
""")
        ])

    @track_latency
    async def generate_report(self, query: str, context: Dict[str, Any]) -> str:
        """
        Tạo báo cáo Markdown.

        Args:
            query: Câu hỏi gốc.
            context: Toàn bộ kết quả từ các executor (đã lưu trong execution context).

        Returns:
            Chuỗi Markdown.
        """
        query = context.get("query", "")
        logger.info(f"Generating report for query: {query}")

        # Format context thành chuỗi để nhét vào prompt
        context_str = self._format_context(context)

        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({"query": query, "context": context_str})
            report = response.content
            logger.info("Report generated successfully.")
            return report
        except Exception as e:
            logger.exception(f"Error generating report: {e}")
            return "Xin lỗi, đã có lỗi khi tạo báo cáo."

    def _format_context(self, context: Dict) -> str:
        """Chuyển context thành text mô tả cho LLM."""
        lines = []
        for key, value in context.items():
            if key.startswith("_"):  # bỏ qua các key nội bộ
                continue
            lines.append(f"--- {key.upper()} ---")
            if isinstance(value, (list, dict)):
                import json
                lines.append(json.dumps(value, indent=2, ensure_ascii=False)[:2000])
            else:
                lines.append(str(value)[:2000])
        return "\n\n".join(lines)