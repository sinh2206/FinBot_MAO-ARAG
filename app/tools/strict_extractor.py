"""
StrictExtractor: Trích xuất chính xác các chỉ số tài chính từ văn bản, không suy luận.
Sử dụng Structured Output của LLM để đảm bảo định dạng.
"""

import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class FinancialMetric(BaseModel):
    """Một chỉ số tài chính được trích xuất từ văn bản."""
    metric_name: str = Field(description="Tên chỉ số, ví dụ: 'Lợi nhuận sau thuế', 'Doanh thu', 'EPS'")
    value: float = Field(description="Giá trị số của chỉ số")
    unit: str = Field(description="Đơn vị (tỷ, triệu, %, ... hoặc rỗng nếu không có)")
    source_document: str = Field(description="Tên tài liệu gốc (nếu biết, có thể là 'Unknown')")
    page_number: Optional[int] = Field(None, description="Số trang (nếu có)")


class StrictExtractor:
    """
    Công cụ trích xuất chỉ số tài chính chỉ dựa trên văn bản gốc,
    không suy luận, không tính toán.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        # Sử dụng structured output để đảm bảo output đúng schema
        self.llm_with_structure = self.llm.with_structured_output(List[FinancialMetric])

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là công cụ trích xuất dữ liệu tài chính.
BẠN KHÔNG ĐƯỢC SUY LUẬN HAY TÍNH TOÁN.
Chỉ trích xuất những con số, chỉ số có sẵn trong văn bản gốc.
Nếu văn bản không chứa dữ liệu, hãy trả về danh sách rỗng.
Metadata đã cho bao gồm tên file và số trang (nếu có). Hãy điền vào các trường source_document và page_number từ metadata.
Chú ý: Không tự thêm đơn vị nếu không có; để trống hoặc ghi là 'unknown'.
Danh sách các chỉ số trả về phải là mảng các object theo schema FinancialMetric.
"""),
            ("human", "Đoạn văn: {text}\nMetadata: {metadata}")
        ])

        self.chain = self.prompt | self.llm_with_structure

    async def extract_metrics(self, text: str, metadata: Dict[str, Any]) -> List[FinancialMetric]:
        """
        Trích xuất các chỉ số tài chính từ đoạn văn.

        Args:
            text: Nội dung văn bản cần trích xuất.
            metadata: Thông tin metadata (filename, page_number, ...)

        Returns:
            Danh sách các FinancialMetric. Nếu không có, trả về [].
        """
        logger.info(f"Extracting metrics from text (length={len(text)})")
        try:
            result = await self.chain.ainvoke({"text": text, "metadata": metadata})
            logger.info(f"Extracted {len(result)} metrics")
            return result
        except Exception as e:
            logger.exception(f"Error during extraction: {e}")
            return []  # Không trả về None, mà list rỗng để xử lý an toàn