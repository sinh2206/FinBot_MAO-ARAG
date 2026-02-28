"""
Auto Evaluator: Sử dụng LLM để chấm điểm câu trả lời (F1, Faithfulness).
"""

import json
import logging
from typing import Dict, Any, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvaluationScore(BaseModel):
    """Điểm đánh giá từ LLM."""
    f1_score: float = Field(ge=0.0, le=1.0, description="Điểm F1 (độ chính xác) từ 0 đến 1")
    faithfulness: float = Field(ge=0.0, le=1.0, description="Điểm faithfulness (không bịa đặt) từ 0 đến 1")
    explanation: str = Field("", description="Giải thích ngắn gọn")


class AutoEvaluator:
    """
    Đánh giá câu trả lời dựa trên câu trả lời đúng kỳ vọng.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=EvaluationScore)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là giám khảo AI chuyên đánh giá chất lượng câu trả lời trong lĩnh vực tài chính.
Bạn sẽ nhận được:
- Câu hỏi gốc
- Câu trả lời đúng (expected)
- Câu trả lời thực tế (actual)

Nhiệm vụ:
1. Tính điểm F1 (precision/recall) dựa trên mức độ trùng khớp thông tin giữa actual và expected. Lưu ý: có thể diễn đạt khác nhưng ý chính phải đúng. Điểm 1 nếu hoàn toàn đúng, 0 nếu sai hoàn toàn.
2. Tính điểm faithfulness: đánh giá xem actual có bịa đặt thông tin không có trong expected hoặc không được hỗ trợ bởi context (coi expected là ground truth). Điểm 1 nếu không bịa, 0 nếu bịa hoàn toàn.
3. Cung cấp giải thích ngắn.

Trả về JSON theo schema.
"""),
            ("human", """
Câu hỏi: {query}
Câu trả lời đúng: {expected}
Câu trả lời thực tế: {actual}
""")
        ])

        self.chain = self.prompt | self.llm | self.parser

    async def evaluate(self, query: str, expected: str, actual: str) -> EvaluationScore:
        """
        Đánh giá cặp expected/actual.
        """
        try:
            result = await self.chain.ainvoke({
                "query": query,
                "expected": expected,
                "actual": actual
            })
            logger.info(f"Evaluation: F1={result.f1_score}, Faith={result.faithfulness}")
            return result
        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            # Trả về điểm mặc định (0) khi lỗi
            return EvaluationScore(f1_score=0.0, faithfulness=0.0, explanation="Evaluation error")

    async def evaluate_from_file(self, golden_path: str, actuals: Dict[int, str]) -> Dict[int, EvaluationScore]:
        """
        Đánh giá toàn bộ golden dataset với các actual answers.

        Args:
            golden_path: Đường dẫn đến file golden_dataset.json.
            actuals: Dict mapping id -> actual answer.

        Returns:
            Dict mapping id -> EvaluationScore.
        """
        with open(golden_path, 'r', encoding='utf-8') as f:
            golden = json.load(f)

        results = {}
        for item in golden:
            qid = item['id']
            if qid in actuals:
                score = await self.evaluate(item['query'], item['expected_answer'], actuals[qid])
                results[qid] = score
            else:
                logger.warning(f"No actual answer for id {qid}")
        return results