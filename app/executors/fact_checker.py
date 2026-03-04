"""
FactCheckerAgent: Kiểm tra tính xác thực của bản nháp, đối chiếu với context gốc,
thêm trích dẫn (kèm nguồn) và loại bỏ hallucination.
"""

import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class FactCheckerAgent:
    """
    Agent kiểm tra sự thật, thêm citation và loại bỏ hallucination.
    Kết quả đầu ra luôn kèm nguồn gốc hoặc nhãn mô phỏng.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một chuyên gia kiểm tra tính xác thực của các báo cáo tài chính.
Nhiệm vụ của bạn là đối chiếu bản nháp (draft) với các tài liệu gốc (context) và thực hiện:

1. Với mỗi câu/cụm từ chứa số liệu hoặc khẳng định quan trọng, hãy kiểm tra xem nó có được hỗ trợ bởi context không.

2. Nếu có (số liệu khớp với context):
   - Giữ nguyên câu và thêm citation ngay cuối câu dưới dạng:
     `[Nguồn: <tên file>, Trang <số trang>]`
   - Nếu tên file chứa từ "simulated" (ví dụ: simulated_data.pdf), hãy ghi:
     `[Nguồn mô phỏng: <tên file>, Trang <số trang>]` (nếu không có trang thì bỏ qua phần trang)

3. Nếu không tìm thấy số liệu/khẳng định trong context (hallucination):
   - Xóa bỏ câu đó hoàn toàn.
   - Trong trường hợp thông tin quan trọng nhưng không có nguồn, có thể thay thế bằng dòng cảnh báo:
     `*[Cảnh báo: Thông tin này không có nguồn gốc từ dữ liệu thực tế, chỉ là mô phỏng.]*`

4. Giữ nguyên cấu trúc Markdown của bản nháp (bảng, hình ảnh, ...), chỉ sửa nội dung câu chữ và thêm citation.

Context là danh sách các đoạn văn kèm metadata (tên file, trang, is_simulated). Bạn phải dựa vào metadata này để trích dẫn.
Hãy trả về văn bản đã được xử lý hoàn chỉnh.
"""),
            ("human", "Bản nháp:\n{draft}\n\nContext:\n{formatted_context}")
        ])

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context để đưa vào prompt, kèm thông tin nguồn và ghi chú mô phỏng."""
        lines = []
        for i, chunk in enumerate(context):
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                source = metadata.get('filename', 'Unknown')
                page = metadata.get('page_number', 'N/A')
                is_sim = metadata.get('is_simulated', False)
                source_type = "Mô phỏng" if is_sim else "Thực tế"
            else:
                content = str(chunk)
                source = 'Unknown'
                page = 'N/A'
                source_type = "Không rõ"
            lines.append(
                f"[Đoạn {i+1}] (Nguồn: {source}, Trang: {page}, Loại: {source_type}):\n{content}\n"
            )
        return "\n".join(lines)

    async def verify_and_cite(self, context: Dict[str, Any]) -> str:
        """
        Kiểm tra và thêm trích dẫn. Context chứa:
        - 'generator': bản nháp từ generator (hoặc 'draft')
        - 'fact_checker_context': list các nguồn.
        """
        draft = context.get("generator", "") or context.get("draft", "")
        source_list = context.get("fact_checker_context", [])

        if not draft:
            logger.warning("No draft provided to fact checker")
            return ""

        logger.info("Starting fact-checking process")
        formatted_context = self._format_context(source_list)

        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({
                "draft": draft,
                "formatted_context": formatted_context
            })
            verified_text = response.content
            logger.info("Fact-checking completed")
            return verified_text
        except Exception as e:
            logger.exception(f"Error during fact-checking: {e}")
            return draft + "\n\n*[Cảnh báo: Quá trình kiểm tra gặp lỗi, vui lòng xem xét độc lập.]*"