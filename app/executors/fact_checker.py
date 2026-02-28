"""
FactCheckerAgent: Kiểm tra tính xác thực của bản nháp, đối chiếu với context gốc,
thêm trích dẫn và loại bỏ hallucination.
"""

import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class FactCheckerAgent:
    """
    Agent kiểm tra sự thật, thêm citation và loại bỏ hallucination.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một chuyên gia kiểm tra tính xác thực của các báo cáo tài chính.
Nhiệm vụ của bạn là đối chiếu bản nháp (draft) với các tài liệu gốc (context) và thực hiện:
1. Với mỗi câu/cụm từ chứa số liệu hoặc khẳng định quan trọng, hãy kiểm tra xem nó có được hỗ trợ bởi context không.
2. Nếu có (số liệu khớp với context), hãy giữ nguyên câu và thêm citation ngay cuối câu dưới dạng: `[Nguồn: <tên file>, Trang <số trang>]`. Nếu không có số trang, chỉ ghi tên file.
3. Nếu không tìm thấy số liệu/khẳng định trong context (hallucination), hãy xóa bỏ câu đó hoàn toàn. Trong trường hợp thông tin quan trọng nhưng không có nguồn, có thể thay thế bằng dòng cảnh báo: `*[Cảnh báo: Không có đủ dữ liệu gốc để xác minh thông tin này]*`.
4. Giữ nguyên cấu trúc Markdown của bản nháp (bảng, hình ảnh, ...), chỉ sửa nội dung câu chữ và thêm citation.

Hãy trả về văn bản đã được xử lý hoàn chỉnh.

Context là danh sách các đoạn văn kèm metadata. Bạn có thể dùng thông tin metadata để trích dẫn.
Nếu context không chứa thông tin phù hợp, hãy đánh dấu là không xác minh được.
"""),
            ("human", "Bản nháp:\n{draft}\n\nContext:\n{formatted_context}")
        ])

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context để đưa vào prompt."""
        lines = []
        for i, chunk in enumerate(context):
            # chunk có thể là dict chứa 'content' và 'metadata', hoặc string
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                source = metadata.get('filename', 'Unknown')
                page = metadata.get('page_number', 'N/A')
            else:
                content = str(chunk)
                source = 'Unknown'
                page = 'N/A'
            lines.append(f"[Đoạn {i+1}] (Nguồn: {source}, Trang: {page}):\n{content}\n")
        return "\n".join(lines)

    async def verify_and_cite(self, draft: str, context: List[Dict[str, Any]]) -> str:
        """
        Kiểm tra và thêm trích dẫn cho bản nháp dựa trên context.

        Args:
            draft: Bản nháp do generator sinh ra.
            context: Danh sách các chunks tài liệu gốc (mỗi chunk là dict chứa content và metadata).

        Returns:
            Văn bản đã được kiểm tra và bổ sung citation.
        """
        logger.info("Starting fact-checking process")
        formatted_context = self._format_context(context)
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
            # Trả về draft gốc nhưng kèm cảnh báo
            return draft + "\n\n*[Cảnh báo: Quá trình kiểm tra gặp lỗi, vui lòng xem xét độc lập.]*"