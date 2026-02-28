"""
Retriever: Tìm kiếm tài liệu từ VectorDB.
"""

import logging
from typing import List, Dict, Any

from app.memory.vector_db import VectorDBManager
from app.mao_core.cost_manager import track_latency

logger = logging.getLogger(__name__)


class Retriever:
    """
    Truy xuất các chunks liên quan từ ChromaDB.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.vector_db = VectorDBManager()  # Khởi tạo kết nối

    @track_latency
    async def retrieve(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các chunks dựa trên query.

        Args:
            context: Context chứa 'query' (câu hỏi gốc hoặc câu hỏi con).

        Returns:
            Danh sách các document dạng dict (có content, metadata, score).
        """
        query = context.get("query", "")
        if not query:
            logger.warning("No query provided for retrieval.")
            return []

        logger.info(f"Retrieving for: {query}")
        try:
            # Giả sử VectorDBManager có method similarity_search_with_score
            # Nếu chưa có, ta implement tạm
            # Ở đây dùng langchain Chroma trực tiếp
            docs_with_score = self.vector_db.vectorstore.similarity_search_with_score(query, k=self.top_k)
            results = []
            for doc, score in docs_with_score:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            logger.info(f"Retrieved {len(results)} documents.")
            return results
        except Exception as e:
            logger.exception(f"Error during retrieval: {e}")
            return []