"""
Document Ranker: Lọc và xếp hạng tài liệu dùng Cohere Rerank hoặc LLM.
"""

import logging
import os
from typing import List, Dict, Any

import aiohttp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.mao_core.cost_manager import track_latency

logger = logging.getLogger(__name__)


class DocumentRanker:
    """
    Xếp hạng lại các documents để chọn lọc những đoạn liên quan nhất.
    Sử dụng Cohere Rerank nếu có API key, fallback dùng LLM.
    """

    def __init__(self, top_k: int = 3, use_cohere: bool = True):
        self.top_k = top_k
        self.use_cohere = use_cohere and os.getenv("COHERE_API_KEY") is not None
        if self.use_cohere:
            self.cohere_api_key = os.getenv("COHERE_API_KEY")
            self.cohere_url = "https://api.cohere.ai/v1/rerank"
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """Bạn là một chuyên gia đánh giá mức độ liên quan của tài liệu với câu hỏi.
Cho câu hỏi và một đoạn văn, hãy trả về một số điểm từ 0 đến 10 (số thực) thể hiện độ liên quan.
Chỉ trả về số, không giải thích."""),
                ("human", "Câu hỏi: {query}\nĐoạn văn: {document}\nĐiểm (0-10):")
            ])

    @track_latency
    async def rank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Xếp hạng documents theo query.

        Args:
            query: Câu hỏi.
            documents: Danh sách document (mỗi doc có 'content').

        Returns:
            Danh sách documents đã được sắp xếp theo điểm giảm dần, lấy top_k.
        """
        if not documents:
            return []

        if self.use_cohere:
            return await self._rank_cohere(query, documents)
        else:
            return await self._rank_llm(query, documents)

    async def _rank_cohere(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Dùng Cohere Rerank API."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "rerank-english-v2.0",
                    "query": query,
                    "documents": [doc["content"] for doc in documents],
                    "top_n": self.top_k
                }
                headers = {
                    "Authorization": f"BEARER {self.cohere_api_key}",
                    "Content-Type": "application/json"
                }
                async with session.post(self.cohere_url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        # Sắp xếp lại documents theo index từ results
                        ranked = []
                        for item in results:
                            idx = item["index"]
                            doc = documents[idx].copy()
                            doc["relevance_score"] = item["relevance_score"]
                            ranked.append(doc)
                        logger.info(f"Cohere reranked {len(ranked)} documents.")
                        return ranked[:self.top_k]
                    else:
                        logger.error(f"Cohere API error: {resp.status}")
                        # Fallback: trả về documents gốc (cắt top_k)
                        return documents[:self.top_k]
        except Exception as e:
            logger.exception(f"Cohere rerank failed: {e}")
            return documents[:self.top_k]

    async def _rank_llm(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Dùng LLM để chấm điểm từng document."""
        scored_docs = []
        for doc in documents:
            try:
                # Gọi LLM để lấy điểm
                chain = self.prompt | self.llm
                response = await chain.ainvoke({"query": query, "document": doc["content"]})
                # response.content là string, parse thành float
                try:
                    score = float(response.content.strip())
                except ValueError:
                    score = 5.0  # default
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = score
                scored_docs.append(doc_copy)
            except Exception as e:
                logger.warning(f"Error scoring document: {e}")
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = 5.0
                scored_docs.append(doc_copy)

        # Sắp xếp theo score giảm dần
        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info(f"LLM reranked {len(scored_docs)} documents.")
        return scored_docs[:self.top_k]