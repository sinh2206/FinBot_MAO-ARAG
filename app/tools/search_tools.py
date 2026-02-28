"""
Search Tools: Tìm kiếm tin tức mới nhất qua Google Serper API.
"""

import logging
import os
from typing import List, Dict, Any

import aiohttp

from app.mao_core.cost_manager import track_latency

logger = logging.getLogger(__name__)


class SearchTools:
    """
    Tìm kiếm tin tức chứng khoán qua Serper.dev.
    """

    def __init__(self, max_results: int = 3):
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            logger.warning("SERPER_API_KEY not set. Search will return empty.")
        self.url = "https://google.serper.dev/search"
        self.max_results = max_results

    @track_latency
    async def search_news(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Tìm kiếm tin tức liên quan đến query.

        Args:
            context: Chứa 'query' hoặc 'ticker'.

        Returns:
            Danh sách các bài báo (title, link, snippet).
        """
        query = context.get("query", "")
        if not query:
            # Có thể dùng ticker từ context
            ticker = context.get("ticker", "")
            if ticker:
                query = f"{ticker} tin tức chứng khoán"
            else:
                logger.warning("No query for search.")
                return []

        logger.info(f"Searching news for: {query}")
        if not self.api_key:
            return self._mock_search(query)

        try:
            async with aiohttp.ClientSession() as session:
                payload = {"q": query, "num": self.max_results, "gl": "vn", "hl": "vi"}
                headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
                async with session.post(self.url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Parse organic results
                        organic = data.get("organic", [])
                        results = []
                        for item in organic[:self.max_results]:
                            results.append({
                                "title": item.get("title"),
                                "link": item.get("link"),
                                "snippet": item.get("snippet")
                            })
                        logger.info(f"Found {len(results)} news items.")
                        return results
                    else:
                        logger.error(f"Serper API error: {resp.status}")
                        return self._mock_search(query)
        except Exception as e:
            logger.exception(f"Search error: {e}")
            return self._mock_search(query)

    def _mock_search(self, query: str) -> List[Dict]:
        """Trả về dữ liệu giả khi không có API key."""
        return [
            {"title": f"Tin giả về {query} số 1", "link": "#", "snippet": "Nội dung mô phỏng..."},
            {"title": f"Tin giả về {query} số 2", "link": "#", "snippet": "Nội dung mô phỏng..."}
        ]