"""
Quản lý cơ sở dữ liệu vector ChromaDB cho RAG.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Load biến môi trường từ .env
load_dotenv()

logger = logging.getLogger(__name__)


class VectorDBManager:
    """
    Quản lý kết nối và thao tác với ChromaDB.
    Sử dụng embedding từ Google Gemini API.
    """

    def __init__(self, persist_directory: str = "storage/vector_rag/"):
        """
        Khởi tạo VectorDBManager với thư mục persist và embedding model.

        Args:
            persist_directory: Đường dẫn thư mục lưu trữ ChromaDB.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Lấy API key từ biến môi trường
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Thiếu GEMINI_API_KEY trong biến môi trường.")

        # Khởi tạo embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.gemini_api_key
        )

        # Khởi tạo hoặc kết nối Chroma collection
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        logger.info(f"Đã kết nối ChromaDB tại {self.persist_directory}")

    def add_documents(self, chunks: List[str], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Thêm các đoạn văn bản (chunks) vào vector store.

        Args:
            chunks: Danh sách các đoạn văn bản.
            metadata: Metadata gắn với tất cả các chunks (ví dụ: filename, ticker).

        Returns:
            True nếu thành công, False nếu có lỗi.
        """
        if not chunks:
            logger.warning("Không có chunks nào để thêm.")
            return True

        try:
            # Tạo danh sách Document từ chunks
            documents = []
            for i, chunk in enumerate(chunks):
                # Tạo bản sao metadata và thêm chunk index nếu cần
                meta = metadata.copy() if metadata else {}
                meta["chunk_index"] = i
                doc = Document(page_content=chunk, metadata=meta)
                documents.append(doc)

            # Thêm vào vectorstore
            self.vectorstore.add_documents(documents)
            # Persist dữ liệu (Chroma tự động persist khi thêm)
            logger.info(f"Đã thêm {len(chunks)} chunks vào vector DB.")
            return True

        except Exception as e:
            logger.exception(f"Lỗi khi thêm documents vào vector DB: {e}")
            return False