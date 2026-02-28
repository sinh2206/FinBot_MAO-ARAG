#!/usr/bin/env python3
"""
Pipeline xử lý và ingest dữ liệu từ các file PDF trong thư mục temp.
Quy trình:
1. Quét tất cả file .pdf trong storage/temp/
2. Kiểm tra SQLite xem file đã được ingest thành công chưa
3. Nếu chưa, dùng OCRTool để trích xuất văn bản
4. Chia nhỏ văn bản thành chunks
5. Lưu chunks vào VectorDB
6. Ghi log thành công vào SQLite
7. Di chuyển file đã xử lý vào storage/archive/
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import các module từ app
from app.tools.ocr_tool import OCRTool
from app.memory.sqlite_db import SQLiteManager
from app.memory.vector_db import VectorDBManager

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class IngestPipeline:
    """Điều phối toàn bộ quy trình ingestion."""

    def __init__(self, temp_dir: str = "storage/temp", archive_dir: str = "storage/archive"):
        """
        Khởi tạo pipeline với các thư mục và các manager.

        Args:
            temp_dir: Thư mục chứa các file PDF đầu vào.
            archive_dir: Thư mục lưu trữ file sau khi xử lý thành công.
        """
        self.temp_dir = Path(temp_dir)
        self.archive_dir = Path(archive_dir)

        # Tạo các thư mục nếu chưa tồn tại
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Khởi tạo các thành phần
        self.ocr_tool = OCRTool()
        self.sqlite_manager = SQLiteManager()
        self.vector_db = VectorDBManager()

        # Cấu hình text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_ticker_from_filename(self, filename: str) -> Optional[str]:
        """
        Trích xuất mã cổ phiếu từ tên file.
        Giả định ticker là chuỗi chữ in hoa ở đầu tên file (trước dấu gạch ngang hoặc dấu cách).
        Ví dụ: "FPT-Q3-2024.pdf" -> "FPT", "VNM_BCKT.pdf" -> "VNM".

        Args:
            filename: Tên file.

        Returns:
            Mã cổ phiếu hoặc None.
        """
        # Loại bỏ phần mở rộng
        base = Path(filename).stem
        # Tìm chuỗi chữ in hoa đầu tiên (2-5 ký tự) ở đầu
        match = re.match(r"^([A-Z]{2,5})", base)
        if match:
            return match.group(1)
        return None

    def process_file(self, file_path: Path) -> bool:
        """
        Xử lý một file PDF: OCR, chunk, vector hóa, log.

        Args:
            file_path: Đường dẫn file PDF.

        Returns:
            True nếu xử lý thành công, False nếu thất bại hoặc đã xử lý trước đó.
        """
        filename = file_path.name
        logger.info(f"Bắt đầu xử lý file: {filename}")

        # Kiểm tra xem file đã được ingest thành công chưa
        if self.sqlite_manager.check_if_ingested(filename):
            logger.info(f"File {filename} đã được ingest trước đó, bỏ qua.")
            return False

        ticker = self.extract_ticker_from_filename(filename)

        # Bước 1: OCR
        logger.info(f"Đang trích xuất văn bản từ {filename}...")
        text = self.ocr_tool.extract(file_path)
        if text is None:
            logger.error(f"Không thể trích xuất văn bản từ {filename}")
            self.sqlite_manager.log_ingestion(filename, ticker, "failed")
            return False
        if not text.strip():
            logger.warning(f"File {filename} không chứa nội dung văn bản.")
            # Vẫn có thể lưu? Nhưng không có gì để chunk -> bỏ qua
            self.sqlite_manager.log_ingestion(filename, ticker, "failed")
            return False

        # Bước 2: Chunking
        logger.info(f"Đang chunk văn bản...")
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Đã tạo {len(chunks)} chunks.")

        if not chunks:
            logger.warning(f"Không tạo được chunk nào từ {filename}")
            self.sqlite_manager.log_ingestion(filename, ticker, "failed")
            return False

        # Bước 3: Lưu vào VectorDB
        metadata = {
            "filename": filename,
            "ticker": ticker if ticker else "unknown",
            "source": str(file_path)
        }
        success = self.vector_db.add_documents(chunks, metadata)
        if not success:
            logger.error(f"Lỗi khi lưu chunks vào vector DB cho {filename}")
            self.sqlite_manager.log_ingestion(filename, ticker, "failed")
            return False

        # Bước 4: Ghi log thành công
        self.sqlite_manager.log_ingestion(filename, ticker, "success")
        logger.info(f"Đã xử lý thành công file {filename}")

        # Bước 5: Di chuyển file vào archive
        dest_path = self.archive_dir / filename
        try:
            shutil.move(str(file_path), str(dest_path))
            logger.info(f"Đã di chuyển {filename} vào archive.")
        except Exception as e:
            logger.exception(f"Lỗi khi di chuyển file {filename} vào archive: {e}")
            # Không đánh trượt pipeline vì đã lưu DB thành công

        return True

    def run(self):
        """Quét thư mục temp và xử lý lần lượt các file PDF."""
        logger.info("=" * 50)
        logger.info("BẮT ĐẦU PIPELINE INGESTION")
        logger.info("=" * 50)

        # Lấy tất cả file PDF trong temp_dir
        pdf_files = list(self.temp_dir.glob("*.pdf")) + list(self.temp_dir.glob("*.PDF"))
        if not pdf_files:
            logger.info("Không tìm thấy file PDF nào trong thư mục temp.")
            return

        logger.info(f"Tìm thấy {len(pdf_files)} file PDF cần xử lý.")

        success_count = 0
        for pdf_file in pdf_files:
            try:
                if self.process_file(pdf_file):
                    success_count += 1
            except Exception as e:
                logger.exception(f"Lỗi không mong muốn khi xử lý {pdf_file.name}: {e}")
                # Vẫn log thất bại
                self.sqlite_manager.log_ingestion(pdf_file.name, None, "failed")

        logger.info("=" * 50)
        logger.info(f"KẾT THÚC PIPELINE: {success_count}/{len(pdf_files)} file thành công.")
        logger.info("=" * 50)


if __name__ == "__main__":
    pipeline = IngestPipeline()
    pipeline.run()