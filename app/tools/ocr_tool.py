"""
Công cụ OCR để trích xuất văn bản và bảng từ file PDF.
Sử dụng pdfplumber để đọc nội dung.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pdfplumber

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OCRTool:
    """
    Lớp OCRTool trích xuất toàn bộ văn bản và bảng từ file PDF.
    """

    def __init__(self):
        """Khởi tạo OCRTool."""
        pass

    def extract(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Trích xuất văn bản và bảng từ file PDF.

        Args:
            file_path: Đường dẫn đến file PDF (có thể là string hoặc Path).

        Returns:
            Chuỗi văn bản đã được trích xuất (bao gồm cả bảng chuyển thành text),
            hoặc None nếu có lỗi.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File không tồn tại: {file_path}")
            return None

        try:
            full_text = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Trích xuất văn bản
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(f"--- Trang {page_num} ---\n{page_text}")

                    # Trích xuất bảng
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables, start=1):
                        if table:
                            # Chuyển bảng thành dạng text (các dòng, các ô cách nhau bởi tab)
                            table_str = "\n".join(
                                "\t".join(cell if cell is not None else "" for cell in row)
                                for row in table
                            )
                            full_text.append(f"--- Bảng {page_num}.{table_num} ---\n{table_str}")

            if not full_text:
                logger.warning(f"Không trích xuất được nội dung từ file: {file_path}")
                return ""

            result = "\n\n".join(full_text)
            logger.info(f"Đã trích xuất {len(full_text)} phần tử (trang/bảng) từ {file_path.name}")
            return result

        except Exception as e:
            logger.exception(f"Lỗi khi xử lý file {file_path}: {e}")
            return None