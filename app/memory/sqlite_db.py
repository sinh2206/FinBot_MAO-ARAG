"""
Quản lý cơ sở dữ liệu SQLite cho lịch sử chat và ingestion.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class SQLiteManager:
    """
    Quản lý kết nối và thao tác với database SQLite lưu thông tin file đã ingest.
    """

    def __init__(self, db_path: str = "storage/MAO.db"):
        """
        Khởi tạo kết nối database và tạo bảng nếu chưa tồn tại.

        Args:
            db_path: Đường dẫn tới file database.
        """
        self.db_path = Path(db_path)
        # Đảm bảo thư mục cha tồn tại
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Tạo bảng ingested_documents nếu chưa có."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ingested_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            ticker TEXT,
            status TEXT NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with self._get_connection() as conn:
                conn.execute(create_table_sql)
                conn.commit()
            logger.info("Đã khởi tạo bảng ingested_documents.")
        except sqlite3.Error as e:
            logger.exception(f"Lỗi khi tạo bảng: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """Tạo và trả về kết nối SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Cho phép truy cập theo tên cột
        return conn

    def check_if_ingested(self, filename: str) -> bool:
        """
        Kiểm tra xem một file đã được ingest thành công chưa.

        Args:
            filename: Tên file cần kiểm tra.

        Returns:
            True nếu đã tồn tại bản ghi với status='success', False nếu chưa.
        """
        query = "SELECT status FROM ingested_documents WHERE filename = ? AND status = 'success'"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(query, (filename,))
                row = cursor.fetchone()
                return row is not None
        except sqlite3.Error as e:
            logger.exception(f"Lỗi khi kiểm tra file {filename}: {e}")
            return False  # Nếu có lỗi, coi như chưa ingest để xử lý lại an toàn

    def log_ingestion(self, filename: str, ticker: Optional[str], status: str) -> None:
        """
        Ghi lại thông tin ingestion của một file.

        Args:
            filename: Tên file.
            ticker: Mã cổ phiếu (nếu có).
            status: Trạng thái ('success' hoặc 'failed').
        """
        # Sử dụng INSERT OR REPLACE để tránh trùng lặp filename
        query = """
        INSERT OR REPLACE INTO ingested_documents (filename, ticker, status, ingested_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(query, (filename, ticker, status))
                conn.commit()
            logger.info(f"Đã ghi log ingestion: {filename} - {status}")
        except sqlite3.Error as e:
            logger.exception(f"Lỗi khi ghi log ingestion cho {filename}: {e}")
            # Không raise để pipeline vẫn tiếp tục với file khác
class ChatHistoryDB:
    """
    Quản lý lịch sử chat của người dùng.
    """

    def __init__(self, db_path: str = "storage/MAO.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        """Tạo bảng chat_logs nếu chưa tồn tại."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            tokens_input INTEGER DEFAULT 0,
            tokens_output INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            workflow TEXT,  -- JSON list
            latency REAL DEFAULT 0.0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_user_id ON chat_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_logs(timestamp);
        """
        try:
            with self._get_connection() as conn:
                conn.executescript(create_table_sql)
                conn.commit()
            logger.info("Initialized chat_logs table.")
        except sqlite3.Error as e:
            logger.exception(f"Error creating chat_logs table: {e}")
            raise

    def save_chat(self, user_id: str, query: str, response: str,
                  tokens_input: int = 0, tokens_output: int = 0,
                  cost: float = 0.0, workflow: Optional[List[str]] = None,
                  latency: float = 0.0, session_id: Optional[str] = None) -> None:
        """
        Lưu một cuộc chat vào database.
        """
        import json
        workflow_json = json.dumps(workflow) if workflow else None
        insert_sql = """
        INSERT INTO chat_logs 
            (user_id, session_id, query, response, tokens_input, tokens_output, cost, workflow, latency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(insert_sql, (
                    user_id, session_id, query, response,
                    tokens_input, tokens_output, cost,
                    workflow_json, latency
                ))
                conn.commit()
            logger.info(f"Saved chat for user {user_id}, cost={cost:.6f}")
        except sqlite3.Error as e:
            logger.exception(f"Error saving chat: {e}")

    def get_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử chat của user, sắp xếp mới nhất trước.
        """
        select_sql = """
        SELECT id, user_id, session_id, query, response,
               tokens_input, tokens_output, cost, workflow, latency, timestamp
        FROM chat_logs
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(select_sql, (user_id, limit))
                rows = cursor.fetchall()
                history = []
                for row in rows:
                    item = dict(row)
                    # Chuyển timestamp thành string
                    if isinstance(item['timestamp'], str):
                        item['timestamp'] = item['timestamp']
                    else:
                        item['timestamp'] = item['timestamp'].isoformat()
                    history.append(item)
                return history
        except sqlite3.Error as e:
            logger.exception(f"Error fetching history for user {user_id}: {e}")
            return []