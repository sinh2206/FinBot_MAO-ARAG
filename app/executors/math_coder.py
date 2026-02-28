"""
Math Coder: Sinh và thực thi code Python để tính toán các chỉ số tài chính.
"""

import logging
import asyncio
import sys
from io import StringIO
from contextlib import redirect_stdout
from typing import Any, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.mao_core.cost_manager import track_latency

logger = logging.getLogger(__name__)


class MathCoder:
    """
    Sinh code Python dựa trên yêu cầu, thực thi an toàn và trả về kết quả.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", timeout: int = 10):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        self.timeout = timeout  # giới hạn thời gian chạy code (giây)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia lập trình Python trong lĩnh vực tài chính.
Nhiệm vụ: Viết một đoạn code Python thuần túy (không dùng thư viện ngoài ngoại trừ các thư viện chuẩn) để giải bài toán tài chính sau.
Code phải:
- Sử dụng các biến đầu vào được cung cấp (nếu cần).
- In ra kết quả cuối cùng (dùng print).
- Không chứa vòng lặp vô hạn, không nhập input().
- Chỉ trả về code, không giải thích.
"""),
            ("human", "Bài toán: {query}\nHãy viết code Python:")
        ])

    @track_latency
    async def execute_math(self, context: Dict[str, Any]) -> Optional[float]:
        """
        Nhận query (bài toán), sinh code, thực thi và trả về kết quả số.

        Args:
            context: Chứa 'query' mô tả bài toán.

        Returns:
            Kết quả số (float) hoặc None nếu lỗi.
        """
        query = context.get("query", "")
        if not query:
            logger.warning("No query provided for math coder.")
            return None

        # 1. Sinh code
        code = await self._generate_code(query)
        if not code:
            return None

        # 2. Thực thi code an toàn
        result = await self._safe_execute(code)
        return result

    async def _generate_code(self, query: str) -> Optional[str]:
        """Gọi LLM để sinh code."""
        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({"query": query})
            code = response.content.strip()
            # Loại bỏ markdown code block nếu có
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
            logger.debug(f"Generated code:\n{code}")
            return code
        except Exception as e:
            logger.exception(f"Error generating code: {e}")
            return None

    async def _safe_execute(self, code: str) -> Optional[float]:
        """
        Thực thi code trong môi trường an toàn, giới hạn thời gian.
        """
        # Tạo namespace an toàn
        allowed_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'abs': abs,
                'round': round,
                'max': max,
                'min': min,
                'sum': sum,
                'pow': pow,
            }
        }

        # Capture output
        stdout = StringIO()
        result = None

        def exec_code():
            nonlocal result
            try:
                with redirect_stdout(stdout):
                    exec(code, allowed_globals, {})
                output = stdout.getvalue().strip()
                # Parse kết quả từ output (giả sử in ra số cuối cùng)
                if output:
                    # Lấy dòng cuối cùng
                    last_line = output.split('\n')[-1]
                    try:
                        result = float(last_line)
                    except ValueError:
                        logger.warning(f"Could not parse result from output: {last_line}")
            except Exception as e:
                logger.exception(f"Error executing code: {e}")

        try:
            # Chạy trong thread riêng để có thể timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(loop.run_in_executor(None, exec_code), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(f"Code execution timed out after {self.timeout}s")
            return None

        return result