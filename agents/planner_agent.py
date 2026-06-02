from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rag_engine.schema import SubQuery, WorkflowPlan


SYSTEM_PROMPT = """Bạn là planner_agent cho một hệ thống RAG chứng khoán Việt Nam.
Nhiệm vụ:
- Phân tích câu hỏi của người dùng.
- Nếu câu hỏi phức tạp, tách thành các sub-queries độc lập hoặc phụ thuộc nhau.
- Chọn strategy là "parallel" khi các sub-queries độc lập, "sequential" khi bước sau cần kết quả bước trước.
- Quyết định có cần retriever và executor hay không.

Chỉ trả về JSON hợp lệ, không markdown, không giải thích.
Schema:
{
  "strategy": "sequential" | "parallel",
  "requires_retrieval": true,
  "requires_execution": true,
  "aggregation_mode": "concat" | "synthesize",
  "sub_queries": [
    {
      "id": "q1",
      "query": "câu hỏi con cụ thể",
      "type": "retrieval_qa",
      "depends_on": [],
      "tool": "retriever"
    }
  ]
}
"""


@dataclass(slots=True)
class PlannerConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    local_files_only: bool = True
    device_map: str | None = "auto"
    load_in_4bit: bool = False
    max_new_tokens: int = 1024
    temperature: float = 0.1
    trust_remote_code: bool = True
    enable_llm: bool = True
    fallback_to_heuristic: bool = True


class PlannerAgent:
    """Planner powered by a local Qwen2.5-7B-Instruct model."""

    def __init__(self, config: PlannerConfig | None = None, system_prompt: str = SYSTEM_PROMPT, **kwargs: Any) -> None:
        self.config = config or PlannerConfig(**kwargs)
        self.system_prompt = system_prompt
        self._tokenizer = None
        self._model = None

    def plan(self, question: str) -> WorkflowPlan:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        if self.config.enable_llm:
            try:
                raw = self._generate_plan(question)
                payload = self._extract_json(raw)
                return WorkflowPlan.from_dict(payload, original_query=question)
            except Exception:
                if not self.config.fallback_to_heuristic:
                    raise

        return self._heuristic_plan(question)

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required to run local Qwen planner_agent") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": self.config.device_map,
        }
        if self.config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch
            except ImportError as exc:
                raise RuntimeError("4-bit loading requires transformers, torch and bitsandbytes") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)

    def _generate_plan(self, question: str) -> str:
        self._load_model()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        tokenizer = self._tokenizer
        model = self._model
        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            prompt = f"{self.system_prompt}\n\nUser: {question}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt")

        try:
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        except AttributeError:
            pass

        outputs = model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_length:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start < 0 or end < start:
                raise
            payload = json.loads(cleaned[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Planner output must be a JSON object")
        return payload

    def _heuristic_plan(self, question: str) -> WorkflowPlan:
        normalized = question.lower()
        no_tool_patterns = ("chào", "hello", "hi", "cảm ơn", "cam on")
        requires_tools = not any(normalized == item or normalized.startswith(item + " ") for item in no_tool_patterns)

        sequential_markers = ("sau đó", "rồi", "tiếp theo", "sau khi", "trước khi")
        parallel_markers = ("đồng thời", "ngoài ra", ";", "\n")
        if any(marker in normalized for marker in sequential_markers):
            strategy = "sequential"
            parts = self._split_question(question, sequential_markers)
        elif any(marker in normalized for marker in parallel_markers):
            strategy = "parallel"
            parts = self._split_question(question, parallel_markers)
        else:
            strategy = "sequential"
            parts = [question]

        sub_queries = [
            SubQuery(id=f"q{i + 1}", query=part, depends_on=[f"q{i}"] if strategy == "sequential" and i else [])
            for i, part in enumerate(parts)
        ]
        return WorkflowPlan(
            original_query=question,
            sub_queries=sub_queries or [SubQuery(id="q1", query=question)],
            strategy=strategy,
            requires_retrieval=requires_tools,
            requires_execution=requires_tools,
            aggregation_mode="concat" if len(parts) <= 2 else "synthesize",
            metadata={"planner": "heuristic"},
        )

    @staticmethod
    def _split_question(question: str, markers: tuple[str, ...]) -> list[str]:
        pattern = "|".join(re.escape(marker) for marker in markers)
        parts = [part.strip(" .?\n\t") for part in re.split(pattern, question, flags=re.IGNORECASE)]
        return [part for part in parts if len(part) >= 8] or [question]
