from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from rag_engine.schema import Answer, Document, RetrievalResult
from rag_engine.retriever import tokenize


EXTRACTIVE_SYSTEM_PROMPT = """Bạn là executor_agent làm extractive QA.
Dựa vào các đoạn văn được cung cấp, hãy trả lời câu hỏi bằng cách trích dẫn nguyên văn cụm từ hoặc câu trả lời xuất hiện trong đoạn.
Không suy diễn, không viết kiến thức ngoài đoạn văn.
Nếu không tìm thấy câu trả lời nguyên văn, chỉ trả lời: KHÔNG TÌM THẤY
"""


@dataclass(slots=True)
class ExecutorConfig:
    model_name: str = "MiniMaxAI/MiniMax-M2.1"
    local_files_only: bool = True
    device_map: str | None = "auto"
    load_in_4bit: bool = False
    max_new_tokens: int = 128
    temperature: float = 0.0
    trust_remote_code: bool = True
    max_context_chars: int = 6000
    enable_model: bool = True
    fallback_to_heuristic: bool = True


class ExecutorAgent:
    """Extractive QA executor backed by local MiniMax-M2.1 when available."""

    def __init__(
        self,
        config: ExecutorConfig | None = None,
        system_prompt: str = EXTRACTIVE_SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        self.config = config or ExecutorConfig(**kwargs)
        self.system_prompt = system_prompt
        self._tokenizer = None
        self._model = None

    def answer(self, sub_query: str, contexts: Iterable[RetrievalResult | Document | str]) -> Answer:
        results = self._normalize_contexts(contexts)
        context_text = self._format_contexts(results)
        if not context_text:
            return Answer(query=sub_query, answer="KHÔNG TÌM THẤY", contexts=results, confidence=0.0)

        if self.config.enable_model:
            try:
                generated = self._generate(sub_query, context_text)
                extracted = self._clean_answer(generated)
                if self._is_verbatim_answer(extracted, context_text):
                    return Answer(query=sub_query, answer=extracted, contexts=results, confidence=0.8)
                if extracted == "KHÔNG TÌM THẤY":
                    return Answer(query=sub_query, answer=extracted, contexts=results, confidence=0.0)
            except Exception:
                if not self.config.fallback_to_heuristic:
                    raise

        fallback = self._heuristic_extract(sub_query, results)
        confidence = 0.35 if fallback != "KHÔNG TÌM THẤY" else 0.0
        return Answer(query=sub_query, answer=fallback, contexts=results, confidence=confidence, metadata={"executor": "heuristic"})

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required to run local MiniMax executor_agent") from exc

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

    def _generate(self, sub_query: str, context_text: str) -> str:
        self._load_model()
        user_prompt = f"Câu hỏi: {sub_query}\n\nĐoạn văn:\n{context_text}\n\nCâu trả lời nguyên văn:"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
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
            prompt = f"{self.system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt")

        try:
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        except AttributeError:
            pass

        outputs = model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 1e-6),
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_length:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _normalize_contexts(self, contexts: Iterable[RetrievalResult | Document | str]) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for index, item in enumerate(contexts):
            if isinstance(item, RetrievalResult):
                results.append(item)
            else:
                document = Document.from_any(item, index=index)
                results.append(RetrievalResult(document=document, score=0.0))
        return results

    def _format_contexts(self, contexts: list[RetrievalResult]) -> str:
        chunks: list[str] = []
        current_length = 0
        for index, item in enumerate(contexts, start=1):
            text = item.document.text.strip()
            if not text:
                continue
            label = f"[Đoạn {index}] {text}"
            remaining = self.config.max_context_chars - current_length
            if remaining <= 0:
                break
            chunks.append(label[:remaining])
            current_length += len(chunks[-1])
        return "\n\n".join(chunks)

    @staticmethod
    def _clean_answer(answer: str) -> str:
        value = answer.strip().strip("\"'“”‘’")
        value = re.sub(r"^Câu trả lời(?: nguyên văn)?:\s*", "", value, flags=re.IGNORECASE)
        if "KHÔNG TÌM THẤY" in value.upper():
            return "KHÔNG TÌM THẤY"
        return value.strip()

    @staticmethod
    def _is_verbatim_answer(answer: str, context_text: str) -> bool:
        if not answer or answer == "KHÔNG TÌM THẤY":
            return False
        return answer in context_text

    def _heuristic_extract(self, question: str, contexts: list[RetrievalResult]) -> str:
        question_terms = set(tokenize(question))
        if not question_terms:
            return "KHÔNG TÌM THẤY"

        best_sentence = ""
        best_score = 0.0
        for result in contexts:
            for sentence in self._split_sentences(result.document.text):
                terms = set(tokenize(sentence))
                if not terms:
                    continue
                overlap = len(question_terms & terms)
                score = overlap / max(len(question_terms), 1)
                score += min(result.score, 1.0) * 0.05
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if best_score <= 0:
            return "KHÔNG TÌM THẤY"
        return best_sentence[:500].strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
        return [piece.strip() for piece in pieces if piece.strip()]
