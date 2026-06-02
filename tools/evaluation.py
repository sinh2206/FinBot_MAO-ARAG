from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    overlap = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in common)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(expected_doc_ids: Iterable[str], retrieved_doc_ids: Iterable[str], k: int) -> float:
    expected = set(expected_doc_ids)
    if not expected:
        return 0.0
    retrieved = set(list(retrieved_doc_ids)[:k])
    return len(expected & retrieved) / len(expected)


@dataclass(slots=True)
class EvaluationResult:
    exact_match: float
    f1: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "exact_match": self.exact_match,
            "f1": self.f1,
            "count": self.count,
        }


def evaluate_qa_pairs(rows: Iterable[dict[str, str]]) -> EvaluationResult:
    items = list(rows)
    if not items:
        return EvaluationResult(exact_match=0.0, f1=0.0, count=0)
    em = [exact_match(item.get("prediction", ""), item.get("answer", "")) for item in items]
    f1 = [f1_score(item.get("prediction", ""), item.get("answer", "")) for item in items]
    return EvaluationResult(exact_match=sum(em) / len(em), f1=sum(f1) / len(f1), count=len(items))
