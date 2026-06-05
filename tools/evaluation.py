from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter


def normalize_answer(text: object) -> str:
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value).lower()
    value = value.replace("đ", "d")
    value = remove_accents(value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def exact_match(prediction: object, reference: object) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1_score(prediction: object, reference: object) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def remove_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
