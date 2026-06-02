from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

from rag_engine.retriever import BM25Index, tokenize
from rag_engine.schema import Document, RetrievalResult


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def tfidf_scores(query: str, documents: Iterable[Document]) -> list[RetrievalResult]:
    docs = list(documents)
    if not docs:
        return []
    query_terms = tokenize(query)
    doc_tokens = [tokenize(doc.text) for doc in docs]
    doc_freq = Counter(term for tokens in doc_tokens for term in set(tokens))
    total_docs = len(docs)
    query_counts = Counter(query_terms)

    results = []
    for doc, tokens in zip(docs, doc_tokens):
        counts = Counter(tokens)
        score = 0.0
        for term, query_tf in query_counts.items():
            if counts[term] == 0:
                continue
            idf = math.log((1 + total_docs) / (1 + doc_freq[term])) + 1
            score += query_tf * counts[term] * idf
        if score > 0:
            results.append(RetrievalResult(document=doc, score=float(score), sparse_score=float(score)))
    results.sort(key=lambda item: item.score, reverse=True)
    return results


def bm25_search(query: str, documents: Iterable[Document], top_k: int = 10) -> list[RetrievalResult]:
    return BM25Index(documents).search(query, top_k=top_k)


def min_max_normalize(scores: Iterable[float]) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(score - low) / (high - low) for score in values]
