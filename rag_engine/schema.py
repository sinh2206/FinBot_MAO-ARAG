from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "có", "co"}:
            return True
        if normalized in {"false", "no", "0", "không", "khong"}:
            return False
    return bool(value)


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any, index: int | None = None) -> "Document":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(id=str(index or 0), text=value)
        if isinstance(value, Mapping):
            doc_id = value.get("id", index if index is not None else 0)
            text = value.get("text") or value.get("content") or value.get("page_content")
            if text is None:
                raise ValueError("Document mapping must contain text/content/page_content")
            metadata = dict(value.get("metadata") or {})
            for key, val in value.items():
                if key not in {"id", "text", "content", "page_content", "metadata"}:
                    metadata[key] = val
            return cls(id=str(doc_id), text=str(text), metadata=metadata)
        raise TypeError(f"Unsupported document type: {type(value)!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    document: Document
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["document"] = self.document.to_dict()
        return payload


@dataclass(slots=True)
class SubQuery:
    id: str
    query: str
    type: str = "retrieval_qa"
    depends_on: list[str] = field(default_factory=list)
    tool: str | None = "retriever"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], index: int = 1) -> "SubQuery":
        query = value.get("query") or value.get("question") or value.get("sub_query")
        if not query:
            raise ValueError("Sub-query must contain query/question/sub_query")
        depends_on = value.get("depends_on") or value.get("dependencies") or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        return cls(
            id=str(value.get("id") or f"q{index}"),
            query=str(query),
            type=str(value.get("type") or "retrieval_qa"),
            depends_on=[str(item) for item in depends_on],
            tool=value.get("tool", "retriever"),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowPlan:
    original_query: str
    sub_queries: list[SubQuery]
    strategy: str = "sequential"
    requires_retrieval: bool = True
    requires_execution: bool = True
    aggregation_mode: str = "concat"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], original_query: str) -> "WorkflowPlan":
        strategy = str(value.get("strategy") or value.get("execution") or "sequential").lower()
        if "parallel" in strategy:
            strategy = "parallel"
        elif "sequential" in strategy:
            strategy = "sequential"
        else:
            strategy = "sequential"

        raw_sub_queries = (
            value.get("sub_queries")
            or value.get("queries")
            or value.get("steps")
            or [{"id": "q1", "query": original_query}]
        )
        if isinstance(raw_sub_queries, Mapping) or isinstance(raw_sub_queries, str):
            raw_sub_queries = [raw_sub_queries]

        sub_queries: list[SubQuery] = []
        for i, item in enumerate(raw_sub_queries):
            if isinstance(item, Mapping):
                sub_queries.append(SubQuery.from_dict(item, index=i + 1))
            elif isinstance(item, str) and item.strip():
                sub_queries.append(SubQuery(id=f"q{i + 1}", query=item.strip()))
        if not sub_queries:
            sub_queries = [SubQuery(id="q1", query=original_query)]

        return cls(
            original_query=original_query,
            sub_queries=sub_queries,
            strategy=strategy,
            requires_retrieval=_coerce_bool(value.get("requires_retrieval", value.get("use_retriever")), True),
            requires_execution=_coerce_bool(value.get("requires_execution", value.get("use_executor")), True),
            aggregation_mode=str(value.get("aggregation_mode") or value.get("aggregation") or "concat"),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sub_queries"] = [item.to_dict() for item in self.sub_queries]
        return payload


@dataclass(slots=True)
class Answer:
    query: str
    answer: str
    contexts: list[RetrievalResult] = field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contexts"] = [context.to_dict() for context in self.contexts]
        return payload
