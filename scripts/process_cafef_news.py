from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine.schema import Document
from tools.text_splitter import ChunkConfig, split_documents


TEXT_COLUMNS = {"Nội dung", "Noi dung", "content"}
TITLE_COLUMNS = {"Tiêu đề", "Tieu de", "title"}
DATE_COLUMNS = {"Thời gian", "Thoi gian", "date", "time"}
URL_COLUMNS = {"URL", "url"}
SENTIMENT_COLUMNS = {"positive", "neutral", "negative"}
PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}
IGNORED_EXTENSIONS = {".pt", ".pth", ".joblib", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CafeF CSV dumps into RAG and training datasets.")
    parser.add_argument(
        "--input_dir",
        default="cafef_news-20260211T204544Z-1-001",
        help="CafeF root folder or the nested cafef_news folder.",
    )
    parser.add_argument("--storage_dir", default="storage_rag/cafef_news")
    parser.add_argument("--documents_out", default="data/documents/cafef_news.jsonl")
    parser.add_argument("--chunks_out", default="data/chunks/cafef_news_chunks.json")
    parser.add_argument("--training_dir", default="data/training")
    parser.add_argument("--metadata_dir", default="data/metadata")
    parser.add_argument("--chunk_size", type=int, default=384)
    parser.add_argument("--chunk_overlap_ratio", type=float, default=0.2)
    parser.add_argument("--min_text_chars", type=int, default=20)
    parser.add_argument("--max_training_rows", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(Path(args.input_dir))
    documents = load_cafef_documents(input_dir, min_text_chars=args.min_text_chars)
    chunks = split_documents(
        documents,
        ChunkConfig(chunk_size=args.chunk_size, chunk_overlap_ratio=args.chunk_overlap_ratio),
    )

    documents_out = Path(args.documents_out)
    chunks_out = Path(args.chunks_out)
    training_dir = Path(args.training_dir)
    metadata_dir = Path(args.metadata_dir)
    storage_dir = Path(args.storage_dir)
    for path in [documents_out.parent, chunks_out.parent, training_dir, metadata_dir, storage_dir]:
        path.mkdir(parents=True, exist_ok=True)

    write_jsonl(documents_out, [doc.to_dict() for doc in documents])
    write_json(chunks_out, [doc.to_dict() for doc in chunks])
    storage_summary = write_storage_like_artifacts(storage_dir, chunks)
    qa_rows = build_extractive_qa(documents, max_rows=args.max_training_rows)
    planner_rows = build_planner_workflows(documents)
    write_jsonl(training_dir / "cafef_extractive_qa.jsonl", qa_rows)
    write_jsonl(training_dir / "cafef_planner_workflows.jsonl", planner_rows)

    summary = {
        "input_dir": str(input_dir),
        "documents": len(documents),
        "chunks": len(chunks),
        "extractive_qa_rows": len(qa_rows),
        "planner_workflow_rows": len(planner_rows),
        "tickers": storage_summary,
        "documents_out": str(documents_out),
        "chunks_out": str(chunks_out),
        "storage_dir": str(storage_dir),
    }
    write_json(metadata_dir / "cafef_processing_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve_input_dir(path: Path) -> Path:
    if (path / "cafef_news").is_dir():
        return path / "cafef_news"
    if path.is_dir():
        return path
    raise FileNotFoundError(path)


def load_cafef_documents(input_dir: Path, min_text_chars: int) -> list[Document]:
    documents: list[Document] = []
    seen: set[str] = set()
    for csv_path in sorted(input_dir.rglob("*.csv")):
        rows = read_csv_rows(csv_path)
        if not rows:
            continue
        header = rows[0]
        data_rows = rows[1:]
        kind = classify_csv(header)
        ticker = infer_ticker(csv_path)
        target_ticker = infer_target_ticker(csv_path)
        for row_number, row in enumerate(data_rows, start=2):
            payload = row_to_payload(header, row)
            document = payload_to_document(
                csv_path=csv_path,
                row_number=row_number,
                kind=kind,
                ticker=ticker,
                target_ticker=target_ticker,
                payload=payload,
            )
            if document is None or len(document.text) < min_text_chars:
                continue
            fingerprint = stable_hash(
                "|".join(
                    [
                        document.metadata.get("url", ""),
                        document.metadata.get("date", ""),
                        document.metadata.get("title", ""),
                        document.text,
                    ]
                )
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            documents.append(document)
    return documents


def read_csv_rows(path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                return list(csv.reader(fh))
        except Exception:
            continue
    return []


def classify_csv(header: list[str]) -> str:
    columns = set(header)
    if PRICE_COLUMNS & columns:
        return "price_feature"
    if has_url_column(header):
        return "news"
    if SENTIMENT_COLUMNS <= columns:
        return "daily_sentiment"
    return "table"


def has_url_column(header: list[str]) -> bool:
    return any(column in URL_COLUMNS for column in header)


def row_to_payload(header: list[str], row: list[str]) -> dict[str, str]:
    payload = {}
    for index, column in enumerate(header):
        payload[column] = row[index].strip() if index < len(row) else ""
    return payload


def payload_to_document(
    csv_path: Path,
    row_number: int,
    kind: str,
    ticker: str,
    target_ticker: str | None,
    payload: dict[str, str],
) -> Document | None:
    metadata = {
        "kind": kind,
        "ticker": ticker,
        "target_ticker": target_ticker or ticker,
        "source": str(csv_path),
        "row_number": row_number,
    }
    date = first_value(payload, DATE_COLUMNS)
    title = first_value(payload, TITLE_COLUMNS)
    url = first_value(payload, URL_COLUMNS)
    content = first_value(payload, TEXT_COLUMNS)
    if date:
        metadata["date"] = date
    if title:
        metadata["title"] = title
    if url:
        metadata["url"] = url

    if kind == "news":
        text = render_news_text(ticker, target_ticker, date, title, url, content, payload)
    elif kind == "daily_sentiment":
        text = render_daily_sentiment_text(ticker, target_ticker, date, payload)
    elif kind == "price_feature":
        text = render_price_feature_text(ticker, target_ticker, date, payload)
    else:
        text = render_table_text(ticker, target_ticker, date, payload)

    if not text.strip():
        return None
    doc_id = stable_hash(f"{csv_path}|{row_number}|{ticker}|{date}|{title}|{url}|{text}")[:24]
    metadata.update(extract_sentiment_metadata(payload))
    return Document(id=f"cafef_{doc_id}", text=text, metadata=metadata)


def render_news_text(
    ticker: str,
    target_ticker: str | None,
    date: str,
    title: str,
    url: str,
    content: str,
    payload: dict[str, str],
) -> str:
    parts = [f"Ticker: {ticker}"]
    if target_ticker and target_ticker != ticker:
        parts.append(f"Target ticker: {target_ticker}")
    if date:
        parts.append(f"Date: {date}")
    if title:
        parts.append(f"Title: {title}")
    if content:
        parts.append(f"Content: {content}")
    if SENTIMENT_COLUMNS <= set(payload):
        parts.append(render_sentiment(payload))
    if url:
        parts.append(f"URL: {url}")
    return "\n".join(parts)


def render_daily_sentiment_text(ticker: str, target_ticker: str | None, date: str, payload: dict[str, str]) -> str:
    parts = [f"Ticker: {ticker}"]
    if target_ticker and target_ticker != ticker:
        parts.append(f"Target ticker: {target_ticker}")
    if date:
        parts.append(f"Date: {date}")
    parts.append(render_sentiment(payload))
    article_count = payload.get("số_bài_báo") or payload.get("article_count") or payload.get("daily_article_count")
    if article_count:
        parts.append(f"Article count: {article_count}")
    dominant = payload.get("dominant_sentiment")
    if dominant:
        parts.append(f"Dominant sentiment: {dominant}")
    return "\n".join(part for part in parts if part)


def render_price_feature_text(ticker: str, target_ticker: str | None, date: str, payload: dict[str, str]) -> str:
    parts = [f"Ticker: {ticker}"]
    if target_ticker and target_ticker != ticker:
        parts.append(f"Target ticker: {target_ticker}")
    if date:
        parts.append(f"Date: {date}")
    for key in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "OBV",
        "return_t_plus_1",
        "return_t+1",
    ]:
        if payload.get(key):
            parts.append(f"{key}: {payload[key]}")
    sentiment_features = [(key, value) for key, value in payload.items() if "sentiment" in key.lower() and value]
    for key, value in sentiment_features[:20]:
        parts.append(f"{key}: {value}")
    return "\n".join(parts)


def render_table_text(ticker: str, target_ticker: str | None, date: str, payload: dict[str, str]) -> str:
    parts = [f"Ticker: {ticker}"]
    if target_ticker and target_ticker != ticker:
        parts.append(f"Target ticker: {target_ticker}")
    if date:
        parts.append(f"Date: {date}")
    for key, value in payload.items():
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def render_sentiment(payload: dict[str, str]) -> str:
    values = []
    for key in ("positive", "neutral", "negative", "sentiment_score", "daily_sentiment_score"):
        if payload.get(key):
            values.append(f"{key}={payload[key]}")
    return "Sentiment: " + ", ".join(values) if values else ""


def extract_sentiment_metadata(payload: dict[str, str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("positive", "neutral", "negative", "sentiment_score", "daily_sentiment_score", "dominant_sentiment"):
        if key in payload and payload[key] != "":
            metadata[key] = payload[key]
    return metadata


def first_value(payload: dict[str, str], candidates: Iterable[str]) -> str:
    for key in candidates:
        if payload.get(key):
            return payload[key]
    return ""


def infer_ticker(path: Path) -> str:
    stem = path.stem.upper()
    parent = path.parent.name.upper()
    for pattern in (r"^T_([A-Z]{2,5})", r"^CAFEF_([A-Z]{2,5})", r"^KET_QUA_SENTIMENT_([A-Z]{2,5})"):
        match = re.search(pattern, stem)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Z]{2,5}", stem):
        return stem
    match = re.match(r"([A-Z]{2,5})_RELATED_COM", parent)
    if match and not re.fullmatch(r".*_SENTIMENT", stem):
        if re.fullmatch(r"[A-Z]{2,5}", stem.replace("_SENTIMENT", "")):
            return stem.replace("_SENTIMENT", "")
        return match.group(1)
    tokens = re.findall(r"[A-Z]{2,5}", stem)
    ignored = {"CAFEF", "FINAL", "DAILY", "NEWS", "MODEL", "ALPHA", "FIXED", "WITH"}
    for token in tokens:
        if token not in ignored:
            return token
    return "MARKET"


def infer_target_ticker(path: Path) -> str | None:
    match = re.match(r"([A-Z]{2,5})_RELATED_COM", path.parent.name.upper())
    if match:
        return match.group(1)
    match = re.match(r"([A-Z]{2,5})_FINAL", path.parent.name.upper())
    if match:
        return match.group(1)
    return None


def write_storage_like_artifacts(storage_dir: Path, chunks: list[Document]) -> dict[str, int]:
    groups: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        ticker = str(chunk.metadata.get("ticker") or "MARKET").upper()
        groups[safe_name(ticker)].append(chunk)

    summary: dict[str, int] = {}
    for ticker, docs in sorted(groups.items()):
        target = storage_dir / ticker
        target.mkdir(parents=True, exist_ok=True)
        docstore_metadata = {}
        ref_doc_info = {}
        docstore_data = {}
        text_id_to_ref_doc_id = {}
        metadata_dict = {}
        for doc in docs:
            doc_hash = stable_hash(doc.text)
            docstore_metadata[doc.id] = {"doc_hash": doc_hash}
            ref_doc_info[doc.id] = {"node_ids": [doc.id], "metadata": doc.metadata}
            docstore_data[doc.id] = {
                "__data__": {
                    "id_": doc.id,
                    "embedding": None,
                    "metadata": doc.metadata,
                    "excluded_embed_metadata_keys": [],
                    "excluded_llm_metadata_keys": [],
                    "relationships": {},
                    "metadata_template": "{key}: {value}",
                    "metadata_separator": "\n",
                    "text": doc.text,
                    "mimetype": "text/plain",
                    "start_char_idx": 0,
                    "end_char_idx": len(doc.text),
                    "text_template": "{metadata_str}\n\n{content}",
                    "class_name": "TextNode",
                },
                "__type__": "1",
            }
            text_id_to_ref_doc_id[doc.id] = doc.id
            metadata_dict[doc.id] = doc.metadata

        write_json(
            target / "docstore.json",
            {
                "docstore/metadata": docstore_metadata,
                "docstore/ref_doc_info": ref_doc_info,
                "docstore/data": docstore_data,
            },
        )
        write_json(
            target / "default__vector_store.json",
            {
                "embedding_dict": {},
                "text_id_to_ref_doc_id": text_id_to_ref_doc_id,
                "metadata_dict": metadata_dict,
            },
        )
        write_json(target / "image__vector_store.json", {"embedding_dict": {}, "metadata_dict": {}})
        write_json(target / "graph_store.json", {"graph_dict": {}, "version": "cafef_processed_v1"})
        write_json(
            target / "index_store.json",
            {
                "index_store/data": {
                    f"cafef_{ticker}": {
                        "__type__": "cafef_text_index",
                        "__data__": {
                            "ticker": ticker,
                            "node_count": len(docs),
                            "source": "cafef_news",
                        },
                    }
                }
            },
        )
        summary[ticker] = len(docs)
    return summary


def build_extractive_qa(documents: list[Document], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc in documents:
        metadata = doc.metadata
        ticker = metadata.get("ticker", "MARKET")
        title = metadata.get("title")
        date = metadata.get("date")
        if title and title in doc.text:
            rows.append(
                {
                    "question": f"Tin CafeF về {ticker} ngày {date or 'không rõ ngày'} có tiêu đề gì?",
                    "context": doc.text,
                    "answer": title,
                    "source": metadata.get("source"),
                    "metadata": metadata,
                }
            )
        for key in ("positive", "neutral", "negative", "dominant_sentiment"):
            value = metadata.get(key)
            if value and str(value) in doc.text:
                rows.append(
                    {
                        "question": f"Chỉ số {key} của tin/dòng dữ liệu {ticker} là bao nhiêu?",
                        "context": doc.text,
                        "answer": str(value),
                        "source": metadata.get("source"),
                        "metadata": metadata,
                    }
                )
        if len(rows) >= max_rows:
            break
    return rows[:max_rows]


def build_planner_workflows(documents: list[Document]) -> list[dict[str, Any]]:
    counts = Counter(str(doc.metadata.get("ticker", "MARKET")).upper() for doc in documents)
    rows = []
    for ticker, count in counts.most_common(30):
        workflow = {
            "strategy": "sequential",
            "requires_retrieval": True,
            "requires_execution": True,
            "aggregation_mode": "concat",
            "sub_queries": [
                {
                    "id": "q1",
                    "query": f"Tóm tắt các tin CafeF liên quan đến {ticker}.",
                    "type": "retrieval_qa",
                    "depends_on": [],
                    "tool": "retriever",
                }
            ],
        }
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "Bạn là planner_agent, chỉ trả về workflow JSON hợp lệ."},
                    {"role": "user", "content": f"Tóm tắt tin tức mới nhất về cổ phiếu {ticker}."},
                    {"role": "assistant", "content": json.dumps(workflow, ensure_ascii=False)},
                ],
                "metadata": {"ticker": ticker, "document_count": count},
            }
        )
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "MARKET"


if __name__ == "__main__":
    main()
