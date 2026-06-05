from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from rag_engine.schema import Document


TOKEN_PATTERN = re.compile(r"\S+", flags=re.UNICODE)


@dataclass(slots=True)
class ChunkConfig:
    chunk_size: int = 384
    chunk_overlap_ratio: float = 0.2
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", "? ", "! ", " "]
    )

    @property
    def overlap_tokens(self) -> int:
        if self.chunk_size <= 0:
            return 0
        return max(0, min(int(self.chunk_size * self.chunk_overlap_ratio), self.chunk_size - 1))


def split_documents(
    documents: Iterable[Document | str | dict],
    config: ChunkConfig | None = None,
) -> list[Document]:
    cfg = config or ChunkConfig()
    if cfg.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks: list[Document] = []
    for doc_index, value in enumerate(documents):
        document = Document.from_any(value, index=doc_index)
        text = document.text.strip()
        if not text:
            continue
        pieces = split_text(text, cfg)
        for chunk_index, piece in enumerate(pieces):
            metadata = {
                **document.metadata,
                "parent_id": document.id,
                "chunk_index": chunk_index,
                "chunk_count": len(pieces),
                "chunk_token_count": count_tokens(piece),
            }
            source_file = metadata.get("source_file") or metadata.get("file_name") or document.id
            chunk_id = f"{document.id}:chunk-{chunk_index:04d}"
            chunks.append(Document(id=chunk_id, text=piece, metadata={**metadata, "source_file": source_file}))
    return chunks


def split_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    cfg = config or ChunkConfig()
    paragraphs = _initial_segments(text, cfg.separators)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)
        if paragraph_tokens > cfg.chunk_size:
            flush_current(chunks, current)
            current = []
            current_tokens = 0
            chunks.extend(split_long_segment(paragraph, cfg))
            continue

        if current and current_tokens + paragraph_tokens > cfg.chunk_size:
            flush_current(chunks, current)
            current = overlap_tail(current, cfg.overlap_tokens)
            current_tokens = count_tokens("\n\n".join(current))

        current.append(paragraph)
        current_tokens += paragraph_tokens

    flush_current(chunks, current)
    return [chunk for chunk in chunks if chunk.strip()]


def split_long_segment(text: str, cfg: ChunkConfig) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return []
    step = max(1, cfg.chunk_size - cfg.overlap_tokens)
    chunks = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + cfg.chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + cfg.chunk_size >= len(tokens):
            break
    return chunks


def overlap_tail(parts: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    kept: list[str] = []
    total = 0
    for part in reversed(parts):
        tokens = count_tokens(part)
        if kept and total + tokens > overlap_tokens:
            break
        kept.append(part)
        total += tokens
        if total >= overlap_tokens:
            break
    return list(reversed(kept))


def flush_current(chunks: list[str], current: list[str]) -> None:
    if current:
        chunks.append("\n\n".join(part.strip() for part in current if part.strip()).strip())


def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _initial_segments(text: str, separators: list[str]) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if "\n\n" in separators:
        parts = re.split(r"\n\s*\n", normalized)
    else:
        parts = [normalized]

    segments: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if count_tokens(part) <= 512:
            segments.append(part)
        else:
            segments.extend(split_sentences(part))
    return segments


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]
