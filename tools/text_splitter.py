from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from rag_engine.schema import Document


@dataclass(slots=True)
class ChunkConfig:
    chunk_size: int = 384
    chunk_overlap_ratio: float = 0.2

    @property
    def chunk_overlap(self) -> int:
        return int(self.chunk_size * self.chunk_overlap_ratio)


def tokenish_length(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def split_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    cfg = config or ChunkConfig()
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _fallback_split(text, cfg)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=tokenish_length,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def split_documents(documents: Iterable[Document], config: ChunkConfig | None = None) -> list[Document]:
    cfg = config or ChunkConfig()
    chunks: list[Document] = []
    for doc in documents:
        for index, chunk in enumerate(split_text(doc.text, cfg), start=1):
            chunks.append(
                Document(
                    id=f"{doc.id}::chunk_{index}",
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        "parent_id": doc.id,
                        "chunk_index": index,
                        "chunk_size": cfg.chunk_size,
                        "chunk_overlap": cfg.chunk_overlap,
                    },
                )
            )
    return chunks


def _fallback_split(text: str, config: ChunkConfig) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunk_size = max(config.chunk_size, 1)
    overlap = min(max(config.chunk_overlap, 0), chunk_size - 1)
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks
