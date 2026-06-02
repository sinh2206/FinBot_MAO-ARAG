from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag_engine.schema import Document
from tools.file_loader import clean_text, load_directory
from tools.text_splitter import ChunkConfig, split_documents


@dataclass(slots=True)
class DocumentProcessorConfig:
    chunk_size: int = 384
    chunk_overlap_ratio: float = 0.2
    recursive: bool = True


class DocumentProcessor:
    def __init__(self, config: DocumentProcessorConfig | None = None) -> None:
        self.config = config or DocumentProcessorConfig()

    def load(self, data_dir: str | Path) -> list[Document]:
        loaded = load_directory(data_dir, recursive=self.config.recursive)
        return [item.to_document() for item in loaded]

    def clean(self, documents: Iterable[Document]) -> list[Document]:
        cleaned = []
        for doc in documents:
            cleaned.append(
                Document(
                    id=doc.id,
                    text=clean_text(doc.text),
                    metadata=doc.metadata,
                )
            )
        return cleaned

    def chunk(self, documents: Iterable[Document]) -> list[Document]:
        return split_documents(
            documents,
            ChunkConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap_ratio=self.config.chunk_overlap_ratio,
            ),
        )

    def process(self, data_dir: str | Path) -> list[Document]:
        return self.chunk(self.clean(self.load(data_dir)))
