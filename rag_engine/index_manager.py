from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_engine.document_processor import DocumentProcessor, DocumentProcessorConfig
from rag_engine.embedder import EmbedderConfig, SentenceTransformerEmbedder
from rag_engine.retriever import BM25Index, HybridRetriever, HybridRetrieverConfig
from rag_engine.schema import Document
from rag_engine.vector_store import FaissVectorStore


@dataclass(slots=True)
class IndexBuildConfig:
    data_dir: Path = Path("data/documents")
    chunk_dir: Path = Path("data/chunks")
    index_dir: Path = Path("data/index")
    embedding_dir: Path = Path("data/embeddings")
    metadata_dir: Path = Path("data/metadata")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None
    chunk_size: int = 384
    chunk_overlap_ratio: float = 0.2
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    local_files_only: bool = False


class IndexManager:
    def __init__(self, config: IndexBuildConfig | None = None) -> None:
        self.config = config or IndexBuildConfig()

    def build(self) -> HybridRetriever:
        cfg = self.config
        for directory in [cfg.chunk_dir, cfg.index_dir, cfg.embedding_dir, cfg.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        processor = DocumentProcessor(
            DocumentProcessorConfig(
                chunk_size=cfg.chunk_size,
                chunk_overlap_ratio=cfg.chunk_overlap_ratio,
            )
        )
        source_documents = processor.clean(processor.load(cfg.data_dir))
        chunks = processor.chunk(source_documents)
        if not chunks:
            raise RuntimeError(f"No chunks were produced from {cfg.data_dir}")

        embedder = SentenceTransformerEmbedder(
            EmbedderConfig(
                model_name=cfg.embedding_model,
                device=cfg.device,
                local_files_only=cfg.local_files_only,
            )
        )
        embeddings = embedder.encode([doc.text for doc in chunks])
        np.save(cfg.embedding_dir / "embeddings.npy", embeddings)

        vector_store = FaissVectorStore(dimension=embeddings.shape[1])
        vector_store.add_vectors(embeddings, chunks)
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            config=HybridRetrieverConfig(
                mode="hybrid",
                dense_weight=cfg.dense_weight,
                sparse_weight=cfg.sparse_weight,
            ),
        )
        retriever.documents = chunks
        retriever.bm25 = BM25Index(chunks)
        retriever.save(cfg.index_dir)

        self._write_json(cfg.chunk_dir / "chunks.json", [doc.to_dict() for doc in chunks])
        self._write_json(cfg.metadata_dir / "documents.json", [doc.to_dict() for doc in source_documents])
        self._write_json(
            cfg.metadata_dir / "index_summary.json",
            {
                "source_count": len(source_documents),
                "chunk_count": len(chunks),
                "embedding_model": cfg.embedding_model,
                "embedding_shape": list(embeddings.shape),
            },
        )
        return retriever

    def load(self) -> HybridRetriever:
        return HybridRetriever.load(self.config.index_dir)

    @staticmethod
    def load_chunks(path: str | Path = "data/chunks/chunks.json") -> list[Document]:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("chunks.json must contain a list of documents")
        return [Document.from_any(item, index=i) for i, item in enumerate(payload)]

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
