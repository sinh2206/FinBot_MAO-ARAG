from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine.embedder import EmbedderConfig, SentenceTransformerEmbedder
from rag_engine.retriever import BM25Index, HybridRetriever, HybridRetrieverConfig
from rag_engine.schema import Document
from rag_engine.vector_store import FaissVectorStore
from scripts.convert_raw_data import normalize_vietnamese_ocr_text
from tools.file_loader import load_directory
from tools.text_splitter import ChunkConfig, split_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline FAISS/BM25 index from processed documents.")
    parser.add_argument(
        "--data_dir",
        default="data/processed_data",
        help="Folder containing processed .txt files or supported source documents.",
    )
    parser.add_argument("--chunk_dir", default="data/chunks", help="Output folder for chunk JSON.")
    parser.add_argument("--index_dir", default="data/index", help="Output folder for FAISS/BM25 index.")
    parser.add_argument("--embedding_dir", default="data/embeddings", help="Output folder for .npy embeddings.")
    parser.add_argument("--metadata_dir", default="data/metadata", help="Output folder for metadata JSON.")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None, help="Optional sentence-transformers device, e.g. cpu/cuda.")
    parser.add_argument("--chunk_size", type=int, default=384, help="Approximate token count per chunk.")
    parser.add_argument("--chunk_overlap_ratio", type=float, default=0.2)
    parser.add_argument("--dense_weight", type=float, default=0.7)
    parser.add_argument("--sparse_weight", type=float, default=0.3)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    chunk_dir = Path(args.chunk_dir)
    index_dir = Path(args.index_dir)
    embedding_dir = Path(args.embedding_dir)
    metadata_dir = Path(args.metadata_dir)

    for directory in [chunk_dir, index_dir, embedding_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    loaded_files = load_directory(data_dir)
    source_documents = [normalize_document_text(item.to_document()) for item in loaded_files]
    if not source_documents:
        raise RuntimeError(f"No supported documents found in {data_dir}")

    chunks = split_documents(
        source_documents,
        ChunkConfig(chunk_size=args.chunk_size, chunk_overlap_ratio=args.chunk_overlap_ratio),
    )
    if not chunks:
        raise RuntimeError("No chunks were produced from the input documents")

    embedder = SentenceTransformerEmbedder(
        EmbedderConfig(
            model_name=args.embedding_model,
            device=args.device,
            local_files_only=args.local_files_only,
        )
    )
    embeddings = embedder.encode([doc.text for doc in chunks])
    np.save(embedding_dir / "embeddings.npy", embeddings)

    vector_store = FaissVectorStore(dimension=embeddings.shape[1])
    vector_store.add_vectors(embeddings, chunks)

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        config=HybridRetrieverConfig(
            mode="hybrid",
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
        ),
    )
    retriever.documents = chunks
    retriever.bm25 = BM25Index(chunks)
    retriever.save(index_dir)

    write_json(chunk_dir / "chunks.json", [doc.to_dict() for doc in chunks])
    write_json(
        metadata_dir / "documents.json",
        [
            {
                "id": doc.id,
                "metadata": doc.metadata,
                "text_length": len(doc.text),
            }
            for doc in source_documents
        ],
    )
    write_json(
        metadata_dir / "index_summary.json",
        {
            "source_count": len(source_documents),
            "chunk_count": len(chunks),
            "embedding_model": args.embedding_model,
            "embedding_shape": list(embeddings.shape),
            "index_dir": str(index_dir),
        },
    )

    print(f"Built index with {len(chunks)} chunks from {len(source_documents)} documents.")
    print(f"FAISS/BM25 index: {index_dir}")
    print(f"Chunks: {chunk_dir / 'chunks.json'}")
    print(f"Embeddings: {embedding_dir / 'embeddings.npy'}")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_document_text(document: Document) -> Document:
    return Document(
        id=document.id,
        text=normalize_vietnamese_ocr_text(document.text),
        metadata=document.metadata,
    )


if __name__ == "__main__":
    main()
