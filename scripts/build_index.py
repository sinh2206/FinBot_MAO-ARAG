from __future__ import annotations

import argparse
import json
import os
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


def load_dotenv_value(name: str, default: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return default

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'")
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline FAISS/BM25 index from processed documents.")
    default_embedding_model = load_dotenv_value("EMBEDDING_MODEL_NAME", "models/embedder")
    default_planner_model = load_dotenv_value("PHI_MODEL_NAME", load_dotenv_value("PLANNER_MODEL_NAME", "models/phi"))
    default_primary_executor = load_dotenv_value("EXECUTOR_MODEL_NAME", "models/qwen")
    default_fallback_executor = load_dotenv_value("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    parser.add_argument(
        "--data_dir",
        default="data/processed_data",
        help="Folder containing processed .txt files or supported source documents.",
    )
    parser.add_argument("--chunk_dir", default="data/chunks", help="Output folder for chunk JSON.")
    parser.add_argument("--index_dir", default="data/index", help="Output folder for FAISS/BM25 index.")
    parser.add_argument("--embedding_dir", default="data/embeddings", help="Output folder for .npy embeddings.")
    parser.add_argument("--metadata_dir", default="data/metadata", help="Output folder for metadata JSON.")
    parser.add_argument(
        "--embedding_model",
        default=default_embedding_model,
        help="Local sentence-transformers folder or Hugging Face repo id. Default: EMBEDDING_MODEL_NAME or models/embedder.",
    )
    parser.add_argument("--device", default=None, help="Optional sentence-transformers device, e.g. cpu/cuda.")
    parser.add_argument("--chunk_size", type=int, default=384, help="Approximate token count per chunk.")
    parser.add_argument("--chunk_overlap_ratio", type=float, default=0.2)
    parser.add_argument(
        "--retrieval_mode",
        default=load_dotenv_value("RAG_RETRIEVAL_MODE", "hybrid"),
        choices=["hybrid", "dense_only", "sparse_only"],
        help="hybrid/dense_only require sentence-transformers. sparse_only builds BM25 without dense embeddings.",
    )
    parser.add_argument("--dense_weight", type=float, default=0.7)
    parser.add_argument("--sparse_weight", type=float, default=0.3)
    parser.add_argument(
        "--planner_model",
        default=default_planner_model,
        help="Planner model used by the downstream pipeline. Stored in metadata only. Default: PHI_MODEL_NAME/PLANNER_MODEL_NAME or models/phi.",
    )
    parser.add_argument(
        "--primary_executor_model",
        default=default_primary_executor,
        help="Primary executor model used by the downstream pipeline. Stored in metadata only.",
    )
    parser.add_argument(
        "--fallback_executor_model",
        default=default_fallback_executor,
        help="Fallback executor model used when Qwen does not answer. Stored in metadata only.",
    )
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_project_path(args.data_dir)
    chunk_dir = resolve_project_path(args.chunk_dir)
    index_dir = resolve_project_path(args.index_dir)
    embedding_dir = resolve_project_path(args.embedding_dir)
    metadata_dir = resolve_project_path(args.metadata_dir)

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

    embedding_model = None
    embeddings = None
    embedder = None
    vector_store = None
    if args.retrieval_mode in {"hybrid", "dense_only"}:
        embedding_model = resolve_embedding_model(args.embedding_model, args.local_files_only)
        if args.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        embedder = SentenceTransformerEmbedder(
            EmbedderConfig(
                model_name=str(embedding_model),
                device=args.device,
                local_files_only=args.local_files_only,
            )
        )
        try:
            embeddings = embedder.encode([doc.text for doc in chunks])
        except OSError as exc:
            raise RuntimeError(
                "Không load được embedding model ở chế độ offline. "
                f"Model hiện tại: {embedding_model}. "
                "Hãy kiểm tra thư mục models/embedder đã có config.json, modules.json, tokenizer và trọng số, "
                "hoặc chạy lại với --embedding_model <đường_dẫn_model_local>. "
                "Chỉ dùng --no-local_files_only khi server được phép tải model từ Hugging Face."
            ) from exc
        except RuntimeError as exc:
            raise RuntimeError(
                "Không load được sentence-transformers để tạo dense embeddings. "
                "Nếu lỗi liên quan torch.float8_e8m0fnu, môi trường đang bị lệch phiên bản transformers/torch. "
                "Cách nhanh nhất: chạy lại build index với --retrieval_mode sparse_only, "
                "hoặc cài lại dependency: pip install 'transformers<5' 'sentence-transformers<5'."
            ) from exc
        np.save(embedding_dir / "embeddings.npy", embeddings)

        vector_store = FaissVectorStore(dimension=embeddings.shape[1])
        vector_store.add_vectors(embeddings, chunks)

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        config=HybridRetrieverConfig(
            mode=args.retrieval_mode,
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
            "retrieval_mode": args.retrieval_mode,
            "embedding_model": str(embedding_model) if embedding_model is not None else None,
            "planner_model": args.planner_model,
            "primary_executor_model": args.primary_executor_model,
            "fallback_executor_model": args.fallback_executor_model,
            "local_files_only": args.local_files_only,
            "embedding_shape": list(embeddings.shape) if embeddings is not None else None,
            "index_dir": str(index_dir),
        },
    )

    print(f"Built index with {len(chunks)} chunks from {len(source_documents)} documents.")
    print(f"FAISS/BM25 index: {index_dir}")
    print(f"Chunks: {chunk_dir / 'chunks.json'}")
    if embeddings is not None:
        print(f"Embeddings: {embedding_dir / 'embeddings.npy'}")
    else:
        print("Embeddings: skipped because retrieval_mode=sparse_only")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_embedding_model(model_name: str | None, local_files_only: bool) -> Path | str:
    if not model_name:
        model_name = "models/embedder"

    model_path = Path(model_name)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if model_path.exists():
        return model_path

    if local_files_only:
        raise FileNotFoundError(
            "Không tìm thấy embedding model local. "
            f"Đường dẫn đã kiểm tra: {model_path}. "
            "Hãy đặt model vào models/embedder hoặc truyền --embedding_model <đường_dẫn_model_local>. "
            "Nếu muốn tải từ Hugging Face, chạy thêm --no-local_files_only."
        )

    return model_name


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
