from pathlib import Path

from rag_engine.document_processor import DocumentProcessor, DocumentProcessorConfig
from tools.evaluation import exact_match, f1_score


def test_document_processor_chunks_text_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "documents"
    data_dir.mkdir()
    (data_dir / "market.txt").write_text(
        "VNINDEX dong cua o 1.280 diem. Thanh khoan dat 18.000 ty dong.",
        encoding="utf-8",
    )
    processor = DocumentProcessor(DocumentProcessorConfig(chunk_size=12, chunk_overlap_ratio=0.2))

    chunks = processor.process(data_dir)

    assert chunks
    assert chunks[0].metadata["parent_id"].endswith("market.txt")


def test_evaluation_metrics() -> None:
    assert exact_match("VNINDEX tang", "vnindex tang") == 1.0
    assert f1_score("VNINDEX tang 12 diem", "VNINDEX tang") > 0.0
