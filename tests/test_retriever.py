from rag_engine.retriever import HybridRetriever, HybridRetrieverConfig
from rag_engine.schema import Document
from tools.search_utils import bm25_search


def test_sparse_retriever_finds_relevant_document() -> None:
    documents = [
        Document(id="a", text="Co phieu ngan hang tang manh."),
        Document(id="b", text="Gia dau the gioi giam nhe."),
    ]
    retriever = HybridRetriever(
        documents=documents,
        config=HybridRetrieverConfig(mode="sparse_only"),
    )

    results = retriever.search("ngan hang tang", top_k=1)

    assert results[0].document.id == "a"


def test_bm25_search_utility() -> None:
    documents = [
        Document(id="a", text="VNINDEX tang diem."),
        Document(id="b", text="USD giam gia."),
    ]

    results = bm25_search("VNINDEX", documents, top_k=1)

    assert results[0].document.id == "a"
