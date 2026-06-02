from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.reranker_agent import RerankerAgent
from agents.retriever_agent import RetrieverAgent
from core.orchestrator import Orchestrator
from rag_engine.retriever import HybridRetriever, HybridRetrieverConfig
from rag_engine.schema import Document


def build_sparse_orchestrator() -> Orchestrator:
    documents = [
        Document(
            id="vnindex_close",
            text="VNINDEX đóng cửa ở 1.280 điểm, tăng 12 điểm so với phiên trước.",
        ),
        Document(
            id="liquidity",
            text="Thanh khoản thị trường đạt 18.000 tỷ đồng, cao hơn mức trung bình 20 phiên.",
        ),
        Document(
            id="banking",
            text="Nhóm ngân hàng dẫn dắt đà tăng với nhiều cổ phiếu vượt tham chiếu.",
        ),
    ]
    retriever = HybridRetriever(
        documents=documents,
        config=HybridRetrieverConfig(mode="sparse_only", output_top_k=10),
    )
    return Orchestrator(
        planner_agent=PlannerAgent(enable_llm=False),
        retriever_agent=RetrieverAgent(retriever=retriever),
        reranker_agent=RerankerAgent(enable_model=False),
        executor_agent=ExecutorAgent(enable_model=False),
    )


def test_sparse_orchestrator_answers_single_question() -> None:
    orchestrator = build_sparse_orchestrator()

    result = orchestrator.run("VNINDEX đóng cửa bao nhiêu điểm?")

    assert "1.280 điểm" in result["answer"]
    assert result["plan"]["strategy"] == "sequential"
    assert len(result["sub_answers"]) == 1


def test_sparse_orchestrator_handles_parallel_sub_queries() -> None:
    orchestrator = build_sparse_orchestrator()

    result = orchestrator.run("VNINDEX đóng cửa bao nhiêu điểm?; Thanh khoản thị trường thế nào?")

    assert "1.280 điểm" in result["answer"]
    assert "18.000 tỷ đồng" in result["answer"]
    assert result["plan"]["strategy"] == "parallel"
    assert len(result["sub_answers"]) == 2
