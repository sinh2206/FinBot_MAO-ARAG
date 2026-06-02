from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.reranker_agent import RerankerAgent
from agents.retriever_agent import RetrieverAgent
from core.orchestrator import Orchestrator
from rag_engine.retriever import HybridRetriever, HybridRetrieverConfig
from rag_engine.schema import Document


def test_orchestrator_returns_answer_payload() -> None:
    documents = [
        Document(id="price", text="VNINDEX dong cua o 1.280 diem."),
        Document(id="volume", text="Thanh khoan dat 18.000 ty dong."),
    ]
    retriever = HybridRetriever(
        documents=documents,
        config=HybridRetrieverConfig(mode="sparse_only"),
    )
    orchestrator = Orchestrator(
        planner_agent=PlannerAgent(enable_llm=False),
        retriever_agent=RetrieverAgent(retriever=retriever),
        reranker_agent=RerankerAgent(enable_model=False),
        executor_agent=ExecutorAgent(enable_model=False),
    )

    result = orchestrator.run("VNINDEX dong cua bao nhieu diem?")

    assert "answer" in result
    assert "plan" in result
    assert "1.280 diem" in result["answer"]
