from agents.executor_agent import ExecutorAgent
from rag_engine.schema import Document, RetrievalResult


def test_executor_heuristic_extracts_sentence_from_context() -> None:
    executor = ExecutorAgent(enable_model=False)
    contexts = [
        RetrievalResult(
            document=Document(id="doc", text="VNINDEX dong cua o 1.280 diem. Thanh khoan tang."),
            score=1.0,
        )
    ]

    answer = executor.answer("VNINDEX dong cua bao nhieu diem?", contexts)

    assert "1.280 diem" in answer.answer
    assert answer.metadata["executor"] == "heuristic"
