from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.reranker_agent import RerankerAgent
from agents.retriever_agent import RetrieverAgent
from core.orchestrator import Orchestrator
from rag_engine.retriever import HybridRetriever, HybridRetrieverConfig
from rag_engine.schema import Document
from tools.evaluation import evaluate_qa_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the MAO-RAG pipeline on a QA file.")
    parser.add_argument("--qa_file", default="data/metadata/qa_eval.json")
    parser.add_argument("--chunks_file", default="data/chunks/chunks.json")
    parser.add_argument("--output_file", default="data/metadata/evaluation_results.json")
    parser.add_argument("--retrieval_mode", default="sparse_only", choices=["hybrid", "dense_only", "sparse_only"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.qa_file)
    documents = load_documents(args.chunks_file)
    retriever = HybridRetriever(
        documents=documents,
        config=HybridRetrieverConfig(mode=args.retrieval_mode),
    )
    orchestrator = Orchestrator(
        planner_agent=PlannerAgent(enable_llm=False),
        retriever_agent=RetrieverAgent(retriever=retriever),
        reranker_agent=RerankerAgent(enable_model=False),
        executor_agent=ExecutorAgent(enable_model=False),
    )

    predictions = []
    for row in rows:
        question = row.get("question") or row.get("query")
        if not question:
            continue
        result = orchestrator.run(question)
        predictions.append(
            {
                **row,
                "prediction": result["answer"],
                "plan": result["plan"],
            }
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = evaluate_qa_pairs(predictions)
    print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))
    print(f"Wrote predictions to {output_path}")


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("QA file must be a JSON list or JSONL file")
    return payload


def load_documents(path: str | Path) -> list[Document]:
    source = Path(path)
    if not source.exists():
        return [
            Document(id="demo_1", text="VNINDEX dong cua o 1.280 diem, tang 12 diem so voi phien truoc."),
            Document(id="demo_2", text="Thanh khoan thi truong dat 18.000 ty dong."),
        ]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Chunks file must contain a JSON list")
    return [Document.from_any(item, index=i) for i, item in enumerate(payload)]


if __name__ == "__main__":
    main()
