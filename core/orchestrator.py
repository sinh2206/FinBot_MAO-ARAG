from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

from agents.aggregator_agent import AggregatorAgent
from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.reranker_agent import RerankerAgent
from agents.retriever_agent import RetrieverAgent
from rag_engine.schema import Answer, Document, RetrievalResult, SubQuery, WorkflowPlan


@dataclass(slots=True)
class OrchestratorConfig:
    retriever_top_k: int = 10
    executor_top_k: int = 5
    max_workers: int = 4


class Orchestrator:
    """Coordinates planner, retriever, reranker, executor and aggregator."""

    def __init__(
        self,
        planner_agent: PlannerAgent | None = None,
        retriever_agent: RetrieverAgent | None = None,
        reranker_agent: RerankerAgent | None = None,
        executor_agent: ExecutorAgent | None = None,
        aggregator_agent: AggregatorAgent | None = None,
        documents: Iterable[Document | str | dict] | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.planner_agent = planner_agent or PlannerAgent()
        self.retriever_agent = retriever_agent or RetrieverAgent(documents=documents)
        self.reranker_agent = reranker_agent or RerankerAgent()
        self.executor_agent = executor_agent or ExecutorAgent()
        self.aggregator_agent = aggregator_agent or AggregatorAgent(executor_agent=self.executor_agent)

    def run(self, question: str) -> dict:
        plan = self.planner_agent.plan(question)
        sub_answers = self._execute_plan(plan)
        final_answer = self.aggregator_agent.aggregate(question, sub_answers, plan)
        return {
            "answer": final_answer.answer,
            "final_answer": final_answer.to_dict(),
            "plan": plan.to_dict(),
            "sub_answers": [answer.to_dict() for answer in sub_answers],
        }

    def ask(self, question: str) -> dict:
        return self.run(question)

    def _execute_plan(self, plan: WorkflowPlan) -> list[Answer]:
        if not plan.requires_retrieval and not plan.requires_execution:
            return [
                Answer(
                    query=plan.original_query,
                    answer=self._direct_response(plan.original_query),
                    confidence=1.0,
                    metadata={"workflow": "direct"},
                )
            ]

        has_dependencies = any(sub_query.depends_on for sub_query in plan.sub_queries)
        if plan.strategy == "parallel" and not has_dependencies and len(plan.sub_queries) > 1:
            return self._execute_parallel(plan)
        return self._execute_sequential(plan)

    def _execute_sequential(self, plan: WorkflowPlan) -> list[Answer]:
        answers: list[Answer] = []
        by_id: dict[str, Answer] = {}
        for sub_query in plan.sub_queries:
            resolved = self._with_dependency_context(sub_query, by_id)
            answer = self._run_sub_query(resolved, plan)
            answer.metadata["sub_query_id"] = sub_query.id
            answers.append(answer)
            by_id[sub_query.id] = answer
        return answers

    def _execute_parallel(self, plan: WorkflowPlan) -> list[Answer]:
        answers_by_index: dict[int, Answer] = {}
        max_workers = min(self.config.max_workers, len(plan.sub_queries))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._run_sub_query, sub_query, plan): index
                for index, sub_query in enumerate(plan.sub_queries)
            }
            for future in as_completed(futures):
                index = futures[future]
                answer = future.result()
                answer.metadata["sub_query_id"] = plan.sub_queries[index].id
                answers_by_index[index] = answer
        return [answers_by_index[index] for index in sorted(answers_by_index)]

    def _run_sub_query(self, sub_query: SubQuery, plan: WorkflowPlan) -> Answer:
        passages: list[RetrievalResult] = []
        if plan.requires_retrieval and sub_query.tool in {None, "retriever"}:
            passages = self.retriever_agent.retrieve(sub_query.query, top_k=self.config.retriever_top_k)
            passages = self.reranker_agent.rerank(
                sub_query.query,
                passages,
                top_k=min(self.config.executor_top_k, len(passages)) or None,
            )

        if plan.requires_execution:
            return self.executor_agent.answer(sub_query.query, passages)

        snippets = [item.document.text for item in passages]
        answer = "\n\n".join(snippets) if snippets else "KHÔNG TÌM THẤY"
        return Answer(query=sub_query.query, answer=answer, contexts=passages, confidence=None)

    @staticmethod
    def _with_dependency_context(sub_query: SubQuery, completed: dict[str, Answer]) -> SubQuery:
        if not sub_query.depends_on:
            return sub_query
        dependency_answers = [
            completed[dependency].answer
            for dependency in sub_query.depends_on
            if dependency in completed and completed[dependency].answer
        ]
        if not dependency_answers:
            return sub_query
        enriched_query = (
            f"{sub_query.query}\n\n"
            "Ngữ cảnh từ bước trước: "
            + " ".join(dependency_answers)
        )
        return SubQuery(
            id=sub_query.id,
            query=enriched_query,
            type=sub_query.type,
            depends_on=sub_query.depends_on,
            tool=sub_query.tool,
            metadata=sub_query.metadata,
        )

    @staticmethod
    def _direct_response(question: str) -> str:
        normalized = question.strip().lower()
        if normalized.startswith(("chào", "hello", "hi")):
            return "Chào bạn, mình sẵn sàng hỗ trợ truy vấn dữ liệu chứng khoán."
        if normalized.startswith(("cảm ơn", "cam on")):
            return "Rất vui được hỗ trợ bạn."
        return "Câu hỏi này không cần truy xuất tài liệu."
