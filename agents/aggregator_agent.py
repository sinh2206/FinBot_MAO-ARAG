from __future__ import annotations

from dataclasses import dataclass

from rag_engine.schema import Answer, Document, RetrievalResult, WorkflowPlan


@dataclass(slots=True)
class AggregatorConfig:
    use_executor_for_synthesis: bool = False
    empty_answer: str = "KHÔNG TÌM THẤY"


class AggregatorAgent:
    """Aggregates sub-answers with a template, optionally delegating synthesis."""

    def __init__(self, config: AggregatorConfig | None = None, executor_agent: object | None = None) -> None:
        self.config = config or AggregatorConfig()
        self.executor_agent = executor_agent

    def aggregate(self, original_query: str, sub_answers: list[Answer], plan: WorkflowPlan) -> Answer:
        if not sub_answers:
            return Answer(query=original_query, answer=self.config.empty_answer, confidence=0.0)

        all_contexts = [context for answer in sub_answers for context in answer.contexts]
        valid_answers = [item for item in sub_answers if item.answer and item.answer != self.config.empty_answer]
        if not valid_answers:
            return Answer(query=original_query, answer=self.config.empty_answer, contexts=all_contexts, confidence=0.0)

        if len(valid_answers) == 1:
            only = valid_answers[0]
            return Answer(
                query=original_query,
                answer=only.answer,
                contexts=only.contexts,
                confidence=only.confidence,
                metadata={"aggregation": "single"},
            )

        if (
            self.config.use_executor_for_synthesis
            and self.executor_agent is not None
            and plan.aggregation_mode == "synthesize"
        ):
            synthetic_contexts = [
                RetrievalResult(
                    document=Document(id=f"answer_{index}", text=answer.answer, metadata={"sub_query": answer.query}),
                    score=answer.confidence or 0.0,
                )
                for index, answer in enumerate(valid_answers, start=1)
            ]
            synthesized = self.executor_agent.answer(original_query, synthetic_contexts)
            synthesized.metadata["aggregation"] = "executor_synthesis"
            return synthesized

        answer_text = "; ".join(answer.answer.rstrip(".") for answer in valid_answers) + "."
        confidences = [answer.confidence for answer in valid_answers if answer.confidence is not None]
        confidence = sum(confidences) / len(confidences) if confidences else None
        return Answer(
            query=original_query,
            answer=answer_text,
            contexts=all_contexts,
            confidence=confidence,
            metadata={"aggregation": "concat", "sub_answer_count": len(valid_answers)},
        )
