"""
Execution Engine: orchestrates executor calls based on the workflow plan.
"""

import inspect
import logging
from typing import Any, Awaitable, Callable, Dict

from app.mao_core.cost_manager import CostManager, track_latency
from app.mao_core.planner_agent import WorkflowPlan

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Workflow orchestrator.
    """

    def __init__(self):
        self.executors: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self.cost_manager = CostManager()

    def register_executor(self, name: str, func: Callable[..., Awaitable[Any]]):
        """Register an async executor."""
        self.executors[name] = func
        logger.info("Registered executor: %s", name)

    def _build_executor_kwargs(
        self, executor: Callable[..., Awaitable[Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build kwargs for mixed executor signatures.
        Supports common params: context, query, documents.
        """
        signature = inspect.signature(executor)
        kwargs: Dict[str, Any] = {}

        for param in signature.parameters.values():
            if param.name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                return {}
            if param.name == "context":
                kwargs["context"] = context
            elif param.name == "query":
                kwargs["query"] = context.get("query", "")
            elif param.name == "documents":
                kwargs["documents"] = (
                    context.get("documents")
                    or context.get("retriever")
                    or context.get("doc_ranker_input")
                    or []
                )
            elif param.name in context:
                kwargs[param.name] = context[param.name]

        return kwargs

    @track_latency
    async def execute_step(self, step_name: str, context: Dict[str, Any]) -> Any:
        """Execute one step."""
        if step_name not in self.executors:
            raise ValueError(f"Executor '{step_name}' not registered.")

        executor = self.executors[step_name]
        kwargs = self._build_executor_kwargs(executor, context)

        if kwargs:
            return await executor(**kwargs)
        return await executor(context)

    async def execute_plan(self, plan: WorkflowPlan, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a full workflow plan and return final context."""
        context = initial_context.copy()
        steps = plan.steps

        logger.info("Starting execution of plan: %s", steps)
        for step in steps:
            logger.info("Executing step: %s", step)
            try:
                result = await self.execute_step(step, context)
                context[step] = result
                logger.info("Step %s completed.", step)
            except Exception as exc:
                logger.exception("Step %s failed: %s", step, exc)
                context[f"{step}_error"] = str(exc)

        logger.info("Execution finished.")
        return context