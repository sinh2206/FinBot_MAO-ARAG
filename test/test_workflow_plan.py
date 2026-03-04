from app.mao_core.planner_agent import WorkflowPlan


def test_workflow_plan_forces_guardrails_steps():
    plan = WorkflowPlan(steps=["retriever", "doc_ranker", "generator"])
    assert plan.steps == ["retriever", "doc_ranker", "strict_extractor", "generator", "fact_checker"]


def test_workflow_plan_defaults_to_guardrails():
    plan = WorkflowPlan(steps=[])
    assert plan.steps == ["strict_extractor", "generator", "fact_checker"]
