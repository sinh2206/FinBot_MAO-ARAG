from app.executors.query_rewriter import DecomposedQueries


def test_decomposed_queries_accepts_queries_field():
    parsed = DecomposedQueries.model_validate({"queries": ["Gia FPT hom nay?"]})
    assert parsed.queries == ["Gia FPT hom nay?"]


def test_decomposed_queries_accepts_questions_alias():
    parsed = DecomposedQueries.model_validate({"questions": ["Gia FPT hom nay?"]})
    assert parsed.queries == ["Gia FPT hom nay?"]
