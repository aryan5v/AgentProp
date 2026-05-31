from agentprop.evaluation import (
    ExactMatchScorer,
    HumanLabelScorer,
    LLMJudgeScorer,
    QualityScore,
    RubricScorer,
    aggregate_quality_scores,
)


def test_exact_match_scorer_normalizes_whitespace() -> None:
    score = ExactMatchScorer().score(expected="hello world", actual=" hello   world ")

    assert score.passed
    assert score.score == 1.0


def test_human_label_scorer_normalizes_label() -> None:
    score = HumanLabelScorer().from_label(4.0, rationale="good answer")

    assert score.score == 0.8
    assert score.passed
    assert score.metadata["raw_label"] == 4.0


def test_rubric_scorer_weights_criteria() -> None:
    scorer = RubricScorer({"correct": 0.7, "concise": 0.3})

    score = scorer.from_criteria({"correct": True, "concise": False})

    assert score.score == 0.7
    assert not score.passed


def test_llm_judge_scorer_uses_injected_judge() -> None:
    def judge(expected: str | None, actual: str, context: str | None) -> QualityScore:
        return QualityScore(
            score=0.9,
            method="custom",
            passed=True,
            rationale=f"expected={expected}, actual={actual}, context={context}",
        )

    score = LLMJudgeScorer(judge).score(expected="x", actual="x", context="demo")

    assert score.method == "llm-judge"
    assert score.passed


def test_aggregate_quality_scores_reports_pass_rate() -> None:
    aggregate = aggregate_quality_scores(
        [
            QualityScore(score=1.0, method="a", passed=True),
            QualityScore(score=0.5, method="b", passed=False),
        ]
    )

    assert aggregate.score == 0.75
    assert aggregate.metadata["pass_rate"] == 0.5
