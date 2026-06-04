from agentprop.evaluation import build_v1_readiness_report, render_v1_readiness_markdown


def test_v1_readiness_is_private_alpha_not_public_ready() -> None:
    report = build_v1_readiness_report()

    assert report.overall_score >= 0.8
    assert report.alpha_ready is True
    assert report.public_ready is False
    assert "Real routed LLM case-study results" in report.blockers
    assert report.counts["missing"] == 0


def test_v1_readiness_markdown_names_validation_blocker() -> None:
    markdown = render_v1_readiness_markdown(build_v1_readiness_report())

    assert "Private alpha ready: yes" in markdown
    assert "Public ready: no" in markdown
    assert "OpenAI-compatible credentials" in markdown
