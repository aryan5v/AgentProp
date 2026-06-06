from agentprop.evaluation import build_v1_readiness_report, render_v1_readiness_markdown


def test_v1_maturity_report_covers_core_surface() -> None:
    report = build_v1_readiness_report()

    assert report.overall_score >= 0.85
    assert report.counts["stable"] >= 6
    assert "blocked" not in report.counts
    titles = {item.title for item in report.items}
    assert "Directed weighted workflow graph backbone" in titles
    assert "Public alpha packaging" in titles


def test_v1_maturity_markdown_is_public_safe() -> None:
    markdown = render_v1_readiness_markdown(build_v1_readiness_report())

    assert "Implementation Maturity" in markdown
    assert "Remaining:" not in markdown
    assert "Blockers" not in markdown
    assert "stay private" not in markdown.lower()
