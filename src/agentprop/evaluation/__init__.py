"""Evaluation metrics and recommendation helpers."""

from agentprop.evaluation.metrics import (
    CostSummary,
    QualityCostSummary,
    RecommendationReport,
    RobustnessSummary,
    compare_routing,
    quality_cost_summary,
    robustness_under_failures,
)
from agentprop.evaluation.pruning import PruningEvaluation, evaluate_pruning
from agentprop.evaluation.quality import (
    ExactMatchScorer,
    HumanLabelScorer,
    LLMJudgeScorer,
    QualityScore,
    QualityScorer,
    RubricScorer,
    aggregate_quality_scores,
)
from agentprop.evaluation.reporting import render_markdown_report, report_to_dict, write_report
from agentprop.evaluation.runner import BenchmarkRow, run_benchmark

__all__ = [
    "BenchmarkRow",
    "CostSummary",
    "ExactMatchScorer",
    "HumanLabelScorer",
    "LLMJudgeScorer",
    "PruningEvaluation",
    "QualityCostSummary",
    "QualityScore",
    "QualityScorer",
    "RecommendationReport",
    "RobustnessSummary",
    "RubricScorer",
    "aggregate_quality_scores",
    "compare_routing",
    "evaluate_pruning",
    "quality_cost_summary",
    "render_markdown_report",
    "report_to_dict",
    "robustness_under_failures",
    "run_benchmark",
    "write_report",
]
