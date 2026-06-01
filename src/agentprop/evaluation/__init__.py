"""Evaluation metrics and recommendation helpers."""

from agentprop.evaluation.llm_execution import (
    LLMExecutionResult,
    LLMUsage,
    OpenAICompatibleChatClient,
    openai_compatible_env_status,
)
from agentprop.evaluation.metrics import (
    CostSummary,
    PruningRiskSummary,
    QualityCostSummary,
    RecommendationReport,
    RobustnessSummary,
    compare_routing,
    quality_cost_summary,
    robustness_under_failures,
)
from agentprop.evaluation.pruning import (
    PruningEvaluation,
    evaluate_pruning,
    summarize_pruning_risk,
)
from agentprop.evaluation.quality import (
    ExactMatchScorer,
    HumanLabelScorer,
    LLMJudgeScorer,
    QualityScore,
    QualityScorer,
    RubricScorer,
    aggregate_quality_scores,
)
from agentprop.evaluation.readiness import (
    ReadinessItem,
    ReadinessReport,
    build_v1_readiness_report,
    render_v1_readiness_markdown,
)
from agentprop.evaluation.reporting import (
    render_html_report,
    render_markdown_report,
    report_to_dict,
    write_report,
)
from agentprop.evaluation.runner import BenchmarkRow, run_benchmark
from agentprop.evaluation.verification import (
    VerificationResult,
    VerificationStatus,
    run_verification_command,
    verification_row_fields,
)

__all__ = [
    "BenchmarkRow",
    "CostSummary",
    "ExactMatchScorer",
    "HumanLabelScorer",
    "LLMJudgeScorer",
    "LLMExecutionResult",
    "LLMUsage",
    "OpenAICompatibleChatClient",
    "PruningEvaluation",
    "PruningRiskSummary",
    "QualityCostSummary",
    "QualityScore",
    "QualityScorer",
    "RecommendationReport",
    "ReadinessItem",
    "ReadinessReport",
    "RobustnessSummary",
    "RubricScorer",
    "VerificationResult",
    "VerificationStatus",
    "aggregate_quality_scores",
    "build_v1_readiness_report",
    "compare_routing",
    "evaluate_pruning",
    "quality_cost_summary",
    "openai_compatible_env_status",
    "render_html_report",
    "render_markdown_report",
    "render_v1_readiness_markdown",
    "report_to_dict",
    "robustness_under_failures",
    "run_verification_command",
    "run_benchmark",
    "summarize_pruning_risk",
    "verification_row_fields",
    "write_report",
]
