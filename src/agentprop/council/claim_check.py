"""Quarantine unsupported claims before synthesis (the negative-weight defense).

DRACO penalizes false/unsupported claims with negative weights (down to −500).
Fusion's synthesizer has no mechanism to catch a confident-but-wrong panel
member. The Council scores each sub-answer's support risk and, when a
calibrated gate fires, drops or flags it before it can poison synthesis.

The risk signal is pluggable. The default is evidence-grounded: a sub-answer
with no citations backing a search-requiring sub-task is high risk. A
``ConformalRiskGate`` converts the risk into a drop/keep decision with a
guaranteed miss rate once calibrated on labeled outcomes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agentprop.council.model_pool import ModelResponse
from agentprop.council.planner import SubTask
from agentprop.ml.conformal import ConformalRiskGate

RiskFn = Callable[[SubTask, ModelResponse], float]


def evidence_support_risk(subtask: SubTask, response: ModelResponse) -> float:
    """Heuristic support risk in [0, 1]: higher = less trustworthy."""

    if not response.ok or not response.text.strip():
        return 1.0
    if subtask.needs_search and not response.citations:
        return 0.85  # a claim that needed sources but cited none
    if subtask.needs_search and len(response.citations) == 1:
        return 0.5
    return 0.15


@dataclass(frozen=True, slots=True)
class CheckedSubAnswer:
    """One sub-answer after claim checking."""

    subtask_id: str
    model: str
    text: str
    risk: float
    quarantined: bool
    citations: tuple[str, ...] = ()


@dataclass(slots=True)
class ClaimChecker:
    """Score sub-answer support risk and quarantine the unsupported ones."""

    risk_fn: RiskFn = field(default=evidence_support_risk)
    gate: ConformalRiskGate | None = None
    risk_threshold: float = 0.7
    """Fallback threshold used when no calibrated gate is provided."""

    def check(
        self,
        subtask: SubTask,
        response: ModelResponse,
    ) -> CheckedSubAnswer:
        risk = self.risk_fn(subtask, response)
        if self.gate is not None and self.gate.is_calibrated:
            quarantined = self.gate.should_flag(risk)
        else:
            quarantined = risk >= self.risk_threshold
        return CheckedSubAnswer(
            subtask_id=subtask.id,
            model=response.model,
            text=response.text,
            risk=risk,
            quarantined=quarantined,
            citations=response.citations,
        )
