"""Zero-API-cost simulation of Council vs ensemble vs single-model economics.

Before spending on live runs, we want the cost-vs-accuracy curve that justifies
the architecture. This models the three strategies analytically:

- **single** — one model answers the whole task. Cost = whole-task tokens at
  that model's price; accuracy = that model's competence.
- **ensemble** (Fusion shape) — every pool model answers the *whole* task, then
  a synthesizer fuses. Cost ≈ N × whole-task + synth; accuracy gets a synthesis
  lift plus a saturating diversity lift (matching Fusion's own ablation).
- **council** (decompose-assign) — the task splits into sub-tasks, each routed
  to the cheapest *capable* model. Cost ≈ Σ sub-task tokens (no redundant
  whole-task work) + synth; accuracy comes from per-sub-task competence,
  improved by claim-checking that removes unsupported sub-answers (the
  negative-weight defense), then synthesized.

Everything is seeded and deterministic. The point is the *shape* — Council
should sit below-and-right of ensemble on cost-vs-accuracy — not absolute
numbers; live DRACO runs supply those.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import fmean

from agentprop.evaluation.intervals import ConfidenceInterval, bootstrap_mean_interval


@dataclass(frozen=True, slots=True)
class SimModel:
    """A model's simulated competence and price."""

    name: str
    competence: float  # P(correct) on a sub-task at its tier, in [0, 1]
    price_per_ktok: float
    tier: int = 1


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Knobs for one simulated task population."""

    whole_task_ktok: float = 600.0
    """Tokens to answer the whole task once (deep research is heavy)."""
    subtask_ktok: float = 90.0
    """Tokens for one sub-task."""
    subtasks: int = 5
    synth_ktok: float = 30.0
    synthesis_lift: float = 0.12
    diversity_lift_cap: float = 0.05
    claim_check_recovery: float = 0.25
    """Avoided-penalty credit when a wrong sub-answer is quarantined before
    synthesis (it stops dragging the score; it does not become correct).
    Deliberately conservative — quarantining catches some, not most, errors."""


@dataclass(frozen=True, slots=True)
class StrategyOutcome:
    """One strategy's simulated cost and accuracy with bootstrap CIs."""

    strategy: str
    accuracy: ConfidenceInterval
    cost_usd: ConfidenceInterval
    mean_accuracy: float
    mean_cost_usd: float

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "accuracy": self.accuracy.to_dict(),
            "cost_usd": self.cost_usd.to_dict(),
            "mean_accuracy": self.mean_accuracy,
            "mean_cost_usd": self.mean_cost_usd,
        }


def _single_trial(model: SimModel, cfg: SimConfig) -> tuple[float, float]:
    cost = cfg.whole_task_ktok * model.price_per_ktok / 1000.0
    return model.competence, cost


def _ensemble_trial(
    pool: list[SimModel], synth: SimModel, cfg: SimConfig, rng: random.Random
) -> tuple[float, float]:
    base = max(m.competence for m in pool)
    diversity = min(cfg.diversity_lift_cap, 0.02 * (len(pool) - 1))
    accuracy = min(1.0, base + cfg.synthesis_lift + diversity)
    accuracy = max(0.0, accuracy + rng.gauss(0.0, 0.02))
    cost = sum(cfg.whole_task_ktok * m.price_per_ktok for m in pool) / 1000.0
    cost += cfg.synth_ktok * synth.price_per_ktok / 1000.0
    return min(1.0, accuracy), cost


def _council_trial(
    pool: list[SimModel], synth: SimModel, cfg: SimConfig, rng: random.Random
) -> tuple[float, float]:
    cheapest_by_tier: dict[int, SimModel] = {}
    for m in sorted(pool, key=lambda x: x.price_per_ktok):
        cheapest_by_tier.setdefault(m.tier, m)
    tiers = sorted({m.tier for m in pool})
    correct = 0.0
    cost = 0.0
    for _ in range(cfg.subtasks):
        required = rng.choice(tiers)
        capable = [m for m in pool if m.tier >= required]
        model = min(capable, key=lambda x: x.price_per_ktok)
        cost += cfg.subtask_ktok * model.price_per_ktok / 1000.0
        if rng.random() < model.competence:
            correct += 1.0
        else:
            # Wrong sub-answer: claim-checking recovers part of the penalty.
            correct += cfg.claim_check_recovery
    accuracy = correct / cfg.subtasks
    accuracy = min(1.0, accuracy + cfg.synthesis_lift * 0.5)
    cost += cfg.synth_ktok * synth.price_per_ktok / 1000.0
    return max(0.0, min(1.0, accuracy + rng.gauss(0.0, 0.02))), cost


def simulate_strategies(
    pool: list[SimModel],
    *,
    cfg: SimConfig | None = None,
    frontier: SimModel | None = None,
    synthesizer: SimModel | None = None,
    trials: int = 500,
    seed: int = 0,
) -> list[StrategyOutcome]:
    """Simulate single (budget + frontier), ensemble, and council strategies."""

    cfg = cfg or SimConfig()
    rng = random.Random(seed)
    synth = synthesizer or max(pool, key=lambda m: m.tier)
    budget = min(pool, key=lambda m: m.price_per_ktok)

    def collect(fn) -> StrategyOutcome:  # type: ignore[no-untyped-def]
        accs, costs = [], []
        for _ in range(trials):
            a, c = fn()
            accs.append(a)
            costs.append(c)
        return StrategyOutcome(
            strategy="",
            accuracy=bootstrap_mean_interval(accs, seed=seed),
            cost_usd=bootstrap_mean_interval(costs, seed=seed),
            mean_accuracy=fmean(accs),
            mean_cost_usd=fmean(costs),
        )

    outcomes = []
    single_budget = collect(lambda: _single_trial(budget, cfg))
    outcomes.append(_relabel(single_budget, "single-budget"))
    if frontier is not None:
        single_frontier = collect(lambda: _single_trial(frontier, cfg))
        outcomes.append(_relabel(single_frontier, "single-frontier"))
    ensemble = collect(lambda: _ensemble_trial(pool, synth, cfg, rng))
    outcomes.append(_relabel(ensemble, "ensemble"))
    council = collect(lambda: _council_trial(pool, synth, cfg, rng))
    outcomes.append(_relabel(council, "council"))
    return outcomes


def _relabel(outcome: StrategyOutcome, name: str) -> StrategyOutcome:
    return StrategyOutcome(
        strategy=name,
        accuracy=outcome.accuracy,
        cost_usd=outcome.cost_usd,
        mean_accuracy=outcome.mean_accuracy,
        mean_cost_usd=outcome.mean_cost_usd,
    )
