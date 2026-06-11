"""Traffic splitting, shadow mode, and canary rollups for controller arms."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True, slots=True)
class TrafficSplit:
    """Deterministic hash-based run assignment across controller arms."""

    weights: Mapping[str, float]
    default_arm: str = "baseline"

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("TrafficSplit requires at least one arm")
        total = sum(max(0.0, weight) for weight in self.weights.values())
        if total <= 0:
            raise ValueError("TrafficSplit weights must sum to a positive value")
        if self.default_arm not in self.weights:
            raise ValueError("default_arm must be present in weights")

    def assign(self, run_id: str) -> str:
        """Assign a run to an arm using a stable hash in [0, 1)."""

        total = sum(max(0.0, weight) for weight in self.weights.values())
        bucket = _hash_fraction(run_id) * total
        cumulative = 0.0
        for arm, weight in self.weights.items():
            cumulative += max(0.0, weight)
            if bucket <= cumulative:
                return arm
        return self.default_arm


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One run outcome used in traffic-split rollups."""

    arm: str
    passed: bool
    tokens: int
    cost: float
    false_local_pass: bool = False
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ArmRollup:
    """Aggregate metrics for one arm."""

    arm: str
    runs: int
    pass_rate: float
    mean_tokens: float
    mean_cost: float
    false_local_pass_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "runs": self.runs,
            "pass_rate": self.pass_rate,
            "mean_tokens": self.mean_tokens,
            "mean_cost": self.mean_cost,
            "false_local_pass_count": self.false_local_pass_count,
        }


@dataclass(frozen=True, slots=True)
class TrafficSplitReport:
    """Per-arm comparison report."""

    rollups: tuple[ArmRollup, ...]
    circuit_breaker_tripped: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollups": [rollup.to_dict() for rollup in self.rollups],
            "circuit_breaker_tripped": list(self.circuit_breaker_tripped),
        }


class ShadowMode:
    """Compute candidate controller decisions without enforcing them."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        *,
        run_id: str,
        arm: str,
        live_action: str,
        shadow_action: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a live-vs-shadow decision comparison."""

        row = {
            "run_id": run_id,
            "arm": arm,
            "live_action": live_action,
            "shadow_action": shadow_action,
            "enforced": not self.enabled,
            "metadata": dict(metadata or {}),
        }
        self.rows.append(row)
        return row


def rollup_arms(
    results: list[ArmResult] | tuple[ArmResult, ...],
    *,
    min_pass_rate: float | None = None,
    min_window: int = 20,
    default_arm: str = "baseline",
) -> TrafficSplitReport:
    """Aggregate pass rate, cost, and circuit-breaker status per arm."""

    by_arm: dict[str, list[ArmResult]] = {}
    for result in results:
        by_arm.setdefault(result.arm, []).append(result)

    rollups: list[ArmRollup] = []
    tripped: list[str] = []
    for arm, rows in sorted(by_arm.items()):
        pass_rate = sum(1 for row in rows if row.passed) / len(rows)
        rollup = ArmRollup(
            arm=arm,
            runs=len(rows),
            pass_rate=pass_rate,
            mean_tokens=mean(row.tokens for row in rows),
            mean_cost=mean(row.cost for row in rows),
            false_local_pass_count=sum(1 for row in rows if row.false_local_pass),
        )
        rollups.append(rollup)
        if (
            min_pass_rate is not None
            and arm != default_arm
            and len(rows) >= min_window
            and pass_rate < min_pass_rate
        ):
            tripped.append(arm)
    return TrafficSplitReport(rollups=tuple(rollups), circuit_breaker_tripped=tuple(tripped))


def _hash_fraction(run_id: str) -> float:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)
