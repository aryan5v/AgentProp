"""Persistent control sessions and shared analysis cache."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentprop.core import AgentGraph, GraphAnalysisCache
from agentprop.ml.risk_predictors import LearnedRiskState
from agentprop.rl import CategoryBanditRoutingPolicy
from agentprop.runtime import ControlSession

_SHARED_ANALYSIS_CACHE: dict[str, GraphAnalysisCache] = {}


def _resolve_store_root(root: str | Path | None) -> Path:
    if root is not None:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        return path
    env = os.environ.get("AGENTPROP_SESSION_DIR")
    if env:
        path = Path(env)
        path.mkdir(parents=True, exist_ok=True)
        return path
    for candidate in (
        Path.home() / ".agentprop" / "sessions",
        Path.cwd() / ".agentprop_sessions",
    ):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    fallback = Path.cwd() / ".agentprop_sessions"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def shared_analysis_cache_for(graph: AgentGraph) -> GraphAnalysisCache | None:
    """Return a cross-session cache entry keyed by graph fingerprint, if warm."""

    fingerprint = graph.analysis_fingerprint()
    if fingerprint is None:
        return None
    return _SHARED_ANALYSIS_CACHE.get(fingerprint)


def warm_shared_analysis_cache(graph: AgentGraph) -> str:
    """Snapshot graph analysis cache into the shared store; return fingerprint."""

    fingerprint = graph.analysis_fingerprint() or graph.warm_analysis_cache()
    if fingerprint:
        _SHARED_ANALYSIS_CACHE[fingerprint] = graph.export_analysis_cache()
    return fingerprint


@dataclass(slots=True)
class PersistedSessionRecord:
    """Serializable control-session metadata."""

    session_id: str
    task_id: str
    category: str
    workflow_name: str
    graph_fingerprint: str
    summary: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "category": self.category,
            "workflow_name": self.workflow_name,
            "graph_fingerprint": self.graph_fingerprint,
            "summary": self.summary,
            "outcome": self.outcome,
        }


class SessionStore:
    """File-backed store for control sessions and learned routing state."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = _resolve_store_root(root)
        self._live_sessions: dict[str, ControlSession] = {}
        self._records: dict[str, PersistedSessionRecord] = {}
        self._bandit = CategoryBanditRoutingPolicy(
            arms=("agentprop_controller", "baseline", "quality-aware-greedy"),
            epsilon=0.05,
            default_arm="agentprop_controller",
        )
        self._risk_state = LearnedRiskState()
        self._load_state()

    def start_session(
        self,
        *,
        workflow: str | AgentGraph,
        task_id: str,
        category: str = "general",
        token_budget: int | None = None,
        wall_time_budget_s: float | None = None,
        baseline_tokens: int | None = None,
    ) -> tuple[str, ControlSession]:
        session = ControlSession.start(
            workflow,
            task_id=task_id,
            category=category,
            token_budget=token_budget,
            wall_time_budget_s=wall_time_budget_s,
            baseline_tokens=baseline_tokens,
        )
        warm_shared_analysis_cache(session.graph)
        session_id = str(uuid4())
        self._live_sessions[session_id] = session
        record = PersistedSessionRecord(
            session_id=session_id,
            task_id=task_id,
            category=category,
            workflow_name=session.workflow_name,
            graph_fingerprint=session.graph.analysis_fingerprint() or "",
            summary=session.summary(),
        )
        self._records[session_id] = record
        self._persist_record(record)
        return session_id, session

    def get_session(self, session_id: str) -> ControlSession:
        if session_id not in self._live_sessions:
            raise KeyError(f"unknown session: {session_id}")
        return self._live_sessions[session_id]

    def finish_session(
        self,
        session_id: str,
        *,
        passed: bool,
        strategy: str = "agentprop_controller",
        quality_score: float | None = None,
        regression_risk: float = 0.0,
    ) -> dict[str, object]:
        session = self.get_session(session_id)
        features = session.tracker.features(
            token_budget=session.config.token_budget,
            wall_time_budget_s=session.config.wall_time_budget_s,
        )
        outcome = session.record_outcome(
            passed=passed,
            strategy=strategy,
            quality_score=quality_score,
            regression_risk=regression_risk,
        )
        self._risk_state.update_from_outcome(
            category=session.config.category,
            passed=passed,
            quality_score=quality_score,
            elapsed_s=features.elapsed_s,
            wall_time_budget_s=session.config.wall_time_budget_s,
        )
        if isinstance(outcome, dict):
            token_savings = float(outcome.get("token_savings", 0.0) or 0.0)
            timeout_risk = self._risk_state.timeout_adjustment(session.config.category)
            self._bandit.update(
                session.config.category,
                strategy,
                passed=passed,
                token_savings=token_savings,
                quality_score=quality_score,
                regression_risk=regression_risk,
                timeout_risk=timeout_risk,
            )
        record = self._records[session_id]
        record.summary = session.summary()
        record.outcome = outcome if isinstance(outcome, dict) else None
        self._persist_record(record)
        self._persist_learned_state()
        return outcome if isinstance(outcome, dict) else {"passed": passed}

    @property
    def bandit(self) -> CategoryBanditRoutingPolicy:
        return self._bandit

    @property
    def risk_state(self) -> LearnedRiskState:
        return self._risk_state

    def list_sessions(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records.values()]

    def _persist_record(self, record: PersistedSessionRecord) -> None:
        path = self.root / f"{record.session_id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")

    def _persist_learned_state(self) -> None:
        bandit_path = self.root / "bandit.json"
        bandit_path.write_text(
            json.dumps(
                {
                    "arms": self._bandit.arms,
                    "epsilon": self._bandit.epsilon,
                    "default_arm": self._bandit.default_arm,
                    "stats": {
                        category: {
                            arm: {"count": stats.count, "value": stats.value}
                            for arm, stats in arms.items()
                        }
                        for category, arms in self._bandit.stats.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        self._risk_state.save(self.root / "risk_state.json")

    def _load_state(self) -> None:
        bandit_path = self.root / "bandit.json"
        if bandit_path.exists():
            data = json.loads(bandit_path.read_text())
            from agentprop.rl.bandit import BanditArmStats

            self._bandit = CategoryBanditRoutingPolicy(
                arms=tuple(data.get("arms", self._bandit.arms)),
                epsilon=float(data.get("epsilon", self._bandit.epsilon)),
                default_arm=data.get("default_arm", self._bandit.default_arm),
            )
            for category, arms in dict(data.get("stats", {})).items():
                for arm, stats in arms.items():
                    self._bandit.stats.setdefault(category, {})[arm] = BanditArmStats(
                        count=int(stats.get("count", 0)),
                        value=float(stats.get("value", 0.0)),
                    )
        risk_path = self.root / "risk_state.json"
        if risk_path.exists():
            self._risk_state = LearnedRiskState.load(risk_path)
        for path in sorted(self.root.glob("*.json")):
            if path.name in {"bandit.json", "risk_state.json"}:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict) or "session_id" not in payload:
                continue
            record = PersistedSessionRecord(
                session_id=str(payload["session_id"]),
                task_id=str(payload.get("task_id", "")),
                category=str(payload.get("category", "general")),
                workflow_name=str(payload.get("workflow_name", "")),
                graph_fingerprint=str(payload.get("graph_fingerprint", "")),
                summary=dict(payload.get("summary", {})),
                outcome=payload.get("outcome"),
            )
            self._records[record.session_id] = record
