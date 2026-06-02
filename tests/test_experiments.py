import json
import shlex
import sys
from pathlib import Path

from experiments import (
    analyze_case_study,
    build_empirical_rows,
    evaluate_ml_generalization,
    evaluate_routing_baselines,
    replay_rl_trajectory,
    run_benchmark,
    run_case_study,
    run_experiment_suite,
    run_ml_rl_sweep,
    run_rl_routing,
    train_edge_pruning_scorer,
    train_learned_propagation,
    train_seed_scorer,
    train_torch_gnn,
)

from agentprop.evaluation import LLMExecutionResult, LLMUsage, load_artifact_registry
from agentprop.workflows import planner_coder_tester_reviewer


class _FakeCaseStudyExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMExecutionResult:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        prompt_tokens = max(1, len(system_prompt.split()) + len(user_prompt.split()))
        completion_tokens = 25
        return LLMExecutionResult(
            model="fake-model",
            prompt=user_prompt,
            response="Final answer: Bug fixed. Verification: pytest passed.",
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_s=0.2,
            raw_response={},
        )


def test_run_benchmark_experiment_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"

    exit_code = run_benchmark.main(
        [
            "--workflows",
            "planner_coder_tester_reviewer",
            "--budget",
            "2",
            "--trials",
            "3",
            "--out-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "savings_by_algorithm.svg").exists()


def test_run_experiment_suite_writes_dry_run_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "suite_manifest.json"
    artifact_root = tmp_path / "artifacts"

    exit_code = run_experiment_suite.main(
        [
            "--config",
            "configs/experiment_suites/ml_core.json",
            "--artifact-root",
            str(artifact_root),
            "--only",
            "ml_generalization_mlp",
            "--dry-run",
            "--manifest",
            str(manifest),
        ]
    )
    payload = json.loads(manifest.read_text())

    assert exit_code == 0
    assert payload["suite"] == "ml_core"
    assert payload["dry_run"]
    assert payload["artifact_root"] == str(artifact_root)
    assert payload["runs"][0]["id"] == "ml_generalization_mlp"
    assert payload["runs"][0]["output"].endswith("generalization_mlp.json")


def test_run_ml_rl_sweep_writes_manifest_registry_and_metrics(tmp_path: Path) -> None:
    artifact_root = tmp_path / "sweep_artifacts"
    manifest = tmp_path / "sweep_manifest.json"

    exit_code = run_ml_rl_sweep.main(
        [
            "--config",
            "configs/sweeps/ml_rl_smoke.json",
            "--artifact-root",
            str(artifact_root),
            "--only",
            "mlp_seed_lr",
            "--manifest",
            str(manifest),
        ]
    )
    payload = json.loads(manifest.read_text())
    records = load_artifact_registry(artifact_root / "model_registry" / "registry.json")

    assert exit_code == 0
    assert payload["sweep"] == "ml_rl_smoke"
    assert payload["run_count"] == 2
    assert payload["success_count"] == 2
    assert payload["best_runs"]["run_id"]
    assert all(Path(run["output"]).exists() for run in payload["runs"])
    assert any(record.kind == "ml-model" for record in records)
    assert any(record.kind == "metrics" for record in records)


def test_build_empirical_rows_converts_trace_and_outcomes(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.JSONL"
    outcomes_path = tmp_path / "results.json"
    output_path = tmp_path / "empirical_rows.json"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task-pass",
                        "policy": "quality_aware_greedy",
                        "events": [
                            {
                                "source": "task",
                                "target": "coder",
                                "full_context": True,
                                "token_cost": 1000,
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task-unlabeled",
                        "policy": "quality_aware_greedy",
                        "events": [{"source": "task", "target": "reviewer"}],
                    }
                ),
            ]
        )
        + "\n"
    )
    outcomes_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_id": "task-pass",
                        "policy": "quality_aware_greedy",
                        "verification_passed": True,
                        "token_cost": 1200,
                    }
                ]
            }
        )
    )

    exit_code = build_empirical_rows.main(
        [
            "--trace",
            str(trace_path),
            "--outcome-results",
            str(outcomes_path),
            "--out",
            str(output_path),
        ]
    )
    payload = json.loads(output_path.read_text())

    assert exit_code == 0
    assert payload["row_count"] == 1
    assert payload["trace_count"] == 2
    assert payload["outcome_count"] == 1
    assert payload["skipped_trace_count"] == 1
    assert payload["rows"][0]["task_id"] == "task-pass"
    assert payload["rows"][0]["selected_seeds"] == ["coder"]
    assert payload["rows"][0]["verification_passed"] is True
    assert payload["rows"][0]["token_cost"] == 1200


def test_run_ml_rl_sweep_dry_run_expands_grid(tmp_path: Path) -> None:
    manifest = tmp_path / "dry_sweep_manifest.json"

    exit_code = run_ml_rl_sweep.main(
        [
            "--config",
            "configs/sweeps/ml_rl_smoke.json",
            "--artifact-root",
            str(tmp_path / "dry_artifacts"),
            "--only",
            "ppo_episode_sweep",
            "--dry-run",
            "--manifest",
            str(manifest),
        ]
    )
    payload = json.loads(manifest.read_text())

    assert exit_code == 0
    assert payload["dry_run"]
    assert payload["run_count"] == 2
    assert all("--episodes" in run["argv"] for run in payload["runs"])


def test_run_case_study_writes_offline_artifacts(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "demo_bug",
                        "category": "bugfix",
                        "prompt": "Fix a demo bug.",
                        "expected": "Bug fixed.",
                        "verification_command": "pytest",
                        "min_coverage": 0.7,
                    },
                    {
                        "id": "demo_feature",
                        "category": "feature",
                        "prompt": "Add a demo feature.",
                        "expected": "Feature added.",
                        "verification_command": "pytest",
                        "min_coverage": 0.7,
                    },
                ]
            }
        )
    )
    output_dir = tmp_path / "case_study"

    exit_code = run_case_study.main(
        [
            "--tasks",
            str(tasks),
            "--trials",
            "2",
            "--episodes",
            "2",
            "--epochs",
            "2",
            "--out-dir",
            str(output_dir),
        ]
    )
    payload = json.loads((output_dir / "results.json").read_text())
    trace_lines = (output_dir / "traces.jsonl").read_text().strip().splitlines()

    assert exit_code == 0
    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "traces.jsonl").exists()
    assert payload["mode"] == "offline-simulated"
    assert payload["task_count"] == 2
    assert {
        "broadcast",
        "optimized_greedy",
        "quality_aware_greedy",
        "ml_message_passing",
        "rl_ppo",
    }.issubset(
        payload["summary"]
    )
    assert len(trace_lines) == 10
    assert all(
        "final" not in row["selected_seeds"]
        for row in payload["rows"]
        if row["policy"] == "broadcast"
    )


def test_run_case_study_preflight_writes_readiness_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "preflight"

    exit_code = run_case_study.main(
        [
            "--execution-mode",
            "llm",
            "--preflight",
            "--tasks",
            "benchmarks/case_study_tasks.json",
            "--trials",
            "2",
            "--episodes",
            "2",
            "--epochs",
            "2",
            "--out-dir",
            str(output_dir),
        ]
    )
    payload = json.loads((output_dir / "preflight.json").read_text())

    assert exit_code == 0
    assert payload["mode"] == "llm-routed"
    assert payload["task_count"] == 20
    assert payload["meets_target_task_count"]
    assert payload["total_task_policy_arms"] == 100
    assert {
        "broadcast",
        "optimized_greedy",
        "quality_aware_greedy",
        "ml_message_passing",
        "rl_ppo",
    }.issubset(
        payload["policies"]
    )
    assert "outputs.jsonl" in payload["expected_artifacts"]
    assert "llm_environment" in payload


def test_run_rl_routing_supports_graph_feature_policy(tmp_path: Path) -> None:
    output = tmp_path / "feature_policy.json"
    checkpoint_root = tmp_path / "registry"

    exit_code = run_rl_routing.main(
        [
            "--policy",
            "feature-policy",
            "--episodes",
            "2",
            "--trials",
            "2",
            "--max-steps",
            "2",
            "--out",
            str(output),
            "--registry-root",
            str(checkpoint_root),
            "--run-id",
            "feature-policy-smoke",
        ]
    )
    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert payload
    assert payload[0]["policy"] == "feature-policy"
    assert payload[0]["training"]["feature_count"] > 0
    assert payload[0]["feature_policy"]["feature_names"]
    assert Path(payload[0]["checkpoint_path"]).exists()


def test_analyze_case_study_writes_tables_and_plots(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    out_dir = tmp_path / "analysis"
    results.write_text(
        json.dumps(
            {
                "mode": "llm",
                "workflow": "planner_coder_tester_reviewer",
                "task_count": 2,
                "rows": [
                    {
                        "task_id": "a",
                        "policy": "broadcast",
                        "verification_passed": True,
                        "quality_score": 1.0,
                        "total_cost": 100.0,
                        "token_cost": 100.0,
                        "message_count": 4,
                        "efficiency_score": 0.9,
                    },
                    {
                        "task_id": "a",
                        "policy": "optimized_greedy",
                        "verification_passed": True,
                        "quality_score": 0.9,
                        "total_cost": 60.0,
                        "token_cost": 60.0,
                        "message_count": 2,
                        "efficiency_score": 0.85,
                    },
                    {
                        "task_id": "b",
                        "policy": "broadcast",
                        "verification_passed": True,
                        "quality_score": 1.0,
                        "total_cost": 120.0,
                        "token_cost": 120.0,
                        "message_count": 4,
                        "efficiency_score": 0.9,
                    },
                    {
                        "task_id": "b",
                        "policy": "optimized_greedy",
                        "verification_passed": True,
                        "quality_score": 0.95,
                        "total_cost": 72.0,
                        "token_cost": 72.0,
                        "message_count": 2,
                        "efficiency_score": 0.88,
                    },
                ],
            }
        )
    )

    exit_code = analyze_case_study.main(["--results", str(results), "--out-dir", str(out_dir)])
    analysis = json.loads((out_dir / "analysis.json").read_text())

    assert exit_code == 0
    assert (out_dir / "analysis.md").exists()
    assert (out_dir / "policy_comparison.csv").exists()
    assert (out_dir / "token_savings_by_policy.svg").exists()
    assert (out_dir / "quality_by_policy.svg").exists()
    assert analysis["acceptance"]["optimized_greedy"]["cost_reduction_at_least_20_percent"]
    assert "optimized_greedy" in (out_dir / "analysis.md").read_text()


def test_prompt_only_case_study_arm_is_marked_not_real_validation() -> None:
    task = run_case_study.CaseStudyTask(
        id="demo_bug",
        category="bugfix",
        prompt="Fix a demo bug.",
        expected="Bug fixed.",
        verification_command="pytest tests/test_cli.py",
        min_coverage=0.7,
    )

    executor = _FakeCaseStudyExecutor()

    row, trace, output = run_case_study._evaluate_prompt_only_task_arm(
        task,
        planner_coder_tester_reviewer(),
        policy_name="optimized_greedy",
        seeds=["planner", "tester"],
        executor=executor,
        trials=2,
        seed=0,
        max_tokens=100,
    )

    assert row["model"] == "fake-model"
    assert row["validation_kind"] == "llm-prompt-only-not-validation"
    assert row["routing_fidelity"] == "cosmetic_prompt_annotation"
    assert row["verification_passed"] is None
    assert row["total_llm_tokens"] > 0
    assert row["quality_passed"]
    assert trace["total_tokens"] == row["total_llm_tokens"]
    assert output["response"].startswith("Final answer")


def test_routed_case_study_arm_executes_multiple_workflow_nodes() -> None:
    task = run_case_study.CaseStudyTask(
        id="demo_bug",
        category="bugfix",
        prompt="Fix a demo bug.",
        expected="Bug fixed.",
        verification_command="pytest tests/test_cli.py",
        min_coverage=0.7,
    )
    executor = _FakeCaseStudyExecutor()

    row, trace, output = run_case_study._evaluate_routed_llm_task_arm(
        task,
        planner_coder_tester_reviewer(),
        policy_name="optimized_greedy",
        seeds=["planner", "tester"],
        executor=executor,
        trials=2,
        seed=0,
        max_tokens=100,
    )

    assert row["validation_kind"] == "llm-routed-multi-agent"
    assert row["routing_fidelity"] == "node_calls_and_context_scope"
    assert row["verification_status"] == "not_run"
    assert row["verification_passed"] is None
    assert len(executor.calls) >= 3
    assert output["node_executions"][-1]["node_id"] == "final"
    assert len(trace["events"]) == len(output["node_executions"])


def test_run_case_study_can_capture_real_verification_logs(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    verification_command = f"{shlex.quote(sys.executable)} -c 'print(\"verified\")'"
    tasks.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "demo_bug",
                        "category": "bugfix",
                        "prompt": "Fix a demo bug.",
                        "expected": "Bug fixed.",
                        "verification_command": verification_command,
                        "min_coverage": 0.7,
                    }
                ]
            }
        )
    )
    output_dir = tmp_path / "case_study_verified"

    original_client = run_case_study.OpenAICompatibleChatClient

    class FakeClientFactory:
        @classmethod
        def from_env(cls, **kwargs):  # type: ignore[no-untyped-def]
            return _FakeCaseStudyExecutor()

    run_case_study.OpenAICompatibleChatClient = FakeClientFactory
    try:
        exit_code = run_case_study.main(
            [
                "--execution-mode",
                "llm-routed",
                "--tasks",
                str(tasks),
                "--trials",
                "2",
                "--episodes",
                "2",
                "--epochs",
                "2",
                "--run-verification",
                "--verification-workdir",
                str(tmp_path),
                "--out-dir",
                str(output_dir),
            ]
        )
    finally:
        run_case_study.OpenAICompatibleChatClient = original_client

    payload = json.loads((output_dir / "results.json").read_text())
    log_lines = (output_dir / "verification_logs.jsonl").read_text().strip().splitlines()
    output_lines = (output_dir / "outputs.jsonl").read_text().strip().splitlines()

    assert exit_code == 0
    assert payload["summary"]["optimized_greedy"]["verification_observed_count"] == 1.0
    assert all(row["verification_passed"] for row in payload["rows"])
    assert len(log_lines) == 5
    assert "verified" in json.loads(log_lines[0])["stdout"]
    assert json.loads(output_lines[0])["verification"]["status"] == "passed"


def test_train_seed_scorer_experiment_writes_model(tmp_path: Path) -> None:
    output = tmp_path / "model.json"
    registry_root = tmp_path / "registry"

    exit_code = train_seed_scorer.main(
        [
            "--trials",
            "3",
            "--epochs",
            "3",
            "--out",
            str(output),
            "--registry-root",
            str(registry_root),
        ]
    )
    records = load_artifact_registry(registry_root / "registry.json")

    assert exit_code == 0
    assert output.exists()
    assert records[0].kind == "ml-model"
    assert Path(records[0].path).exists()


def test_train_seed_scorer_experiment_writes_pairwise_model(tmp_path: Path) -> None:
    output = tmp_path / "pairwise_model.json"

    exit_code = train_seed_scorer.main(
        [
            "--model",
            "pairwise",
            "--trials",
            "2",
            "--epochs",
            "2",
            "--l2-penalty",
            "0.01",
            "--out",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert output.exists()
    assert payload["model"] == "pairwise"
    assert payload["weights"]


def test_train_seed_scorer_experiment_writes_regression_model(tmp_path: Path) -> None:
    output = tmp_path / "regression_model.json"

    exit_code = train_seed_scorer.main(
        [
            "--model",
            "regression",
            "--trials",
            "2",
            "--epochs",
            "2",
            "--out",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert output.exists()
    assert payload["model"] == "regression"
    assert payload["weights"]


def test_train_seed_scorer_experiment_uses_empirical_rows(tmp_path: Path) -> None:
    rows = tmp_path / "empirical_rows.json"
    output = tmp_path / "empirical_model.json"
    rows.write_text(
        json.dumps(
            [
                {
                    "task_id": "task-pass",
                    "policy": "quality_aware_greedy",
                    "selected_seeds": ["coder", "tester"],
                    "context_allocations": {"coder": 1.0, "tester": 1.0},
                    "verification_passed": True,
                    "cost_adjusted_success": 0.8,
                },
                {
                    "task_id": "task-fail",
                    "policy": "greedy",
                    "selected_seeds": ["planner"],
                    "context_allocations": {"planner": 1.0, "coder": 0.25},
                    "verification_passed": False,
                },
                {
                    "task_id": "task-infra",
                    "policy": "greedy",
                    "selected_seeds": ["planner"],
                    "verification_passed": False,
                    "retry_recommended": True,
                },
            ]
        )
    )

    exit_code = train_seed_scorer.main(
        [
            "--model",
            "linear",
            "--empirical-results",
            str(rows),
            "--epochs",
            "2",
            "--l2-penalty",
            "0.01",
            "--out",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert payload["label_source"] == "empirical-outcome"
    assert payload["model"] == "linear"
    assert payload["l2_penalty"] == 0.01
    assert payload["weights"]


def test_train_seed_scorer_experiment_uses_empirical_verifier_rows(
    tmp_path: Path,
) -> None:
    rows = tmp_path / "empirical_verifier_rows.json"
    output = tmp_path / "empirical_verifier_model.json"
    rows.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_id": "task-pass",
                        "policy": "quality_aware_greedy",
                        "activated_verifiers": ["tester"],
                        "verification_passed": True,
                        "cost_adjusted_success": 0.75,
                    },
                    {
                        "task_id": "task-fail",
                        "policy": "greedy",
                        "activated_verifiers": ["reviewer"],
                        "verification_passed": False,
                    },
                ]
            }
        )
    )

    exit_code = train_seed_scorer.main(
        [
            "--model",
            "linear",
            "--task",
            "verifier",
            "--empirical-results",
            str(rows),
            "--epochs",
            "2",
            "--out",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert payload["task"] == "verifier"
    assert payload["label_source"] == "empirical-outcome"
    assert payload["weights"]


def test_train_torch_gnn_builds_empirical_seed_examples(tmp_path: Path) -> None:
    rows = tmp_path / "torch_empirical_rows.json"
    rows.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_id": "task-pass",
                        "policy": "quality_aware_greedy",
                        "selected_seeds": ["coder"],
                        "context_allocations": {"coder": 1.0},
                        "verification_passed": True,
                    },
                    {
                        "task_id": "task-fail",
                        "policy": "greedy",
                        "selected_seeds": ["planner"],
                        "context_allocations": {"planner": 1.0, "coder": 0.25},
                        "verification_passed": False,
                    },
                ]
            }
        )
    )

    examples, label_source = train_torch_gnn._build_training_examples(
        task="seed",
        workflow="planner_coder_tester_reviewer",
        empirical_results=rows,
        budget=2,
        trials=2,
    )

    assert label_source == "empirical-outcome"
    assert len(examples) == 2
    assert examples[0].labels["coder"] == 1.0
    assert examples[1].labels["planner"] == 0.0
    assert examples[0].edge_features is not None


def test_train_torch_gnn_builds_empirical_verifier_examples(tmp_path: Path) -> None:
    rows = tmp_path / "torch_empirical_verifier_rows.json"
    rows.write_text(
        json.dumps(
            [
                {
                    "task_id": "task-pass",
                    "activated_verifiers": ["tester"],
                    "verification_passed": True,
                },
                {
                    "task_id": "task-fail",
                    "activated_verifiers": ["reviewer"],
                    "verification_passed": False,
                },
            ]
        )
    )

    examples, label_source = train_torch_gnn._build_training_examples(
        task="verifier",
        workflow="planner_coder_tester_reviewer",
        empirical_results=rows,
        budget=2,
        trials=2,
    )

    assert label_source == "empirical-outcome"
    assert len(examples) == 2
    assert examples[0].labels["tester"] == 1.0
    assert examples[1].labels["reviewer"] == 0.0


def test_train_edge_pruning_scorer_experiment_writes_model(tmp_path: Path) -> None:
    output = tmp_path / "edge_model.json"
    registry_root = tmp_path / "edge_registry"

    exit_code = train_edge_pruning_scorer.main(
        ["--epochs", "3", "--out", str(output), "--registry-root", str(registry_root)]
    )
    records = load_artifact_registry(registry_root / "registry.json")

    assert exit_code == 0
    assert output.exists()
    assert records[0].metadata["epochs"] == 3


def test_train_edge_pruning_scorer_experiment_uses_empirical_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "edge_empirical_model.json"
    rows = tmp_path / "edge_rows.json"
    rows.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_id": "safe-prune",
                        "policy": "rl_ppo",
                        "pruned_edges": [["planner", "reviewer"]],
                        "verification_passed": True,
                    },
                    {
                        "task_id": "unsafe-prune",
                        "policy": "rl_ppo",
                        "pruned_edges": [["coder", "tester"]],
                        "verification_passed": False,
                    },
                ]
            }
        )
    )

    exit_code = train_edge_pruning_scorer.main(
        [
            "--empirical-results",
            str(rows),
            "--epochs",
            "2",
            "--l2-penalty",
            "0.02",
            "--out",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert payload["label_source"] == "empirical-outcome"
    assert payload["l2_penalty"] == 0.02
    assert payload["weights"]


def test_train_learned_propagation_experiment_writes_model(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        """
        {
          "events": [
            {"source": "planner", "target": "coder", "success": true, "token_cost": 10},
            {"source": "coder", "target": "tester", "success": true, "token_cost": 10}
          ]
        }
        """
    )
    output = tmp_path / "learned.json"

    exit_code = train_learned_propagation.main(
        ["--trace", str(trace), "--trials", "2", "--out", str(output)]
    )

    assert exit_code == 0
    assert output.exists()


def test_evaluate_ml_generalization_writes_results(tmp_path: Path) -> None:
    output = tmp_path / "generalization.json"

    exit_code = evaluate_ml_generalization.main(
        ["--trials", "2", "--epochs", "2", "--out", str(output)]
    )

    assert exit_code == 0
    assert output.exists()


def test_evaluate_routing_baselines_writes_comparison(tmp_path: Path) -> None:
    output = tmp_path / "routing_baselines.json"

    exit_code = evaluate_routing_baselines.main(
        [
            "--workflows",
            "chain,star",
            "--trials",
            "2",
            "--episodes",
            "2",
            "--epochs",
            "2",
            "--max-steps",
            "4",
            "--out",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    policies = {row["policy"] for row in payload["rows"]}

    assert exit_code == 0
    assert output.exists()
    assert {
        "broadcast",
        "greedy",
        "celf",
        "message_passing_gnn",
        "message_passing_gnn_routing_policy",
        "reinforce",
        "ppo",
        "feature_policy",
    }.issubset(policies)
    assert {"q_learning_expanded", "reinforce_expanded", "ppo_expanded"}.issubset(policies)
    assert {
        "mlp_routing_policy",
        "pairwise_ranker_routing_policy",
        "marginal_gain_regressor_routing_policy",
    }.issubset(policies)
    assert {"pairwise_ranker", "marginal_gain_regressor"}.issubset(policies)
    assert payload["summary"]["greedy"]["workflows"] == 2.0
    expanded_rows = [row for row in payload["rows"] if row["policy"].endswith("_expanded")]
    assert expanded_rows
    assert all("actions" in row for row in expanded_rows)
    assert all("reward_trace" in row for row in expanded_rows)
    assert all("control_reward" in row["reward_trace"][0] for row in expanded_rows)
    assert all("activated_verifiers" in row for row in expanded_rows)
    assert all("pruned_edges" in row for row in expanded_rows)
    assert all(
        "final" not in row["seeds"]
        for row in payload["rows"]
        if row["policy"] != "broadcast"
    )
    assert all(
        "node_5" not in row["seeds"]
        for row in payload["rows"]
        if row["workflow"] == "chain" and row["policy"] != "broadcast"
    )


def test_rl_routing_experiment_writes_trajectory(tmp_path: Path) -> None:
    output = tmp_path / "rl.json"

    exit_code = run_rl_routing.main(["--trials", "3", "--out", str(output)])
    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert output.exists()
    assert payload[0]["summary"]["total_reward"] != 0
    assert "cost_adjusted_success" in payload[0]["summary"]


def test_rl_routing_experiment_uses_empirical_reward_calibration(tmp_path: Path) -> None:
    output = tmp_path / "rl_calibrated.json"
    rows = tmp_path / "reward_rows.json"
    rows.write_text(
        json.dumps(
            [
                {
                    "verification_passed": True,
                    "token_cost": 1000,
                    "message_cost": 100,
                    "latency": 5,
                },
                {
                    "verification_passed": False,
                    "token_cost": 3000,
                    "message_cost": 500,
                    "latency": 25,
                },
            ]
        )
    )

    exit_code = run_rl_routing.main(
        [
            "--policy",
            "greedy",
            "--trials",
            "2",
            "--max-steps",
            "2",
            "--reward-calibration-rows",
            str(rows),
            "--out",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert payload[0]["reward_profile"]["source"] == "empirical"
    assert payload[0]["reward_profile"]["example_count"] == 2
    assert "success_profile" not in payload[0]
    assert payload[0]["trajectory"][0]["info"]["reward_target"] == "coverage"
    assert "cumulative_reward" in payload[0]["trajectory"][0]


def test_rl_routing_experiment_can_reward_expected_success(tmp_path: Path) -> None:
    output = tmp_path / "rl_success_calibrated.json"
    rows = tmp_path / "success_rows.json"
    rows.write_text(
        json.dumps(
            [
                {
                    "verification_passed": True,
                    "token_cost": 1000,
                    "message_cost": 100,
                    "latency": 5,
                    "context_allocations": {"coder": 1.0, "tester": 1.0},
                    "selected_seeds": ["coder"],
                },
                {
                    "verification_passed": False,
                    "token_cost": 900,
                    "message_cost": 100,
                    "latency": 5,
                    "context_allocations": {"coder": 0.25, "tester": 1.0},
                    "selected_seeds": ["planner"],
                },
            ]
        )
    )

    exit_code = run_rl_routing.main(
        [
            "--policy",
            "greedy",
            "--trials",
            "2",
            "--max-steps",
            "2",
            "--reward-calibration-rows",
            str(rows),
            "--out",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert payload[0]["success_profile"]["example_count"] == 2
    assert payload[0]["success_profile"]["node_context_penalties"]["coder"] == 1.0
    assert payload[0]["trajectory"][0]["info"]["reward_target"] == "expected_success"
    assert payload[0]["trajectory"][0]["expected_success"] is not None
    assert "final_expected_success" in payload[0]["summary"]


def test_replay_rl_trajectory_experiment_imports_exported_trajectory(tmp_path: Path) -> None:
    source = tmp_path / "rl.json"
    replayed = tmp_path / "replayed.json"
    run_rl_routing.main(
        [
            "--policy",
            "greedy",
            "--trials",
            "2",
            "--max-steps",
            "4",
            "--out",
            str(source),
        ]
    )

    exit_code = replay_rl_trajectory.main(
        [
            "--trajectory",
            str(source),
            "--workflow",
            "planner_coder_tester_reviewer",
            "--policy",
            "greedy",
            "--trials",
            "2",
            "--seed",
            "0",
            "--out",
            str(replayed),
        ]
    )
    payload = json.loads(replayed.read_text())

    assert exit_code == 0
    assert replayed.exists()
    assert payload["rows"][0]["workflow"] == "planner_coder_tester_reviewer"
    assert payload["rows"][0]["steps"]
    assert payload["rows"][0]["final_state"]["selected_seeds"]


def test_rl_routing_experiment_writes_reinforce_trajectory(tmp_path: Path) -> None:
    output = tmp_path / "rl_reinforce.json"

    exit_code = run_rl_routing.main(
        [
            "--policy",
            "reinforce",
            "--episodes",
            "3",
            "--trials",
            "2",
            "--max-steps",
            "5",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()


def test_rl_routing_experiment_writes_ppo_trajectory(tmp_path: Path) -> None:
    output = tmp_path / "rl_ppo.json"
    registry_root = tmp_path / "rl_registry"

    exit_code = run_rl_routing.main(
        [
            "--policy",
            "ppo",
            "--episodes",
            "3",
            "--trials",
            "2",
            "--max-steps",
            "5",
            "--out",
            str(output),
            "--registry-root",
            str(registry_root),
        ]
    )

    payload = json.loads(output.read_text())
    records = load_artifact_registry(registry_root / "registry.json")

    assert exit_code == 0
    assert output.exists()
    assert payload[0]["policy"] == "ppo"
    assert "values" in payload[0]
    assert payload[0]["checkpoint_path"]
    assert any(record.kind == "rl-policy" for record in records)
