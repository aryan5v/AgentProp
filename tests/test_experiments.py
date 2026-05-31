import json
from pathlib import Path

from experiments import (
    evaluate_ml_generalization,
    evaluate_routing_baselines,
    replay_rl_trajectory,
    run_benchmark,
    run_rl_routing,
    train_edge_pruning_scorer,
    train_learned_propagation,
    train_seed_scorer,
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


def test_train_seed_scorer_experiment_writes_model(tmp_path: Path) -> None:
    output = tmp_path / "model.json"

    exit_code = train_seed_scorer.main(
        ["--trials", "3", "--epochs", "3", "--out", str(output)]
    )

    assert exit_code == 0
    assert output.exists()


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


def test_train_edge_pruning_scorer_experiment_writes_model(tmp_path: Path) -> None:
    output = tmp_path / "edge_model.json"

    exit_code = train_edge_pruning_scorer.main(
        ["--epochs", "3", "--out", str(output)]
    )

    assert exit_code == 0
    assert output.exists()


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
    assert {"broadcast", "greedy", "celf", "message_passing_gnn", "reinforce", "ppo"}.issubset(
        policies
    )
    assert {"pairwise_ranker", "marginal_gain_regressor"}.issubset(policies)
    assert payload["summary"]["greedy"]["workflows"] == 2.0
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

    assert exit_code == 0
    assert output.exists()


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
        ]
    )

    payload = json.loads(output.read_text())

    assert exit_code == 0
    assert output.exists()
    assert payload[0]["policy"] == "ppo"
    assert "values" in payload[0]
