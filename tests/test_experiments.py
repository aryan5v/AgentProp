from pathlib import Path

from experiments import (
    evaluate_ml_generalization,
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


def test_rl_routing_experiment_writes_trajectory(tmp_path: Path) -> None:
    output = tmp_path / "rl.json"

    exit_code = run_rl_routing.main(["--trials", "3", "--out", str(output)])

    assert exit_code == 0
    assert output.exists()
