from pathlib import Path

from experiments import run_benchmark, run_rl_routing, train_seed_scorer


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


def test_train_seed_scorer_experiment_writes_model(tmp_path: Path) -> None:
    output = tmp_path / "model.json"

    exit_code = train_seed_scorer.main(
        ["--trials", "3", "--epochs", "3", "--out", str(output)]
    )

    assert exit_code == 0
    assert output.exists()


def test_rl_routing_experiment_writes_trajectory(tmp_path: Path) -> None:
    output = tmp_path / "rl.json"

    exit_code = run_rl_routing.main(["--trials", "3", "--out", str(output)])

    assert exit_code == 0
    assert output.exists()
