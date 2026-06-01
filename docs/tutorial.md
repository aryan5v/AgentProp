# Tutorial: Optimize A Planner-Coder-Tester Workflow

This walkthrough proves the main AgentProp loop end to end.

## 1. Install

```bash
python -m pip install -e .
```

## 2. Optimize Context Seeding

```bash
agentprop optimize benchmarks/workflows/planner_coder_tester_reviewer.json --budget 2 --trials 20
```

Expected shape:

```text
AgentProp Optimization Report
Recommended seeds: planner, tester
Coverage: 100.0%
Estimated savings: 20-35%
```

The exact savings may vary slightly by propagation model and trials.

## 3. Compare Algorithms

```bash
agentprop benchmark planner_coder_tester_reviewer --budget 2 --trials 20
```

Expected output columns:

```text
workflow,algorithm,model,seeds,coverage,expected_time,full_activation,savings
```

This lets you compare random, centrality, greedy, CELF, and cost-aware greedy methods across propagation models.

## 4. Generate A Report

```bash
agentprop report planner_coder_tester_reviewer --budget 2 --out reports/tutorial.md
agentprop report planner_coder_tester_reviewer --budget 2 --out reports/tutorial.html --format html
```

Expected artifact:

```text
reports/tutorial.md
reports/tutorial.html
```

The report includes:

- recommended seed nodes
- coverage
- propagation time
- broadcast vs optimized cost
- bottlenecks
- pruning candidates
- pruning target/risk estimate
- robustness under node and edge failure
- verifier candidates

## 5. Export A Graph

```bash
agentprop viz planner_coder_tester_reviewer --out reports/tutorial.dot
```

Render with Graphviz:

```bash
dot -Tpng reports/tutorial.dot -o reports/tutorial.png
```

## 6. Run Reproducible Experiments

```bash
python experiments/run_benchmark.py --trials 20 --out-dir results/tutorial_benchmark
python experiments/train_seed_scorer.py --trials 10 --epochs 20 --out results/tutorial_ml/model.json
python experiments/run_rl_routing.py --trials 10 --out results/tutorial_rl/trajectory.json
```

These produce saved artifacts that can be used for research tables and plots.
