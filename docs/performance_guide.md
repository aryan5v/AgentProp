# AgentProp Performance Guide

This guide answers the question: **which algorithm should I use, and how fast is it?**

---

## Algorithm Decision Tree

```
Graph size (n = nodes)?
│
├─ n ≤ 15  →  greedy (CELF)
│             Exact, O(k·n·trials). Sub-millisecond on tiny graphs.
│             Always use unless you have strict latency requirements.
│
├─ 15 < n ≤ 60  →  rzf-centrality
│                   Near-greedy quality, 5–50× faster than exact greedy.
│                   Good default for agent workflows (most are 5–30 nodes).
│
└─ n > 60  →  imm (sketch-based)
              Approximation guarantee (1−1/e−ε). Sub-second even at n=200.
              Trades a small accuracy margin for near-linear scaling.
```

The CLI `--algorithm auto` applies this decision tree automatically.

---

## Propagation Model Choices

| Model | Best for | Notes |
| --- | --- | --- |
| `independent-cascade` | Default; correctness studies | Stochastic edge activation. Most-studied in literature. |
| `rzf` / `randomized-zero-forcing` | High-reliability graphs (reliability > 0.9) | Secondary result; prefer for graphs where most edges fire. |
| `quality-cascade` | Context-compression modeling | Models quality decay through long chains. Use for multi-hop RAG. |
| `linear-threshold` | Threshold-driven workflows | Node activates when weighted in-degree exceeds threshold. |
| `learned` | Calibrated from real traces | Requires `calibrate_graph_from_trace` to fit parameters first. |
| `bootstrap` | Dense fault-tolerant graphs | Percolation-style; activates when k-of-n neighbors are active. |

For most users: start with `independent-cascade`. Switch to `rzf` only if your graph
has uniformly high edge reliabilities (>0.9).

---

## Microbenchmark Results

Measured on a single core (no parallelism), `trials=50`, `budget k=2`.
Results are representative; your hardware will differ.

| Algorithm | Graph size | Nodes | Time (ms) |
| --- | --- | ---: | ---: |
| greedy | small | 5 | < 1 |
| greedy | small | 10 | ~4 |
| rzf-centrality | medium | 20 | ~40 |
| rzf-centrality | medium | 40 | ~230 |
| imm | large | 80 | ~300 |
| imm | large | 120 | ~260 |

**Key takeaways:**
- `greedy` is effectively free at n≤10.
- `rzf-centrality` is fast enough for interactive CLI use up to n≈40.
- `imm` scales sub-linearly at large n due to sketch amortization.

Run `python benchmarks/perf_micro.py` to reproduce on your hardware.

---

## Budget (`-k`) vs. Coverage Trade-off

| Budget k | Typical coverage gain | Recommendation |
| --- | --- | --- |
| 1 | baseline | Only for single critical node identification |
| 2 | +15–25% over k=1 | **Default — good coverage, minimal overhead** |
| 3 | +5–10% over k=2 | Useful for hub-spoke workflows |
| 4+ | Diminishing returns | Rarely worth the cost unless graph is very dense |

Use `agentprop benchmark <workflow> --budget 4` to see the full k-curve for your graph.

---

## Plugin Models

Third-party propagation models can be registered without forking:

```toml
# your_package/pyproject.toml
[project.entry-points."agentprop.propagation"]
my-model = "your_package.propagation:MyModel"
```

Then call `agentprop.propagation.load_plugins()` at startup to make `my-model`
available to `--model my-model` and `make_propagation_model("my-model")`.

See `src/agentprop/propagation/plugins.py` for the full API
(`register_plugin`, `load_plugins`, `list_plugins`, `get_plugin`).

---

## Tuning `--trials`

Increasing `--trials` reduces variance in coverage estimates at the cost of
proportionally more wall-clock time (all models scale linearly with trials).

| Trials | Use case |
| --- | --- |
| 20–50 | Interactive CLI, quick comparisons |
| 100 (default) | Balanced — good for most workflows |
| 500+ | High-confidence evidence reports (`docs/results/`) |

For final published numbers, use `--trials 500` or higher.
