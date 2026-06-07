# Quality Cascade vs Independent Cascade

Deterministic routing comparison on built-in workflow templates: quality-aware
seeds with quality-cascade propagation vs PageRank/greedy seeds with
independent-cascade propagation.

## Reproduce

```bash
pip install -e ".[dev]"
python experiments/quality_cascade_vs_ic.py
```

No API keys required. Writes `results.json` in this directory.

## Arms

| Arm | Seeds | Propagation |
| --- | --- | --- |
| `qc_quality_aware` | quality-aware-greedy | quality-cascade |
| `ic_pagerank` | pagerank | independent-cascade |
| `ic_greedy` | greedy | independent-cascade |

Workflows: `chain`, `planner_coder_tester_reviewer`, `research_writer_verifier`,
`rag_pipeline`, `dense_graph`, `layered_pipeline`.

## Files

- [REPORT.md](REPORT.md) — summary and limitations
- [results.json](results.json) — per-workflow, per-arm metrics
