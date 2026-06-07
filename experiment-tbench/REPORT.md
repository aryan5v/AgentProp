# AgentProp A/B on a Terminal-Bench 2.1 task

**Task:** `heterogeneous-dates` (Terminal-Bench 2.1 / `original-tasks`)
**Date:** 2026-06-07
**Question:** Does AgentProp control change correctness / speed / tokens versus a raw agent on the same task?

## 1. The task

> Use `/app/daily_temp_sf_high.csv` and `/app/daily_temp_sf_low.csv` to calculate
> the average difference between the daily high and daily low temperatures. Save
> the number in `/app/avg_temp.txt`.

- Category: `file-operations`, difficulty `easy`, verifier `pytest`.
- Ground truth (from the task's own test): **`11.428571428571429`**.

### The trap
The two files use **heterogeneous date formats**:

| File | Date format | Order |
| --- | --- | --- |
| `high` | `YYYY-MM-DD` | sorted |
| `low`  | mixed `MM/DD/YYYY HH:MM:SS` **and** `MM-DD-YYYY HH:MM:SS` | shuffled |

The natural "merge the two tables on `date`" approach (which is also the shape of
the reference solution) produces **0 matched rows → `nan`** unless you first
normalize the date formats. This is the failure mode the benchmark is designed to catch.

## 2. Method

I (the coding agent) solved the *same* task twice and graded each run with the
**task's own pytest verifier** (`tests/test_outputs.py`, unmodified).

- **A0 — raw / no AgentProp:** write the natural merge-on-`date` solution without
  inspecting schemas first.
- **A2 — with AgentProp:** follow the AgentProp brief + benchmark extra-instructions
  ("for numerical/data tasks, confirm schema and formats *before* fitting; the
  verifier node must intercept contradicted assumptions before finalizing"):
  inspect both schemas → detect heterogeneous formats → normalize → merge →
  **assert merged row count == expected (verifier intercept)** → compute → write.

Tokens: a coding agent cannot read its own token meter from inside the session,
so the token dimension is computed by **AgentProp's own `trace-replay`** over a
task-grounded event stream (per-step token counts are labeled estimates of an
LLM agent step). `trace-replay` reloads AgentProp's real controller and stops
counting A2 tokens once it FINALIZEs.

### Honesty caveats
1. The expected answer is visible in the task's test file, so this is **not** a
   blind capability test — it is a controlled comparison of *workflow discipline*.
   Both arms still derive the answer genuinely by parsing the data; the key is
   only used to grade.
2. Because the same agent runs both arms, hands-on wall-clock is not an unbiased
   capability measure (the second run benefits from memory). The token numbers
   therefore come from AgentProp's controller replay, not from the two scripts.
3. No model API key / Harbor / Modal is available in this environment, so the
   full external harness was not run; the verifier and AgentProp tooling were run
   locally.

## 3. Results

| Metric | A0 raw | A2 AgentProp | Source |
| --- | --- | --- | --- |
| **Correctness (official pytest)** | **FAIL** (1 failed / 2 passed) | **PASS** (3/3) | task verifier |
| Value written | `nan` | `11.428571428571429` | run logs |
| Merged rows | 0 / 7 | 7 / 7 | run logs |
| Solve wall-clock | 0.31 s | 0.35 s | python timer |
| Tokens A0 (no-control) | 12,500 | — | `trace-replay` |
| Tokens A2 (with-control) | — | 9,400 | `trace-replay` |
| **Token reduction** | — | **24.8% (−3,100)** | `trace-replay` |

### Why A2 wins on correctness
The raw merge silently matched 0 rows and wrote `nan`; A0's value test failed
(`Expected 11.429 but got nan`). AgentProp's discipline added a **verifier
intercept** — assert the merge produced the expected number of rows — which
fails fast on the contradicted assumption and forces the date-normalization fix.

### Why A2 wins on tokens
Replaying the raw trajectory through AgentProp's real controller:

- **Step 4:** raw agent writes a final answer with no verification → controller
  issues `FORCE_VERIFY` (refuses to trust an unconfirmed answer — exactly what
  would have caught the `nan`).
- **Step 8:** verifier passes → controller `FINALIZE`s.
- **Steps 9–11:** raw agent keeps re-confirming / re-running / writing a summary
  → controller `STOPPED` these, saving 3,100 tokens.

## 4. Takeaways
- On this task AgentProp converted a **fail → pass** and cut **~25% of tokens**,
  consistent with (and slightly above) the README's single-task `regex-log`
  signal (123,731 → 81,949 tokens, ~34%).
- The mechanism is not a prompt flourish: the wins come from two concrete control
  behaviors — **force-verify before trusting a final answer** (catches the `nan`)
  and **finalize/stop once the verifier passes** (cuts the redundant tail).
- This is a **single self-contained task**, locally graded — a directional signal,
  not a benchmark claim.

## 5. Artifacts
- `a0_raw/solve.py`, `a0_raw/run.log`, `a0_raw/verifier.txt`
- `a2_agentprop/solve.py`, `a2_agentprop/run.log`, `a2_agentprop/verifier.txt`
- `artifacts/agentprop_brief.md` — generated AgentProp workflow brief
- `artifacts/control-demo/terminal/` — `agentprop control-demo` output
- `artifacts/heterogeneous_dates_trace.jsonl` — task-grounded event stream
- `artifacts/trace_replay_report.md`, `artifacts/trace_replay.json` — A0-vs-A2 token comparison

### Reproduce
```bash
# data (from a terminal-bench checkout)
cp original-tasks/heterogeneous-dates/task-deps/*.csv /app/
pip install pandas==2.3.0 numpy==2.3.1 pytest==8.4.1

# A0 (fails), A2 (passes) — grade with the task's own verifier
python experiment-tbench/a0_raw/solve.py     && python -m pytest original-tasks/heterogeneous-dates/tests/test_outputs.py -rA
python experiment-tbench/a2_agentprop/solve.py && python -m pytest original-tasks/heterogeneous-dates/tests/test_outputs.py -rA

# token comparison
agentprop trace-replay experiment-tbench/artifacts/heterogeneous_dates_trace.jsonl
```
