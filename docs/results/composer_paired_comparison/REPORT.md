# Composer 2.5 paired Terminal-Bench runs: full report

**Label:** live Harbor runs on Terminal Bench 2.1 — A0 (raw `cursor-cli`) vs A2 (`agentprop-cursor` control loop).  
**Model:** `cursor/composer-2.5` · **Harness:** Harbor + Docker · **Date:** 2026-06-07 – 2026-06-08

## Executive summary

We ran **six paired tasks** (plus one A2 rerun on `build-cython-ext`) comparing raw Cursor Composer 2.5 against the same model inside AgentProp’s plan-mode control loop. Both arms share the same API key, task images, and verifier.

| Outcome | Count (paired) | Notes |
|--------|----------------|-------|
| Both pass | 2 | `regex-log`, `nginx-request-logging` |
| A0 pass, A2 fail | 1 | `build-cython-ext` |
| Both fail | 3 | `path-tracing`, `torch-pipeline-parallelism`, `bn-fit-modify` |

**Where control helped:** On `path-tracing`, A2 reached **4/5 verifier checks** and **0.932 image similarity** in ~9 minutes of agent time while A0 timed out at 30 minutes with **0/5** and no artifacts. On `regex-log`, A2 passed with **less agent time** than A0.

**Where control hurt or broke even:** A2 was often **much slower and more expensive** (`nginx` 11× agent time; `bn-fit-modify` 9× cost) for the same Harbor outcome. Several A2 runs **crashed the inner loop** (`NonZeroAgentExitCodeError`) even when Harbor’s verifier passed (`nginx`). Harness integration — not the controller’s stopping logic — was the main failure mode.

---

## Experimental setup

| Parameter | Value |
|-----------|--------|
| **A0** | Harbor `cursor-cli` agent — one long `--yolo` session |
| **A2** | Harbor `agentprop-cursor` — plan-mode proposer + `ControlledTerminalLoop` |
| **Runner** | `experiments/run_composer_tbench_comparison.sh` |
| **Task source** | `~/.cache/harbor/tasks/packages/terminal-bench` (random pick; skips tasks with existing A0) |
| **Wheel** | Built to `/tmp/agentprop-wheels`, passed via `AGENTPROP_WHEEL_PATH` |
| **API key** | `benchmark-results/.env.local` → Harbor `--ae CURSOR_API_KEY=...` |
| **Demo viewer** | `experiments/refresh_demo_view.sh` → unified `demo-view/` for `harbor view -p 8080` |

Artifacts (gitignored): `benchmark-results/composer-comparison/{a0-cursor-cli,a2-agentprop,paired-report,demo-view}/`

---

## All runs

Latest job per task per arm. Times are **agent execution** seconds from Harbor `result.json`.

| Task | A0 reward | A0 agent time | A0 cost | A0 exception | A2 reward | A2 agent time | A2 cost | A2 proposals | A2 exception |
|------|----------:|--------------:|--------:|--------------|----------:|--------------:|--------:|-------------:|--------------|
| **regex-log** | 1.0 | 406s (~6.8m) | $0.173 | — | 1.0 | 263s (~4.4m) | — * | 7 † | NonZeroAgentExitCodeError |
| **path-tracing** | 0.0 | 1799s (30m timeout) | — | AgentTimeoutError | 0.0 | 557s (~9.3m) | — | — † | NonZeroAgentExitCodeError |
| **torch-pipeline-parallelism** | 0.0 | 900s (15m timeout) | — | AgentTimeoutError | 0.0 | 180s (~3m) | — | — † | NonZeroAgentExitCodeError |
| **nginx-request-logging** | 1.0 | 58s (~1m) | $0.037 | — | 1.0 | 644s (~10.7m) | $0.094 | 7 | NonZeroAgentExitCodeError |
| **build-cython-ext** | 1.0 | 665s (~11m) | $0.122 | — | 0.0 | 449s (~7.5m) ‡ | $0.060 | 5 ‡ | NonZeroAgentExitCodeError |
| **bn-fit-modify** | 0.0 | 162s (~2.7m) | $0.067 | — | 0.0 | 1763s (~29.4m) | $0.606 | 19 | NonZeroAgentExitCodeError |

\* Early A2 runs predated Harbor cost/token export; later runs write `agentprop-cursor-usage.json`.  
† Proposal counts from usage file where present; early crashes may omit telemetry.  
‡ A2 **rerun** after parse-resilience fixes (`2026-06-07__23-40-27`). First attempt (`23-32-15`): 2 proposals, $0.019, crashed immediately on parse.

### Per-task notes

**regex-log** — First end-to-end A2 success after Harbor install fixes. Both pass; A2 faster on agent wall clock. Harbor still flagged `NonZeroAgentExitCodeError` on A2 (inner process exit code ≠ 0 despite verifier pass).

**path-tracing** — Strongest control story. A0: 0/5 tests, no `image.c`. A2: 4/5 pass; only `test_image_similarity` failed (0.932 vs 0.99 required). Inner loop died on plan-mode integration error before timeout.

**torch-pipeline-parallelism** — Both fail. A0 hit agent timeout; A2 exited early (~3m) with inner-loop crash before meaningful progress.

**nginx-request-logging** — Both pass on verifier. A2 took **~11×** longer and **~2.5×** the cost for the same binary reward. Parse crash on later proposal; Harbor graded pass anyway.

**build-cython-ext** — A0 passes using system Python / global packages. A2 failed twice: first run (2 proposals) died on JSON/markdown parse; rerun (5 proposals) got further but still failed verifier — likely **venv-isolated Python** (`/opt/agentprop-venv`) missing system `pybind11`/Cython context the task expects.

**bn-fit-modify** — Both fail verifier. A0 failed quickly (~3m). A2 ran **19 proposals over ~29m** and spent **~9×** A0’s cost without improving outcome; crashed on inner-loop exit. Post-run paired summarize once failed on PyPI DNS when collecting report metadata.

---

## Fixes implemented to make runs possible

These were required before any meaningful A2 numbers existed.

### Harbor agent bootstrap (`harbor_agent.py`)

1. **`BaseInstalledAgent` + `exec_as_agent`** — Initial A2 runs crashed with `AttributeError: no attribute 'exec_as_agent'`.
2. **Container packages** — Added `git` to `apt-get install` (needed for pip VCS installs).
3. **`_agent_env()`** — Read `AGENTPROP_WHEEL_PATH` and install vars from Harbor `--ae` (`self._extra_env`), not only host `os.environ`. Without this, container installed PyPI `agentprop` (no `agentprop.benchmarks`).
4. **Minimal wheel upload** — Stage only `src/` + `pyproject.toml` instead of full repo (large `jobs/` tree caused upload timeout).
5. **Venv Python path** — Run `cursor_terminal_agent` via `/opt/agentprop-venv/bin/python` after install.
6. **`AGENTPROP_HARBOR_LOGS_DIR=/logs/agent`** — So usage JSON lands where Harbor post-run can read it.

### Plan-mode proposer resilience (`cursor_agent.py`, `cursor_usage.py`)

1. **`stream-json` default** — Parse structured Cursor events instead of raw text only.
2. **`decode_cursor_agent_stdout()`** — Extract proposal text from assistant/result stream events.
3. **Brace-balanced JSON + markdown fence stripping** — Recover proposals when model wraps JSON in fences.
4. **`_is_plausible_shell_command()`** — Reject `json`, prose, and fence markers mistaken for shell commands.
5. **Fallback ladder** — stream-json → text → recovery prompt → **`true` no-op** on parse failure (avoid killing the loop on bad formatting).
6. **Stronger system prompt** — Explicit “no markdown fences around JSON”.
7. **`CursorUsageAccumulator` + `to_harbor_payload()`** — Aggregate tokens/cost across proposal calls for Harbor.

### Terminal runner (`cursor_terminal_agent.py`)

1. **`_write_harbor_usage()` in `finally`** — Always write `/logs/agent/agentprop-cursor-usage.json`.
2. **Exit code 0 on normal loop completion** — Harbor runs its own verifier; exiting 1 when `result.passed` was false caused spurious `NonZeroAgentExitCodeError` even on verifier pass.

### Tooling

1. **`experiments/run_composer_tbench_comparison.sh`** — Wheel build, random task selection, paired summarize.
2. **`experiments/refresh_demo_view.sh`** — Symlink all jobs into `demo-view/` for unified Harbor UI on port 8080.

### Tests

- `tests/test_cursor_usage.py` (new)
- Updates to `tests/test_cursor_agent.py`, `tests/test_cursor_terminal_agent.py`

---

## AgentProp weaknesses and limitations discovered

### 1. Plan-mode overhead dominates on easy tasks

A0 is one uninterrupted `--yolo` session with full tool access. A2 pays **per-step** latency: spawn `cursor-agent --mode plan`, wait for JSON, execute one command, update state, repeat. On `nginx-request-logging`, both pass but A2 used **644s vs 58s** agent time.

**Implication:** The control layer needs a **fast path** (e.g. yolo for N steps, or batch proposals) on tasks where verification density is low.

### 2. Parse fragility (partially fixed, not solved)

Composer in plan mode often returns markdown fences, rationale paragraphs, or the literal token `json`. Before fixes, the harness executed these as shell commands (`exit 127`/`2`), burning steps. Fallback to `true` prevents crashes but **wastes step budget** without progress.

**Still open:** Rationale text occasionally passes plausibility checks and runs as shell (seen in `bn-fit-modify` mid-run steps).

### 3. Uncaught subprocess failures still kill the loop

`subprocess.TimeoutExpired`, Cursor API errors, and non-zero `cursor-agent` exits on later proposals still surface as **`NonZeroAgentExitCodeError`** at the Harbor layer. Parse resilience does not cover **process-level** failures.

**Implication:** `_run_cursor_agent` should catch timeouts and transient failures → retry or no-op, not abort the trial.

### 4. Harbor reward ≠ inner-loop `result.passed`

Harbor grades **container state** after the agent exits. A2 can crash internally yet **pass the verifier** (`nginx`). Conversely, A2 can run 19 proposals and still **fail** verifier (`bn-fit-modify`). Reporting must separate:

- Harbor `reward`
- Inner-loop exit code / exception
- Verifier check breakdown (CTRF)

### 5. Heavy per-trial cold start

Each A2 job installs venv, uploads wheel, installs `cursor-agent` (~1–2 minutes) before task work. Fair for apples-to-apples with A0’s image, but expensive for sweep experiments.

### 6. Wrong execution environment for some tasks

A2 prepends `/opt/agentprop-venv/bin` to `PATH`. Tasks like **`build-cython-ext`** expect **system** Python with preinstalled scientific/build packages. A0 passes; A2 fails in an isolated venv.

**Implication:** Harbor agent should support **`AGENTPROP_USE_SYSTEM_PYTHON=1`** or task-specific PATH profiles.

### 7. No-op steps burn budget without learning

The `true` fallback keeps the loop alive but consumes `max_steps` and API calls. On hard tasks this can mean **high cost, zero verifier progress** (`bn-fit-modify`: $0.61, 19 proposals, reward 0).

### 8. Cost without quality on hard tasks

A2 spent **~9×** A0 cost on `bn-fit-modify` with the same failure. Control adds observability (per-step traces) but **does not guarantee better search** on open-ended statistical / ML tasks without stronger verifiers in the loop.

### 9. Task-type sensitivity

| Task type | Control effect |
|-----------|----------------|
| Structured file output (`regex-log`) | Neutral to positive (faster pass) |
| Iterative compile-run-measure (`path-tracing`) | Strong positive (artifacts + partial credit) |
| Sysadmin config (`nginx`) | Same pass, much slower/costlier |
| Build / system Python (`build-cython-ext`) | Negative (environment mismatch) |
| Probabilistic modeling (`bn-fit-modify`) | Negative (slow, expensive, same fail) |

### 10. Telemetry and reporting gaps

- Early runs: null cost on A2 in Harbor UI.
- `cost_source: "estimated"` in usage JSON — not invoice-grade.
- One paired summarize failed on **post-run PyPI DNS** — `bn-fit-modify` may lack `paired-report/` markdown even though trials exist.
- `NonZeroAgentExitCodeError` on many A2 passes/fails confuses Harbor’s success semantics.

### 11. Controller features under-exercised

These runs mostly stress **propose → execute → repeat**. AgentProp’s graph-native verifier placement, stopping on progress stalls, and strategy switching were not the differentiator — **harness reliability and proposal parsing** were.

---

## Scorecard

| Metric | A0 | A2 |
|--------|----|----|
| Tasks run | 6 | 6 (+1 rerun on cython) |
| Harbor pass rate | 2/6 (33%) | 2/6 (33%) |
| Mean agent time (all tasks) | ~647s | ~641s † |
| Mean cost (tasks with telemetry) | ~$0.10 | ~$0.21 ‡ |
| Runs with inner-loop crash flag | 0/6 | 6/6 |

† Skewed by unequal timeouts; not a fair latency benchmark.  
‡ Only tasks with exported usage; incomplete on early runs.

---

## Hardening shipped (pre–full-suite)

Implemented before the A2-only 89-task sweep:

| Change | Files |
|--------|--------|
| `AGENTPROP_HARBOR_SCORE_ONLY=1` — Harbor verifier owns pass/fail; inner process exits 0 unless fatal misconfig | `harbor_agent.py`, `cursor_terminal_agent.py` |
| Crash-safe `main()` + `agentprop_cursor_crash.json` / `agentprop_cursor_exit.json` | `cursor_terminal_agent.py` |
| Proposer catches all `Exception` → `true` fallback (no harness abort) | `cursor_agent.py` |
| Default **yolo-until-verifier-miss** fast path (900s) | `harbor_agent.py` |
| Default **system Python** for task commands (venv dropped from PATH) | `harbor_agent.py`, `cursor_terminal_agent.py` |
| Proposal retry without burning steps (up to 3/step) | `terminal_loop.py` |
| Verifier failure feedback in proposer prompt; `verifier_failed_count >= 2` → force verify | `cursor_agent.py`, `control_loop.py` |
| Meaningful strategy switch (recovery → yolo repair → tight verify) | `cursor_terminal_agent.py` |
| Workspace-change progress detection; per-step token deltas | `cursor_terminal_agent.py`, `cursor_usage.py` |

**Scripts:** `experiments/run_a2_smoke_gate.sh` (6-task gate), `experiments/run_a2_full_suite.sh` (89-task A2-only).

---

## Recommended next steps

1. **Run smoke gate** — `./experiments/run_a2_smoke_gate.sh` must pass before full suite.
2. **Run full A2 suite** — `./experiments/run_a2_full_suite.sh` with `RESUME=1`.
3. **Blog** — extend `docs/blog/composer-terminal-bench-paired-runs.md` after smoke/full-suite results land.

---

## Reproduce

```bash
# One task
./experiments/run_composer_tbench_comparison.sh regex-log

# Random task (skips tasks that already have A0)
./experiments/run_composer_tbench_comparison.sh

# Refresh Harbor demo viewer
./experiments/refresh_demo_view.sh
harbor view --jobs benchmark-results/composer-comparison/demo-view -p 8080
```

Requires: Harbor CLI, Docker, `CURSOR_API_KEY` in `benchmark-results/.env.local`, Terminal Bench 2.1 dataset cached by Harbor.

---

## Related artifacts

| Artifact | Path |
|----------|------|
| Public blog (2 tasks) | `docs/blog/composer-terminal-bench-paired-runs.md` |
| Raw Harbor jobs | `benchmark-results/composer-comparison/` |
| Run logs | `benchmark-results/composer-comparison/random-run-*.log` |
| Paired reports (per task) | `benchmark-results/composer-comparison/paired-report/{task}/report.md` |
