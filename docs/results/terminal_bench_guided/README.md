# AgentProp-Guided Terminal-Bench Benchmark

This report summarizes the first AgentProp-guided Terminal-Bench run. It is a
directional evaluation of AgentProp-style routing guidance for a coding agent,
not a leaderboard submission.

## Setup

- Date: 2026-06-01
- Harness: Harbor
- Environment: Modal
- Dataset: `terminal-bench/terminal-bench-2`
- Agent: `gemini-cli`
- Model: `google/gemini-3.5-flash`
- Comparison set: 27 Terminal-Bench tasks where the local baseline had a saved
  result.
- AgentProp condition: same agent and model with an AgentProp skill plus extra
  instructions to use planner, implementer, verifier, and finalizer phases.
- Baseline condition: same local Gemini CLI benchmark artifacts without the
  AgentProp-guided skill/instruction layer.

One task, `mteb-leaderboard`, hung in the external harness and did not produce a
result artifact. It is excluded from the headline comparison.

## Headline Result

| Metric | Baseline | AgentProp-guided |
|---|---:|---:|
| Completed matched tasks | 26 | 26 |
| Passes | 17 | 18 |
| Pass rate | 65.4% | 69.2% |
| Improvements | - | 3 |
| Regressions | - | 2 |
| Ties | - | 21 |

AgentProp-guided routing improved the matched pass count by one task on this
snapshot. The signal is positive but small, and the run also exposed important
timeout and over-exploration failure modes.

## Baseline Uplift

The initial local baseline passed 17 of the 26 completed matched tasks and
failed nine. AgentProp-guided routing converted three of those baseline failures
into passes:

| Task | Baseline | AgentProp-guided | What changed |
|---|---:|---:|---|
| `build-pov-ray` | 0.0 | 1.0 | Converted a legacy build task from fail to pass while using fewer raw input+output tokens than the baseline artifact. |
| `caffe-cifar-10` | 0.0 | 1.0 | Converted a long-running build/train/verify task from timeout/fail to pass with lower reported cost in the matched artifacts. |
| `sanitize-git-repo` | 0.0 | 1.0 | Converted a repository-hygiene task from fail to pass by keeping the search, edit, and verification loop explicit. |

Two baseline passes regressed in the guided run, leaving a net improvement of
one pass on the completed matched subset. The useful signal is the shape of the
improvements: AgentProp did not merely add a prompt flourish; it recovered older
failing tests where task execution benefits from structured handoff between
planning, implementation, and verification phases.

## Token And Cost Snapshot

Token accounting is available for 24 of the 26 completed matched tasks. Two
timeout tasks had incomplete token artifacts on at least one side.

| Metric | Baseline | AgentProp-guided | Delta |
|---|---:|---:|---:|
| Input tokens | 19,294,016 | 18,795,792 | -2.6% |
| Output tokens | 395,539 | 423,414 | +7.0% |
| Input + output tokens | 19,689,555 | 19,219,206 | -2.4% |
| Cached tokens | 14,658,404 | 14,184,460 | -3.2% |
| Reported cost | $12.71 | $12.86 | +1.1% |

The guided run used fewer raw input+output tokens on matched tasks with token
data, but slightly higher reported cost. The cost difference comes from the mix
of cache-discounted input tokens and additional output tokens on some harder
tasks.

## Task Outcomes

| Task | Baseline | AgentProp-guided | Outcome | Exception |
|---|---:|---:|---|---|
| `break-filter-js-from-html` | 0.0 | 0.0 | same | - |
| `build-pov-ray` | 0.0 | 1.0 | improved | - |
| `caffe-cifar-10` | 0.0 | 1.0 | improved | - |
| `chess-best-move` | 1.0 | 0.0 | regressed | - |
| `cobol-modernization` | 1.0 | 1.0 | same | AgentTimeoutError |
| `constraints-scheduling` | 1.0 | 1.0 | same | - |
| `crack-7z-hash` | 1.0 | 1.0 | same | - |
| `db-wal-recovery` | 0.0 | 0.0 | same | AgentTimeoutError |
| `distribution-search` | 1.0 | 1.0 | same | - |
| `dna-insert` | 0.0 | 0.0 | same | - |
| `extract-elf` | 0.0 | 0.0 | same | AgentTimeoutError |
| `financial-document-processor` | 0.0 | 0.0 | same | - |
| `fix-git` | 1.0 | 1.0 | same | - |
| `git-leak-recovery` | 1.0 | 1.0 | same | - |
| `hf-model-inference` | 1.0 | 1.0 | same | - |
| `kv-store-grpc` | 1.0 | 1.0 | same | - |
| `log-summary-date-ranges` | 1.0 | 1.0 | same | - |
| `mteb-leaderboard` | 1.0 | hung | excluded | harness hang |
| `openssl-selfsigned-cert` | 1.0 | 1.0 | same | - |
| `overfull-hbox` | 1.0 | 1.0 | same | - |
| `prove-plus-comm` | 1.0 | 1.0 | same | - |
| `pytorch-model-recovery` | 1.0 | 1.0 | same | - |
| `raman-fitting` | 0.0 | 0.0 | same | AgentTimeoutError |
| `regex-log` | 1.0 | 1.0 | same | - |
| `sanitize-git-repo` | 0.0 | 1.0 | improved | - |
| `tune-mjcf` | 1.0 | 0.0 | regressed | AgentTimeoutError |
| `vulnerable-secret` | 1.0 | 1.0 | same | - |

## What Improved

The three improved tasks are the most important part of this first run because
they show initial product value: AgentProp-guided routing helped turn baseline
failures into passes.

- `build-pov-ray` moved from fail to pass. This points to value on legacy
  build/setup tasks where the agent has to preserve exact installation details
  and verify the produced binary.
- `caffe-cifar-10` moved from fail to pass after the baseline timed out. This
  points to value on multi-stage ML engineering tasks where build, training, and
  accuracy checks must stay aligned.
- `sanitize-git-repo` moved from fail to pass. This points to value on
  repository-wide search/edit/verify tasks where missing one contaminated value
  is enough to fail the benchmark.

These wins are not enough to claim broad benchmark dominance, but they are a
strong early signal for AgentProp's core hypothesis: routing discipline should
spend context and verification effort on the phases most likely to determine
task success.

## What Regressed

- `chess-best-move`
- `tune-mjcf`

`tune-mjcf` ended with `AgentTimeoutError`, so it should be interpreted as a
timeout-sensitive regression. `chess-best-move` is the cleaner behavioral
regression and should be reviewed before making stronger benchmark claims.

## Failure Modes

Several tasks timed out or over-explored:

- `cobol-modernization`
- `db-wal-recovery`
- `extract-elf`
- `raman-fitting`
- `tune-mjcf`

The `raman-fitting` trace is especially useful. The agent spent a long time on
exploratory numerical fitting and produced values in the wrong coordinate system.
That points to a concrete product improvement: numerical and scientific tasks
need early checks for units, coordinate conventions, and verifier expectations.

## Interpretation

This run supports a conservative claim:

> On a 26-task matched Terminal-Bench snapshot, AgentProp-guided instructions
> improved pass rate from 65.4% to 69.2% while using slightly fewer raw tokens on
> tasks with complete token accounting.

It does not prove a general benchmark win. The next evaluation should run the
official full benchmark setup end to end, include repeated trials, and add
budget-aware stop conditions so the guidance improves reliability without
encouraging unbounded exploration.
