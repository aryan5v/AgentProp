# AgentProp GAIA-Style Benchmark — Final Results

**Model:** `gemini-3.5-flash`  
**Benchmark:** 50-question multi-hop QA (`benchmarks/gaia_style_qa.json`)  
**Workflow:** `research_writer_verifier` — planner → researcher_a ‖ researcher_b → writer → verifier  
**Seeds (budget=3):** planner, writer, verifier (selected via greedy IndependentCascade)  
**Date:** 2026-06-01

---

## ⚠️ Data Quality First

This run was executed against a free/shared Gemini endpoint that returned
intermittent 503 "high demand" errors. **12 of 49 attempted tasks lost at
least one arm to an API timeout** (a 0-token, empty-answer result after the
retry budget was exhausted). Those are infrastructure failures, not routing
outcomes, so the **headline result is reported on the clean 37-task subset
where both arms actually produced output.**

- Tasks attempted: 49 (q014 skipped — persistent socket hang)
- Tasks dropped (≥1 arm timed out): **12** → q001, q002, q030, q033, q034, q035, q038, q039, q040, q041, q042, q046
- **Clean tasks used for the headline: 37**

The full 49-task raw data (including the failed tasks) is preserved in
`results.json` for transparency.

---

## Headline Results — Clean 37-Task Subset

| Arm | Correct / 37 | Accuracy | Total Tokens | vs Broadcast |
|---|---|---|---|---|
| Broadcast | 26 / 37 | 70.3% | 145,989 | — |
| **AgentProp** | **27 / 37** | **73.0%** | **119,057** | **−18.4%** |

**AgentProp slightly exceeds broadcast accuracy while using 18.4% fewer tokens.**

The +1 accuracy is within noise — the honest claim is **accuracy parity at
~18% lower token cost.** AgentProp did not trade quality for efficiency.

---

## For Reference — Full 49-Task Set (contaminated)

Including the 12 API-failed tasks (each failed arm scored wrong / 0 tokens):

| Arm | Correct / 49 | Accuracy | Total Tokens | vs Broadcast |
|---|---|---|---|---|
| Broadcast | 29 / 49 | 59.2% | 162,931 | — |
| AgentProp | 29 / 49 | 59.2% | 127,117 | −22.0% |

These numbers are deflated by infrastructure failures and should **not** be
used as the headline. They are included only for full disclosure.

---

## Token Efficiency (clean subset)

| Metric | Value |
|---|---|
| Broadcast total tokens | 145,989 |
| AgentProp total tokens | 119,057 |
| Absolute saving | 26,932 tokens |
| Relative saving | **−18.4%** |

The saving comes from routing non-seed agents (researcher_a, researcher_b)
through a compressed context summary while seed agents (planner, writer,
verifier) keep the full guidelines document.

---

## Interpretation

### What the numbers show
On the 37 tasks where the API behaved, AgentProp's selective seeding
achieves **accuracy parity (indeed a slight edge) with ~18% fewer tokens**
across a 5-stage multi-agent pipeline. The efficiency gain does not cost
answer quality.

### Concessions

1. **Infrastructure, not method, capped the sample.** 12/49 tasks were lost
   to provider 503s on a shared endpoint. A paid/dedicated endpoint or a
   re-run of the failed tasks would restore the full 49-task set. The method
   itself never errored.

2. **+1 accuracy is noise.** With n=37, a one-task difference is not
   statistically significant. The defensible claim is *parity*, not
   *superiority*, on accuracy — with a real and consistent token saving.

3. **Compression is aggressive.** Budget=3 IndependentCascade compresses
   non-seed context heavily. A larger summary budget is a tunable knob that
   could shift the accuracy/cost trade-off.

### Research claim this supports
> *"In a 5-stage multi-agent research-synthesis pipeline, graph-based seed
> selection routes full context to the 3 highest-influence agents and
> compresses it for the rest — achieving equal answer accuracy at ~18% lower
> token cost on a 37-question clean multi-hop QA subset."*

---

## Methodology Notes

- **Seed selection:** `greedy_seed_selection` with `IndependentCascade`, budget K=3
- **Graph construction:** `graph_from_trace_dict` re-fits edge weights from each task's execution trace
- **Context compression:** one-shot LLM call to summarise the shared guidelines doc
- **Scoring:** case-insensitive exact match, whitespace/punctuation normalised, substring tolerance for multi-word answers
- **Retry policy:** 8 retries, exponential backoff 3 s → 45 s with ±25% jitter
- **Parallelism:** 4–6 workers via `ThreadPoolExecutor`; socket-level timeout 50 s
- **Skipped:** q014 (persistent socket hang across 3 independent runs)
- **Dropped from headline:** 12 tasks with ≥1 zero-token arm (API timeout)
