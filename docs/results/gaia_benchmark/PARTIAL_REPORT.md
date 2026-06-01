# AgentProp GAIA-Style Benchmark — Partial Results

**Model:** `gemini-3.5-flash`  **Updated:** 2026-06-01 06:15 UTC  **Progress:** 2/50 tasks

> This is a live partial report updated every ~2 minutes while the run is active.

## Running Tally

| Arm | Correct so far | Accuracy | Total tokens | vs broadcast |
|---|---|---|---|---|
| Broadcast | 2/2 | 100% | 8,029 | — |
| AgentProp | 2/2 | 100% | 6,240 | +22.3% |

## Per-Task Results

| # | Task | Gold (short) | Broadcast | AgentProp | B tokens | A tokens |
|---|---|---|---|---|---|---|
| q001 | What is the capital of the country ... | — | ✓ `Washington, D.C.` | ✓ `Washington, D.C.` | 4,066 | 3,096 |
| q002 | How many sides does the polygon tha... | — | ✓ `4` | ✓ `4` | 3,963 | 3,144 |
