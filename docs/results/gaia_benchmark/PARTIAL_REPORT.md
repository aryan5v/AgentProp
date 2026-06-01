# AgentProp GAIA-Style Benchmark — Partial Results

**Model:** `gemini-3.5-flash`  **Updated:** 2026-06-01 05:15 UTC  **Progress:** 13/50 tasks

> This is a live partial report updated every ~2 minutes while the run is active.

## Running Tally

| Arm | Correct so far | Accuracy | Total tokens | vs broadcast |
|---|---|---|---|---|
| Broadcast | 11/13 | 85% | 60,820 | — |
| AgentProp | 10/13 | 77% | 45,297 | +25.5% |

## Per-Task Results

| # | Task | Gold (short) | Broadcast | AgentProp | B tokens | A tokens |
|---|---|---|---|---|---|---|
| q001 | What is the capital of the country ... | — | ✓ `Washington, D.C.` | ✓ `Washington, D.C.` | 4,070 | 3,868 |
| q002 | How many sides does the polygon tha... | — | ✓ `4` | ✓ `4` | 3,985 | 3,168 |
| q003 | In what year was the company that m... | — | ✓ `1976` | ✓ `1976` | 3,740 | 2,587 |
| q004 | What is the name of the river that ... | — | ✓ `Seine` | ✓ `Seine` | 3,432 | 3,049 |
| q005 | How many bones are in the hand of t... | — | ✓ `27` | ✓ `27` | 4,143 | 3,389 |
| q006 | What is the currency of the country... | — | ✓ `Peruvian sol` | ✓ `Peruvian sol` | 3,919 | 2,892 |
| q007 | What programming language was creat... | — | ✗ `None` | ✗ `None` | 9,121 | 4,767 |
| q008 | How many time zones does the countr... | — | ✓ `6` | ✓ `6` | 4,251 | 3,394 |
| q009 | What is the official language of th... | — | ✓ `Swahili and English` | ✓ `Swahili and English` | 4,155 | 3,411 |
| q010 | How many chambers does the heart of... | — | ✓ `4` | ✗ `four` | 4,230 | 3,054 |
| q011 | In what decade was the author of '1... | — | ✓ `1900s` | ✓ `1900s` | 3,947 | 2,847 |
| q012 | What is the atomic number of the el... | — | ✓ `47` | ✓ `47` | 5,751 | 6,842 |
| q013 | How many countries share a land bor... | — | ✗ `6` | ✗ `` | 6,076 | 2,029 |
