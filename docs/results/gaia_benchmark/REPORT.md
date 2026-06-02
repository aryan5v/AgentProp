# AgentProp GAIA-Style Benchmark — First Results

**Model:** `gemini-3.5-flash`
**Benchmark:** 50-question multi-hop QA (`benchmarks/gaia_style_qa.json`)
**Workflow:** `research_writer_verifier` — planner -> researcher_a + researcher_b -> writer -> verifier
**Seeds (budget=3):** `planner, writer, verifier`

## Data Quality First

12 of 49 tasks lost an arm to API failures. Headline result uses the clean 37-task subset.

- Tasks attempted: 49
- Skipped tasks: q014
- Dropped from headline: 12 task(s) with at least one 0-token arm
- Clean tasks used for the headline: 37

The full raw data is preserved in `results.json`; infrastructure-failed rows are disclosed below but not used for the headline claim.

## Headline Results — Clean Subset

| Arm | Correct / 37 | Accuracy | Total tokens | vs broadcast |
|---|---:|---:|---:|---:|
| Broadcast | 26 / 37 | 70.3% | 145,989 | — |
| **AgentProp** | **27 / 37** | **73.0%** | **119,057** | **-18.4%** |

**Takeaway:** AgentProp reached accuracy parity on the clean subset while using materially fewer tokens. The +1 task accuracy edge is noise; the defensible claim is parity at lower cost.

## Full Set Disclosure

| Arm | Correct / attempted | Accuracy | Total tokens | vs broadcast |
|---|---:|---:|---:|---:|
| Broadcast | 29 / 49 | 59.2% | 162,931 | — |
| AgentProp | 29 / 49 | 59.2% | 127,117 | -22.0% |

These numbers include provider-failed rows and are shown for transparency, not as the headline.

## Routing Setup

- Broadcast: all five stages receive the full guidelines document (~244 rough tokens).
- AgentProp: seed stages receive full context; non-seed researcher stages receive a compressed summary.
- Scoring: exact-answer QA match from the final verifier output, without human labels or rubric scoring.

## Methodology Notes

- Seed selection: `greedy_seed_selection` with `IndependentCascade`, budget K=3.
- Graph: `research_writer_verifier`, a 5-stage fan-out and synthesis workflow.
- Context compression: one shared summary is produced for non-seed stages in the AgentProp arm.
- Retry policy: exponential backoff around provider calls; zero-token exhausted retries are separated from routing outcomes.
- Parallelism: task-level batching via `ThreadPoolExecutor`; the saved run skipped q014 after repeated socket hangs.
- Dropped-row rule: a task is excluded from the headline if either arm returns 0 tokens.

## Per-Task Detail

| Task | Included | Question (truncated) | Gold | Broadcast | AgentProp | B tokens | A tokens |
|---|---|---|---|---|---|---:|---:|
| q001 | no | What is the capital of the country that won the most gold medals at the… | Washington DC | PASS | FAIL | 4,408 | 0 |
| q002 | no | How many sides does the polygon that forms the base of the Washington M… | 4 | FAIL | PASS | 0 | 3,227 |
| q003 | yes | In what year was the company that makes the iPhone founded? | 1976 | PASS | PASS | 3,170 | 2,783 |
| q004 | yes | What is the name of the river that flows through the city where the Eif… | Seine | PASS | PASS | 3,463 | 2,954 |
| q005 | yes | How many bones are in the hand of the species that built the Great Wall… | 27 | PASS | PASS | 4,008 | 3,854 |
| q006 | yes | What is the currency of the country where the Amazon River originates? | Peruvian sol | PASS | PASS | 3,883 | 3,052 |
| q007 | yes | What programming language was created by the same person who created Li… | C | FAIL | FAIL | 7,207 | 4,258 |
| q008 | yes | How many time zones does the country with the world's longest coastline… | 6 | PASS | PASS | 3,889 | 3,433 |
| q009 | yes | What is the official language of the country where Mount Kilimanjaro is… | Swahili | PASS | PASS | 4,105 | 3,041 |
| q010 | yes | How many chambers does the heart of the animal that is the national sym… | 4 | PASS | PASS | 4,095 | 3,618 |
| q011 | yes | In what decade was the author of '1984' born? | 1900s | PASS | PASS | 3,556 | 2,789 |
| q013 | yes | How many countries share a land border with the country that has the wo… | 14 | FAIL | FAIL | 5,436 | 4,736 |
| q015 | yes | How many strings does the instrument that Jimi Hendrix played have? | 6 | PASS | PASS | 3,271 | 2,895 |
| q017 | yes | In which city was the person who painted the Mona Lisa born? | Vinci | PASS | PASS | 3,622 | 2,653 |
| q012 | yes | What is the atomic number of the element whose chemical symbol is the s… | 47 | PASS | PASS | 7,293 | 6,503 |
| q016 | yes | What is the tallest mountain in the country that borders both the Unite… | Pico de Orizaba | FAIL | FAIL | 6,506 | 7,015 |
| q018 | yes | What is the boiling point in Celsius of the most abundant gas in Earth'… | -196 | FAIL | PASS | 4,486 | 3,060 |
| q019 | yes | How many players are on the field for one team in the sport that was in… | 6 | FAIL | FAIL | 7,090 | 6,274 |
| q020 | yes | What is the name of the space agency that landed the first humans on th… | NASA | PASS | PASS | 3,598 | 2,768 |
| q021 | yes | How many letters are in the name of the capital of the country with the… | 6 | FAIL | FAIL | 4,633 | 3,714 |
| q022 | yes | What is the speed of light in km/s rounded to the nearest thousand? | 300000 | PASS | PASS | 4,144 | 4,070 |
| q023 | yes | In what year did the country that first put a satellite into orbit laun… | 1957 | PASS | PASS | 3,602 | 2,963 |
| q024 | yes | What is the largest organ in the human body by surface area? | skin | PASS | PASS | 4,663 | 3,010 |
| q025 | yes | How many planets in our solar system have rings? | 4 | PASS | PASS | 4,051 | 2,976 |
| q026 | yes | What language is the word 'tsunami' derived from? | Japanese | PASS | PASS | 3,574 | 3,248 |
| q027 | yes | How many colors are in a standard rainbow? | 7 | PASS | PASS | 3,933 | 2,802 |
| q028 | yes | What is the chemical formula for table salt? | NaCl | PASS | PASS | 3,676 | 2,700 |
| q029 | yes | In what year was the World Wide Web invented? | 1989 | PASS | PASS | 3,618 | 3,162 |
| q030 | no | How many sides does a dodecagon have? | 12 | FAIL | FAIL | 1,040 | 0 |
| q031 | yes | What is the name of the longest bone in the human body? | femur | PASS | PASS | 3,687 | 2,632 |
| q032 | yes | How many teeth does an adult human have including wisdom teeth? | 32 | FAIL | FAIL | 4,133 | 1,960 |
| q033 | no | What is the name of the galaxy closest to the Milky Way? | Andromeda | FAIL | FAIL | 817 | 0 |
| q034 | no | In what year did the Berlin Wall fall? | 1989 | FAIL | FAIL | 0 | 0 |
| q035 | no | What is the name of the process by which plants convert sunlight into f… | photosynthesis | FAIL | PASS | 0 | 2,953 |
| q036 | yes | How many prime numbers are less than 20? | 8 | PASS | PASS | 3,937 | 2,738 |
| q037 | yes | What is the hardest natural substance on Earth? | diamond | PASS | FAIL | 3,831 | 1,144 |
| q038 | no | In what year did the Titanic sink? | 1912 | FAIL | FAIL | 0 | 731 |
| q039 | no | What is the square root of 144? | 12 | PASS | FAIL | 4,364 | 0 |
| q040 | no | How many continents are there on Earth? | 7 | FAIL | FAIL | 2,016 | 0 |
| q041 | no | What is the name of the treaty that ended World War I? | Treaty of Versailles | FAIL | FAIL | 912 | 0 |
| q042 | no | How many elements are in the periodic table as of 2023? | 118 | FAIL | FAIL | 0 | 1,149 |
| q043 | yes | What planet is known as the Red Planet? | Mars | FAIL | PASS | 583 | 2,336 |
| q044 | yes | What is the name of the force that keeps planets in orbit around the Su… | gravity | FAIL | PASS | 821 | 3,091 |
| q045 | yes | How many vertices does a cube have? | 8 | PASS | PASS | 3,825 | 3,249 |
| q046 | no | What is the chemical symbol for gold? | Au | PASS | FAIL | 3,385 | 0 |
| q047 | yes | In which decade did the first commercial jet passenger service begin? | 1950s | PASS | FAIL | 4,408 | 875 |
| q048 | yes | What is the name of the longest river in the United States? | Missouri River | FAIL | FAIL | 2,070 | 2,948 |
| q049 | yes | How many symphonies did Beethoven complete? | 9 | FAIL | PASS | 596 | 3,157 |
| q050 | yes | What is the name of the scientist who developed the theory of general r… | Albert Einstein | PASS | FAIL | 3,526 | 596 |

## Interpretation

This benchmark is an initial evidence slice, not a leaderboard claim. It shows the product value in the regime AgentProp targets: a multi-agent fan-out workflow where shared context can be routed selectively instead of broadcast blindly.

The important improvement over the earlier coding run is failure awareness. Rows where a provider returned no output are separated from real routing outcomes, and the report no longer lets timeout artifacts masquerade as model or routing errors.

## Limits

- Self-contained QA does not exercise retrieval-heavy contexts where AgentProp should save more.
- One model and one run means the accuracy delta should be treated as directional.
- Short-answer QA is less sensitive to context compression than coding or long-form synthesis.

*Results in `results.json`; per-stage outputs in `outputs.jsonl`.*
