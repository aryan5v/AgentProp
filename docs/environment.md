# Environment Setup

AgentProp tiers by how much setup you need.

## Tier 1: Graph analysis (no secrets)

```bash
python -m pip install agentprop
agentprop doctor --tier graph
agentprop analyze planner_coder_tester_reviewer --json
```

No API keys or external tools required.

## Tier 2: Development checkout

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
agentprop doctor --tier dev
pytest
```

Editable install makes `PYTHONPATH=src` unnecessary for `experiments/` and
`examples/` when the package is installed in the venv.

## Tier 3: LLM validation

Required for `experiments/run_real_routing_case_study.py` and OpenAI-compatible
evaluation helpers.

| Variable | Required for | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI-compatible LLM runs | For local experiments only; keep prompts and keys out of git |
| `OPENAI_MODEL` | Model selection | e.g. `gpt-4o-mini` |
| `OPENAI_BASE_URL` | Custom endpoints | Optional; defaults to OpenAI |
| `TOKEN_ROUTER_API_KEY` | Token-router harness | Alternative to `OPENAI_API_KEY` |
| `TOKEN_ROUTER_BASE_URL` | Token-router harness | OpenAI-compatible base URL |
| `TOKEN_ROUTER_MODEL` | Token-router harness | Model slug for router |
| `GEMINI_API_KEY` | GAIA-style benchmark | `experiments/run_gaia_style_benchmark.py` |

Copy [.env.example](../.env.example) to `.env` locally (never commit `.env`).

## Tier 4: Terminal-Bench / Harbor

| Variable / tool | Required for | Notes |
| --- | --- | --- |
| Harbor CLI | External benchmark runs | `agentprop terminal-bench prepare` |
| `MODAL_TOKEN_ID` | Optional Modal GPU | Only for large DL sweeps |
| `MODAL_TOKEN_SECRET` | Optional Modal GPU | See `configs/experiment_suites/ml_core.json` |
| Graphviz `dot` | Rendering `.dot` exports | `brew install graphviz` or system package |

## Optional Python extras

| Extra | Install | Enables |
| --- | --- | --- |
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, mypy |
| `ml` | `pip install -e ".[ml]"` | numpy-backed ML scorers |
| `rl` | `pip install -e ".[rl]"` | Gymnasium RL routing |
| `dl` | `pip install -e ".[dl]"` | torch GNN experiments |
| `mcp` | `pip install -e ".[mcp]"` | `agentprop-mcp` FastMCP server |

## Verify your setup

```bash
agentprop doctor --tier graph
agentprop doctor --tier dev
agentprop doctor --tier llm      # checks API key presence only
agentprop readiness --json
```
