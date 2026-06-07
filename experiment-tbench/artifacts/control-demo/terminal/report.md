# AgentProp Control Session Report

- Task: `terminal-control-demo`
- Category: `terminal-repair`
- Workflow: `tool_use_pipeline`
- Events observed: `4`
- Latest decision: `FINALIZE`

## Analysis Snapshot
- Nodes: `6`
- Edges: `7`
- Verifier candidates: `tester, planner`
- Bottlenecks: `code_tool, tester, analyst`
- Pruning candidates: `tester->code_tool`

## Runtime Features
- Tokens used: `140`
- Elapsed seconds: `17.0`
- Steps since verifier: `0`
- Repeated errors: `0`
- Unconfirmed pass: `False`

## Decisions
- `1` `CONTINUE`: within execution budget
- `2` `CONTINUE`: within execution budget
- `3` `SWITCH_STRATEGY`: same error repeated
- `4` `FINALIZE`: independent verifier passed

## Outcome
- Passed: `True`
- Token savings: `0.2222222222222222`
