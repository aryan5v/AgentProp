# Benchmark Manifest

## Workflow Fixtures

### `workflows/planner_coder_tester_reviewer.json`

The first end-to-end demo workflow. It models a common software-agent loop:

```text
Planner -> Coder -> Tester -> Reviewer -> Final
Planner -> Reviewer
Tester -> Coder
```

This fixture is designed to test:

- Context seeding
- Correction propagation from tester back to coder
- Verifier placement around tester/reviewer
- Cost savings against broadcast routing
- Low-weight edge pruning

Expected qualitative behavior:

- `planner` should often rank highly because it reaches coder and reviewer.
- `tester` is a strong verifier candidate because it detects implementation errors and routes corrections back to coder.
- `planner -> reviewer` is useful but lower-weight than the main pipeline.
- Broadcast routing wastes tokens by giving full context to every downstream node upfront.
