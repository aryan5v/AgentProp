# Public Release Decision

## Decision

Keep the GitHub repository private until the first real LLM case-study result is
attached, or publish it explicitly as `0.1.0-alpha` with the limitation stated in
the release notes.

## Rationale

AgentProp now has a working framework, green CI, docs, benchmark artifacts,
classical propagation models, optional torch GNNs, and Q-learning routing.
However, the strongest public claim depends on real LLM workflow validation, not
only synthetic graph benchmarks.

## Public-Ready Criteria

The repository can be made public once one of these is true:

- preferred: `docs/results/case_study_001/` contains the completed real LLM case
  study from [research/case_study_protocol.md](research/case_study_protocol.md)
- acceptable alpha: release notes clearly say real LLM task-quality validation
  is pending

## Current Recommendation

Release privately as `v0.1.0-alpha.1`, run the first case study, then make the
repository public as `v0.1.0` or `v1.0.0-alpha` depending on how strong the
results are.
