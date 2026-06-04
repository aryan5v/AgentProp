# Quality Scoring

AgentProp separates graph optimization metrics from task-quality metrics.

The quality scoring layer supports:

- exact-match scoring for deterministic outputs
- human labels normalized to 0..1
- weighted task-specific rubrics
- LLM-as-judge adapters through an injected judge function
- aggregate quality summaries with pass-rate metadata

## Examples

```python
from agentprop.evaluation import ExactMatchScorer, HumanLabelScorer, RubricScorer

exact = ExactMatchScorer().score(expected="done", actual="done")
human = HumanLabelScorer().from_label(4.0, rationale="correct but verbose")
rubric = RubricScorer({"correct": 0.7, "concise": 0.3}).from_criteria(
    {"correct": True, "concise": False}
)
```

## LLM-As-Judge

AgentProp does not own LLM credentials. The LLM judge scorer accepts an injected
function:

```python
from agentprop.evaluation import LLMJudgeScorer, QualityScore

def judge(expected, actual, context):
    return QualityScore(score=0.8, method="external", passed=True)

score = LLMJudgeScorer(judge).score(expected=None, actual="...", context="...")
```

This keeps provider-specific code out of the core package while making the
case-study runner easy to wire later.
