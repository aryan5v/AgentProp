from agentprop.evaluation import compare_routing
from agentprop.integrations import render_coding_agent_instructions
from agentprop.propagation import ZeroForcing
from agentprop.workflows import planner_coder_tester_reviewer


def test_render_coding_agent_instructions_contains_routing_brief() -> None:
    graph = planner_coder_tester_reviewer()
    model = ZeroForcing()
    propagation = model.simulate(graph, ["planner", "tester"], trials=1)
    report = compare_routing(
        graph,
        ["planner", "tester"],
        model.name,
        propagation,
        verifier_candidates=["tester"],
        pruning_candidates=[("planner", "reviewer")],
    )

    markdown = render_coding_agent_instructions(
        report,
        workflow_name="planner_coder_tester_reviewer",
        target="codex",
    )

    assert "AgentProp Brief For Codex" in markdown
    assert "`planner`" in markdown
    assert "Verifier Placement" in markdown
    assert "run_experiment_suite.py" in markdown
    assert "Suggested Agent Prompt" in markdown
