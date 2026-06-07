.PHONY: install dev test lint typecheck demo tutorial doctor

install:
	python3 -m pip install -e .

dev:
	python3 -m pip install -e ".[dev]"

test: dev
	pytest

lint: dev
	ruff check .

typecheck: dev
	mypy src

demo:
	agentprop control-demo --demo terminal --out-dir reports/control-demo

tutorial:
	agentprop optimize planner_coder_tester_reviewer --budget 2 --trials 20
	agentprop benchmark planner_coder_tester_reviewer --budget 2 --trials 20
	agentprop report planner_coder_tester_reviewer --budget 2 --out reports/tutorial.md

doctor:
	agentprop doctor --tier graph
	agentprop doctor --tier dev
