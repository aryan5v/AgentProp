#!/usr/bin/env bash
# Re-run the 6 paired-comparison smoke tasks on A2 and enforce harness gate criteria.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${CURSOR_MODEL:-cursor/composer-2.5}"
if [[ "$MODEL" != */* ]]; then
  MODEL="cursor/${MODEL}"
fi
OUT_ROOT="${OUT_ROOT:-benchmark-results/composer-comparison/a2-smoke-gate}"
DEFAULT_SMOKE_TASKS=(
  regex-log
  nginx-request-logging
  build-cython-ext
  path-tracing
  torch-pipeline-parallelism
  bn-fit-modify
)
if [[ -n "${SMOKE_ONLY:-}" ]]; then
  IFS=',' read -r -a SMOKE_TASKS <<< "$SMOKE_ONLY"
else
  SMOKE_TASKS=("${DEFAULT_SMOKE_TASKS[@]}")
fi

if [[ -z "${CURSOR_API_KEY:-}" ]]; then
  if [[ -f benchmark-results/.env.local ]]; then
    set -a
    # shellcheck disable=SC1091
    source benchmark-results/.env.local
    set +a
  fi
fi
if [[ -z "${CURSOR_API_KEY:-}" ]]; then
  echo "Error: export CURSOR_API_KEY or create benchmark-results/.env.local" >&2
  exit 2
fi

WHEEL_DIR="${WHEEL_DIR:-/tmp/agentprop-wheels}"
mkdir -p "$WHEEL_DIR" "$OUT_ROOT"
python3 -m pip wheel "$ROOT" -w "$WHEEL_DIR" --no-deps -q
export AGENTPROP_WHEEL_PATH="${AGENTPROP_WHEEL_PATH:-$(ls -t "$WHEEL_DIR"/*.whl | head -1)}"
export AGENTPROP_MAX_STEPS="${AGENTPROP_MAX_STEPS:-64}"

HARBOR_PYTHON="${HARBOR_PYTHON:-$HOME/.local/share/uv/tools/harbor/bin/python}"
if ! "$HARBOR_PYTHON" -c "import agentprop" 2>/dev/null; then
  uv pip install --python "$HARBOR_PYTHON" -e "$ROOT" -q
fi

SUMMARY_JSONL="${SUMMARY_JSONL:-${OUT_ROOT}/smoke-results.jsonl}"
if [[ "${SMOKE_APPEND:-0}" != "1" ]]; then
  : >"$SUMMARY_JSONL"
fi
FAILURES=0

for TASK in "${SMOKE_TASKS[@]}"; do
  TASK_REF="terminal-bench/${TASK}"
  TASK_DIR="${OUT_ROOT}/${TASK}"
  EXISTING="$(find "$TASK_DIR" -mindepth 3 -maxdepth 3 -name result.json 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "$EXISTING" ]]; then
  GATE_CHECK="$(python3 - "$EXISTING" "$TASK" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
task = sys.argv[2]
reward = (payload.get("verifier_result") or {}).get("rewards", {}).get("reward")
exc = (payload.get("exception_info") or {}).get("exception_type")
ctrf_path = Path(sys.argv[1]).parent / "verifier" / "ctrf.json"
checks_passed = None
if ctrf_path.exists():
    tests = (json.loads(ctrf_path.read_text()).get("results") or {}).get("tests") or []
    checks_passed = sum(1 for t in tests if t.get("status") == "passed")
passed = reward == 1.0
ok = not exc
if task in {"regex-log", "nginx-request-logging", "build-cython-ext"}:
    ok = ok and passed
elif task == "path-tracing":
    ok = ok and (passed or (checks_passed is not None and checks_passed >= 4))
print("reuse" if ok else "rerun")
PY
)"
    if [[ "$GATE_CHECK" == "reuse" ]]; then
      echo "=== A2 smoke: ${TASK} (reuse existing pass) ==="
      ROW="$(python3 - "$EXISTING" "$TASK" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
task = sys.argv[2]
reward = (payload.get("verifier_result") or {}).get("rewards", {}).get("reward")
exc = (payload.get("exception_info") or {}).get("exception_type")
print(json.dumps({"task": task, "reward": reward, "exception": exc, "gate_ok": True, "reasons": ["reused"]}))
PY
)"
      echo "$ROW" >>"$SUMMARY_JSONL"
      echo "GATE PASS: $ROW"
      continue
    fi
  fi
  echo "=== A2 smoke: ${TASK} ==="
  harbor run \
    -d terminal-bench/terminal-bench-2-1 \
    -t "$TASK_REF" \
    --agent-import-path agentprop.benchmarks.harbor_agent:AgentPropCursorAgent \
    -m "$MODEL" \
    --env docker \
    -n 1 \
    -o "$TASK_DIR" \
    --ae "CURSOR_API_KEY=$CURSOR_API_KEY" \
    --ae "CURSOR_MODEL=${MODEL#cursor/}" \
    --ae "AGENTPROP_WHEEL_PATH=$AGENTPROP_WHEEL_PATH" \
    --ae "AGENTPROP_MAX_STEPS=$AGENTPROP_MAX_STEPS" \
    --ae "AGENTPROP_HARBOR_SCORE_ONLY=1" \
    --ae "AGENTPROP_USE_SYSTEM_PYTHON=1" \
    --ae "AGENTPROP_FAST_PATH=yolo-until-verifier-miss" \
    --ae "AGENTPROP_FAST_PATH_TIMEOUT_S=900" \
    -q

  RESULT_JSON="$(find "$TASK_DIR" -mindepth 3 -maxdepth 3 -name result.json 2>/dev/null | sort | tail -1)"
  GATE="$(python3 - "$RESULT_JSON" "$TASK" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
task = sys.argv[2]
payload = json.loads(result_path.read_text())
reward = (payload.get("verifier_result") or {}).get("rewards", {}).get("reward")
exc = (payload.get("exception_info") or {}).get("exception_type")
passed = reward == 1.0
ctrf_path = result_path.parent / "verifier" / "ctrf.json"
checks_passed = None
checks_total = None
if ctrf_path.exists():
    ctrf = json.loads(ctrf_path.read_text())
    tests = (ctrf.get("results") or {}).get("tests") or []
    checks_total = len(tests)
    checks_passed = sum(1 for test in tests if test.get("status") == "passed")

ok = True
reasons: list[str] = []
if exc:
    ok = False
    reasons.append(f"exception={exc}")
if task in {"regex-log", "nginx-request-logging", "build-cython-ext"}:
    if not passed:
        ok = False
        reasons.append("expected reward=1.0")
elif task == "path-tracing":
    if not passed and (checks_passed is None or checks_passed < 4):
        ok = False
        reasons.append("expected pass or >=4/5 checks")
elif task in {"torch-pipeline-parallelism", "bn-fit-modify"}:
    if exc and exc not in {None, ""}:
        ok = False
        reasons.append(f"harness failure {exc}")

print(json.dumps({
    "task": task,
    "reward": reward,
    "exception": exc,
    "checks_passed": checks_passed,
    "checks_total": checks_total,
    "gate_ok": ok,
    "reasons": reasons,
}))
PY
)"
  echo "$GATE" >>"$SUMMARY_JSONL"
  if [[ "$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['gate_ok'])" "$GATE")" != "True" ]]; then
    echo "GATE FAIL: $GATE" >&2
    FAILURES=$((FAILURES + 1))
  else
    echo "GATE PASS: $GATE"
  fi
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "Smoke gate failed on ${FAILURES} task(s). See ${SUMMARY_JSONL}" >&2
  exit 1
fi

echo "Smoke gate passed for all ${#SMOKE_TASKS[@]} tasks."
echo "Summary: ${SUMMARY_JSONL}"
