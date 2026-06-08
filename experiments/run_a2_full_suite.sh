#!/usr/bin/env bash
# Run A2 (agentprop-cursor + Composer 2.5) across all Terminal-Bench tasks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${CURSOR_MODEL:-cursor/composer-2.5}"
if [[ "$MODEL" != */* ]]; then
  MODEL="cursor/${MODEL}"
fi
OUT_ROOT="${OUT_ROOT:-benchmark-results/composer-comparison/a2-full-suite}"
RESUME="${RESUME:-1}"
TASK_CACHE="${TASK_CACHE:-$HOME/.cache/harbor/tasks/packages/terminal-bench}"

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

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running." >&2
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

SUMMARY_JSONL="${OUT_ROOT}/suite-results.jsonl"
touch "$SUMMARY_JSONL"

mapfile -t TASKS < <(python3 - <<'PY'
import os
cache = os.path.expanduser("~/.cache/harbor/tasks/packages/terminal-bench")
print("\n".join(sorted(os.listdir(cache))))
PY
)

echo "Running A2 on ${#TASKS[@]} tasks -> ${OUT_ROOT}"
for TASK in "${TASKS[@]}"; do
  TASK_DIR="${OUT_ROOT}/${TASK}"
  if [[ "$RESUME" == "1" ]]; then
    LATEST="$(find "$TASK_DIR" -mindepth 3 -maxdepth 3 -name result.json 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "$LATEST" ]]; then
      REWARD="$(python3 -c "import json; print(json.load(open('$LATEST'))['verifier_result']['rewards']['reward'])")"
      if [[ "$REWARD" == "1.0" ]]; then
        echo "Skip ${TASK} (existing pass)"
        continue
      fi
    fi
  fi

  echo "=== A2 full suite: ${TASK} ==="
  harbor run \
    -d terminal-bench/terminal-bench-2-1 \
    -t "terminal-bench/${TASK}" \
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
    -q || true

  RESULT_JSON="$(find "$TASK_DIR" -mindepth 3 -maxdepth 3 -name result.json 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "$RESULT_JSON" ]]; then
    python3 - "$RESULT_JSON" "$TASK" "$SUMMARY_JSONL" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

result_path = Path(sys.argv[1])
task = sys.argv[2]
summary_path = Path(sys.argv[3])
payload = json.loads(result_path.read_text())
reward = (payload.get("verifier_result") or {}).get("rewards", {}).get("reward")
exc = (payload.get("exception_info") or {}).get("exception_type")
agent = payload.get("agent_result") or {}
row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "task": task,
    "reward": reward,
    "exception": exc,
    "cost_usd": agent.get("cost_usd"),
}
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
print(json.dumps(row))
PY
  fi
done

if [[ -x experiments/refresh_demo_view.sh ]]; then
  OUT_ROOT="$OUT_ROOT" ./experiments/refresh_demo_view.sh || true
fi

echo "Full suite complete. Summary: ${SUMMARY_JSONL}"
