#!/usr/bin/env bash
# Compare Terminal-Bench 2.1: raw Cursor CLI (A0) vs AgentProp-controlled Cursor (A2).
# Model: composer-2.5 (Composer 2.5)
#
# Prerequisites:
#   export CURSOR_API_KEY="..."   # or: cursor-agent login
#   Docker running
#
# Usage:
#   ./experiments/run_composer_tbench_comparison.sh [task-name]
#   ./experiments/run_composer_tbench_comparison.sh          # picks random task
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${CURSOR_MODEL:-cursor/composer-2.5}"
if [[ "$MODEL" != */* ]]; then
  MODEL="cursor/${MODEL}"
fi
OUT_ROOT="${OUT_ROOT:-benchmark-results/composer-comparison}"
TASK="${1:-}"

if [[ -z "${CURSOR_API_KEY:-}" ]]; then
  echo "Error: export CURSOR_API_KEY (Harbor runs cursor-cli inside Docker)." >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running." >&2
  exit 2
fi

if [[ -z "$TASK" ]]; then
  TASK="$(python3 - <<'PY'
import os, random
from pathlib import Path
done = {
    p.name
    for p in Path("benchmark-results/composer-comparison/a0-cursor-cli").glob("*")
    if p.is_dir()
}
tasks = sorted(os.listdir(os.path.expanduser("~/.cache/harbor/tasks/packages/terminal-bench")))
remaining = [t for t in tasks if t not in done]
print(random.choice(remaining or tasks))
PY
)"
fi

TASK_REF="terminal-bench/${TASK}"
A0_DIR="${OUT_ROOT}/a0-cursor-cli/${TASK}"
A2_DIR="${OUT_ROOT}/a2-agentprop/${TASK}"
REPORT_DIR="${OUT_ROOT}/paired-report/${TASK}"

export AGENTPROP_MAX_STEPS="${AGENTPROP_MAX_STEPS:-64}"
WHEEL_DIR="${WHEEL_DIR:-/tmp/agentprop-wheels}"
mkdir -p "$WHEEL_DIR"
python3 -m pip wheel "$ROOT" -w "$WHEEL_DIR" --no-deps -q
shopt -s nullglob
wheels=("$WHEEL_DIR"/*.whl)
shopt -u nullglob
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "Error: No wheel files found in $WHEEL_DIR" >&2
  exit 1
fi
export AGENTPROP_WHEEL_PATH="${AGENTPROP_WHEEL_PATH:-$(ls -t "${wheels[@]}" | head -1)}"
# Harbor's Python must be able to import agentprop (and networkx).
HARBOR_PYTHON="${HARBOR_PYTHON:-$HOME/.local/share/uv/tools/harbor/bin/python}"
if ! "$HARBOR_PYTHON" -c "import agentprop" 2>/dev/null; then
  echo "Installing agentprop into Harbor Python ($HARBOR_PYTHON)..."
  uv pip install --python "$HARBOR_PYTHON" -e "$ROOT"
fi

mkdir -p "$OUT_ROOT"
echo "Task: $TASK_REF"
echo "Model: $MODEL"
echo "Output: $OUT_ROOT"
echo ""

echo "=== A0: raw cursor-cli (no AgentProp) ==="
harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  -t "$TASK_REF" \
  -a cursor-cli \
  -m "$MODEL" \
  --env docker \
  -n 1 \
  -o "$A0_DIR" \
  -q

echo ""
echo "=== A2: AgentProp-controlled cursor (agentprop-cursor) ==="
harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  -t "$TASK_REF" \
  --agent-import-path agentprop.benchmarks.harbor_agent:AgentPropCursorAgent \
  -m "$MODEL" \
  --env docker \
  -n 1 \
  -o "$A2_DIR" \
  --ae "CURSOR_API_KEY=$CURSOR_API_KEY" \
  --ae "CURSOR_MODEL=${MODEL#cursor/}" \
  --ae "AGENTPROP_WHEEL_PATH=$AGENTPROP_WHEEL_PATH" \
  --ae "AGENTPROP_MAX_STEPS=$AGENTPROP_MAX_STEPS" \
  --ae "AGENTPROP_HARBOR_SCORE_ONLY=1" \
  --ae "AGENTPROP_USE_SYSTEM_PYTHON=1" \
  --ae "AGENTPROP_FAST_PATH=yolo-until-verifier-miss" \
  --ae "AGENTPROP_FAST_PATH_TIMEOUT_S=900" \
  -q

AGENTPROP_CLI=(python3 -m agentprop.cli)
if ! python3 -c "import agentprop" 2>/dev/null; then
  uv pip install -e "$ROOT" -q
fi

echo ""
echo "=== Summarize arms ==="
"${AGENTPROP_CLI[@]}" terminal-bench summarize \
  --results-root "$A0_DIR" \
  --out-dir "${A0_DIR}/report" \
  --title "A0 cursor-cli composer-2.5: ${TASK}"

"${AGENTPROP_CLI[@]}" terminal-bench summarize \
  --results-root "$A2_DIR" \
  --out-dir "${A2_DIR}/report" \
  --title "A2 agentprop-cursor composer-2.5: ${TASK}"

echo ""
echo "=== Paired comparison (A0 vs A2) ==="
"${AGENTPROP_CLI[@]}" terminal-bench compare \
  --arm "A0=${A0_DIR}" \
  --arm "A2=${A2_DIR}" \
  --baseline-arm A0 \
  --out-dir "$REPORT_DIR" \
  --title "Composer 2.5 paired: ${TASK}"

echo ""
echo "Done."
echo "  A0 report: ${A0_DIR}/report/report.md"
echo "  A2 report: ${A2_DIR}/report/report.md"
echo "  Paired:    ${REPORT_DIR}/report.md"
