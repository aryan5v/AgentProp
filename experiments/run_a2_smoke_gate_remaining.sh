#!/usr/bin/env bash
# Re-run the 3 smoke tasks that failed or were skipped; merge with prior passes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-benchmark-results/composer-comparison/a2-smoke-gate}"
PRIOR="${OUT_ROOT}/smoke-results.jsonl"
MERGED="${OUT_ROOT}/smoke-results-merged.jsonl"

python3 - "$PRIOR" "$MERGED" <<'PY'
import json
import sys
from pathlib import Path

prior = Path(sys.argv[1])
merged = Path(sys.argv[2])
lines: list[str] = []
if prior.exists():
    for line in prior.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("gate_ok"):
            lines.append(line)
merged.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
print(f"Kept {len(lines)} passing row(s) in {merged}")
PY

export SMOKE_ONLY="path-tracing,torch-pipeline-parallelism,bn-fit-modify"
export SMOKE_APPEND=1
export SUMMARY_JSONL="$MERGED"
LOG="${OUT_ROOT}-run-remaining.log"

echo "=== A2 smoke gate: remaining 3 tasks ===" | tee "$LOG"
./experiments/run_a2_smoke_gate.sh 2>&1 | tee -a "$LOG"
cp "$MERGED" "$PRIOR"
echo "Final summary: $PRIOR"
