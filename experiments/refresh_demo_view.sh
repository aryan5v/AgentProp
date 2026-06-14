#!/usr/bin/env bash
# Symlink all Harbor job runs into one folder for `harbor view` demos.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/benchmark-results/composer-comparison"
VIEW="${OUT}/demo-view"
mkdir -p "$VIEW"

link_jobs() {
  local arm="$1" prefix="$2"
  local arm_dir="${OUT}/${arm}"
  [[ -d "$arm_dir" ]] || return 0
  for task_dir in "$arm_dir"/*; do
    [[ -d "$task_dir" ]] || continue
    local task
    task="$(basename "$task_dir")"
    for job_dir in "$task_dir"/20*; do
      [[ -d "$job_dir" ]] || continue
      [[ -f "$job_dir/config.json" ]] || continue
      local job_ts
      job_ts="$(basename "$job_dir")"
      local name="${prefix}-${task}__${job_ts}"
      ln -sfn "$job_dir" "${VIEW}/${name}"
    done
  done
}

link_jobs "a0-cursor-cli" "a0"
link_jobs "a2-agentprop" "a2"

echo "demo-view: $(find -L "$VIEW" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') jobs linked"
