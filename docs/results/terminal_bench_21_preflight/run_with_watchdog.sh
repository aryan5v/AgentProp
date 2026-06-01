#!/usr/bin/env bash
set -euo pipefail

mkdir -p "benchmark-results/terminal-bench-2.1/terminus-2-agentprop"
python experiments/run_with_watchdog.py \
  --timeout 21600 \
  --idle-timeout 1800 \
  --poll-interval 5 \
  --log "benchmark-results/terminal-bench-2.1/terminus-2-agentprop/launcher.log" \
  --status-json "benchmark-results/terminal-bench-2.1/terminus-2-agentprop/watchdog-status.json" \
  -- harbor run -d terminal-bench/terminal-bench-2-1 -a terminus-2 -m google/gemini-3.1-pro-preview --env modal
