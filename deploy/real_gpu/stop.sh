#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/artifacts/real-gpu/logs}"

for name in proxy prefiller decoder aggregated; do
  pid_file="$LOG_DIR/$name.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill -- "-$pid" 2>/dev/null || kill "$pid"
      echo "stopped $name ($pid)"
    fi
  fi
done

# GNU setsid may fork when its caller is already a process-group leader. In that
# case $! records the short-lived wrapper instead of the service. Sweep only the
# project-specific command signatures so a stale pidfile cannot leave a proxy or
# vLLM engine behind on a billed rental node.
while read -r pid pgid command; do
  case "$command" in
    *".venv/bin/vllm serve"*|*" -m pdserve.nixl_proxy "*|*" -m pdserve.pd_proxy "*)
      kill -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      echo "stopped orphaned service ($pid)"
      ;;
  esac
done < <(ps -eo pid=,pgid=,args=)
