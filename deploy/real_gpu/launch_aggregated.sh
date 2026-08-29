#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL="${SERVED_MODEL:-Qwen2.5-7B-Instruct}"
LOG_DIR="${LOG_DIR:-$ROOT/artifacts/real-gpu/logs}"
mkdir -p "$LOG_DIR"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

CUDA_VISIBLE_DEVICES=0 \
setsid "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --port 8001 --dtype float16 --max-model-len 8192 \
  --gpu-memory-utilization 0.82 --enforce-eager \
  >"$LOG_DIR/aggregated.log" 2>&1 &
echo "$!" > "$LOG_DIR/aggregated.pid"
echo "Aggregated endpoint is starting at http://127.0.0.1:8001/v1"
