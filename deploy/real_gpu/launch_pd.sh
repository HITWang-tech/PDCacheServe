#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL="${SERVED_MODEL:-Qwen2.5-7B-Instruct}"
LOG_DIR="${LOG_DIR:-$ROOT/artifacts/real-gpu/logs}"
mkdir -p "$LOG_DIR" /root/autodl-tmp/lmcache/prefiller
export PATH="$ROOT/.venv/bin:$PATH"

wait_for_url() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 180); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is ready"
      return 0
    fi
    sleep 2
  done
  echo "$name failed to become ready" >&2
  return 1
}

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export UCX_TLS=cuda_ipc,cuda_copy,tcp

CUDA_VISIBLE_DEVICES=1 \
LMCACHE_CONFIG_FILE="$ROOT/deploy/real_gpu/configs/decoder.yaml" \
setsid "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --port 8200 --dtype float16 --max-model-len 8192 \
  --gpu-memory-utilization 0.82 --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumer1","skip_last_n_tokens":1}}' \
  >"$LOG_DIR/decoder.log" 2>&1 &
echo "$!" > "$LOG_DIR/decoder.pid"
wait_for_url http://127.0.0.1:8200/health decoder

CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE="$ROOT/deploy/real_gpu/configs/prefiller.yaml" \
setsid "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --port 8100 --dtype float16 --max-model-len 8192 \
  --gpu-memory-utilization 0.82 --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producer1"}}' \
  >"$LOG_DIR/prefiller.log" 2>&1 &
echo "$!" > "$LOG_DIR/prefiller.pid"
wait_for_url http://127.0.0.1:8100/health prefiller

setsid "$ROOT/.venv/bin/python" -m pdserve.pd_proxy \
  --host 0.0.0.0 --port 8000 \
  --prefill-url http://127.0.0.1:8100 \
  --decode-url http://127.0.0.1:8200 \
  >"$LOG_DIR/proxy.log" 2>&1 &
echo "$!" > "$LOG_DIR/proxy.pid"
wait_for_url http://127.0.0.1:8000/healthz proxy

echo "P/D endpoint: http://127.0.0.1:8000/v1"
