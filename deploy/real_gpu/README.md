# Real GPU validation

该部署用于同一物理主机上的 2×RTX 4090，版本建议固定为：

- vLLM 0.28.0
- LMCache 0.5.4
- NIXL 1.4.0
- Qwen2.5-7B-Instruct，FP16，`max-model-len=8192`

## 1. 环境和模型

```bash
uv venv .venv --python 3.12
UV_TORCH_BACKEND=auto uv pip install \
  -e '.[api,dev]' vllm==0.28.0 lmcache==0.5.4 nixl==1.4.0

HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

## 2. 硬件记录

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id \
  --format=csv,noheader > artifacts/real-gpu/gpu-inventory.txt
nvidia-smi topo -m > artifacts/real-gpu/gpu-topology.txt
```

## 3. Aggregated + tiered cache baseline

```bash
bash deploy/real_gpu/launch_aggregated.sh
pdserve gpu-benchmark --url http://127.0.0.1:8001 \
  --model Qwen2.5-7B-Instruct --requests 20 --concurrency 2 \
  --input-tokens 2048 --output-tokens 128 --shared-prefix \
  --output artifacts/real-gpu/aggregated-shared-prefix.json
bash deploy/real_gpu/stop.sh
```

## 4. 1P1D + NIXL + tiered cache

```bash
bash deploy/real_gpu/launch_pd.sh
pdserve gpu-benchmark --url http://127.0.0.1:8000 \
  --model Qwen2.5-7B-Instruct --requests 20 --concurrency 2 \
  --input-tokens 2048 --output-tokens 128 --shared-prefix \
  --output artifacts/real-gpu/pd-shared-prefix.json
bash deploy/real_gpu/stop.sh
```

`gpu-benchmark` 以 SSE 首个非空 token 的到达时间计算 TTFT，以后续 token
平均间隔计算 TPOT，并保存每个请求的原始指标。报告必须同时保留
成功率、SLO 达标率与失败原因，不得只挑选最优请求。

对照官方设计：

- <https://docs.vllm.ai/en/stable/features/disagg_prefill/>
- <https://docs.vllm.ai/en/stable/examples/disaggregated/lmcache/>
- <https://docs.lmcache.ai/disaggregated_prefill/nixl/1p1d.html>
