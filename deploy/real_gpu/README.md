# Real GPU validation

该部署用于同一物理主机上支持双向 CUDA P2P 的双 GPU，
本次验证版本固定为：

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

## 3. Native aggregated baseline

```bash
bash deploy/real_gpu/launch_aggregated.sh
pdserve gpu-benchmark --url http://127.0.0.1:8001 \
  --model Qwen2.5-7B-Instruct --requests 20 --concurrency 2 \
  --input-tokens 2048 --output-tokens 128 --shared-prefix --ignore-eos \
  --output artifacts/real-gpu/aggregated-shared-prefix.json
bash deploy/real_gpu/stop.sh
```

## 4. 1P1D + native NIXL

```bash
bash deploy/real_gpu/launch_pd.sh
pdserve gpu-benchmark --url http://127.0.0.1:8000 \
  --model Qwen2.5-7B-Instruct --requests 20 --concurrency 2 \
  --input-tokens 2048 --output-tokens 128 --shared-prefix --ignore-eos \
  --output artifacts/real-gpu/pd-shared-prefix.json
bash deploy/real_gpu/stop.sh
```

启动脚本会先检查两张 GPU 是否支持双向 CUDA Peer Access。检查失败时默认拒绝
启动，因为某些云平台上的 RTX 4090 虽然位于同一节点，但会禁用 P2P，导致 NIXL
传输完成却产生错误 token。`ALLOW_NO_P2P=1` 仅用于复现和诊断，不得用于性能结论。

聚合基线不启用外部 KV Cache，避免把缓存收益混入 P/D 数据面对比。
分层 CPU/SSD KV Cache 由控制面 `KVDirectory` 与独立 LMCache 配置验证；
当前实时 P/D 传输使用 vLLM 原生 `NixlConnector`，避免已废弃的
LMCache 进程内连接器并发缓存污染问题。

`gpu-benchmark` 以 SSE 首个非空 token 的到达时间计算 TTFT，以后续 token
平均间隔计算 TPOT，并保存每个请求的原始指标。报告必须同时保留
成功率、SLO 达标率与失败原因，不得只挑选最优请求。

对照官方设计：

- <https://docs.vllm.ai/en/stable/features/disagg_prefill/>
- <https://docs.vllm.ai/en/stable/features/nixl_connector_usage/>
- <https://docs.lmcache.ai/disaggregated_prefill/nixl/1p1d.html>
