# 双 GPU vLLM / LMCache / NIXL 验证指南

## 推荐环境

- 同一主机 2 张 RTX 4090 24GB，或 2 张 A100；
- NVIDIA Driver、CUDA 与容器运行时版本满足所选 vLLM 镜像要求；
- Qwen2.5-7B-Instruct，Prefill 与 Decode 使用完全相同的 revision、dtype、block size；
- 锁定 vLLM、LMCache、NIXL 镜像/包版本，不直接跟随 `latest`。

官方示例：

- <https://docs.vllm.ai/en/stable/examples/disaggregated/lmcache/>
- <https://docs.vllm.ai/en/stable/features/disagg_prefill/>
- <https://docs.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview>

## 接入顺序

1. 先运行单实例 Aggregated vLLM，记录同一 Trace 的 TTFT、TPOT、吞吐和显存；
2. 使用官方 LMCache/NIXL 示例启动固定 1P1D，确认 KV Transfer 日志和输出一致性；
3. 在 Prefill、Decode 实例旁增加轻量 Sidecar，实现：
   - `POST /internal/prefill`：执行 Prefill，返回 KV transfer handle 与布局元数据；
   - `POST /internal/decode`：消费 handle，继续 Decode 并返回生成结果；
4. 向 PDCacheServe 注册两个 Worker，并持续上报队列、active sequence、GPU/KV 利用率；
5. 设置 `PDSERVE_EXECUTOR=http` 后通过 `/v1/completions` 执行真实路径；
6. 分别运行短输入、长输入、混合长度和共享前缀 Trace。

## 必须验证

- Prefill/Decode 的模型、dtype、block size、KV layout 完全一致；
- KV Transfer 没有经过 Python 网关复制 Tensor；
- 4090 的 PCIe 与 A100 NVLink/NVSwitch 结果分开记录；
- 压测客户端部署在服务节点同网段，避免端口转发污染 P95；
- 同时报告 Aggregated 负面/正面结果，不只展示最有利 workload；
- 真实结果写入 `artifacts/real-gpu/`，未运行前不得在简历填写百分比。
