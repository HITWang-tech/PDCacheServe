# 架构与设计决策

## 控制面与数据面分离

PDCacheServe 只在控制面保存 KV Cache 元数据。KV Tensor 不进入 FastAPI 或 Redis，
而是由 vLLM Connector、LMCache 或 NIXL 在数据面直接传输。这样可避免 Python 网关成为
带宽瓶颈，也能把路由策略与具体传输后端解耦。

## 兼容性硬约束

只有 `model_id`、`dtype`、`block_size` 与 `kv_layout` 全部一致的 Prefill/Decode
Worker 才能组成 Pair。生产部署还应校验模型 revision、TP/PP 布局和 Cache Connector
协议版本；兼容性失败必须拒绝，而不是尝试转换 KV Tensor。

## 分层 KV Cache

- L1 GPU：最低读取延迟、容量最小，优先 Decode 本地命中；
- L2 CPU DRAM：容量较大，通过 CUDA IPC/NIXL/PCIe 回填；
- L3 SSD：容量最大，仅适用于高复用长前缀，需计入读取和传输成本。

目录条目包含 Prefix Hash、模型布局、Token 数、字节数、位置、TTL 和 Lease。
有 Lease 的条目不会被过期或 LRU 回收；Worker 被 Fencing 时，其本地 Cache 条目失效。

## 路由目标

每个兼容 P/D Pair 的核心估算为：

```text
TTFT = prefill_queue + uncached_prefill + kv_transfer + first_decode_token
TPOT = decode_profile * concurrency_factor
KVTransfer = fixed_latency + kv_bytes / measured_bandwidth
```

最终 Score 综合 TTFT、TPOT、队列、GPU 利用率、传输开销和 Cache 命中收益。
若最优 Pair 仍无法满足请求 SLO，则返回 `predicted_slo_miss`，由上层选择排队、降级或拒绝。

## 故障模型

Worker 心跳超时后：

1. Registry 将 Worker 标记为不健康，不再参与新路由；
2. KV Directory 删除该 Worker 的本地 Cache 元数据；
3. 已持有 Lease 的真实请求应由数据面在超时后释放；
4. 客户端使用 request ID 保证重试幂等。

当前实现覆盖 1、2；分布式 Lease 协调和执行中请求迁移属于后续工作。

## Benchmark 边界

离散事件模拟器建模队列、continuous batching 槽位、异构设备画像、Prefix 复用、KV 字节数
和传输带宽。它用于验证策略方向和回归，不用于替代真实 GPU Benchmark。
