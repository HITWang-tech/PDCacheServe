# PDCacheServe 离散事件仿真报告

> 本报告是控制面策略仿真，不是生产 GPU 实测。真实双卡结果需要单独执行。

- 随机种子：10
- 每个 workload/seed 请求数：300
- 策略运行数：120
- 累计策略请求：36000

| Workload | Policy | SLO attainment | Goodput req/s | P95 TTFT ms | P95 TPOT ms | Cache hit tokens | GPU util |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long | aggregated | 99.43% | 1.502 | 401.9 | 15.7 | 66.33% | 15.63% |
| long | pd-kv-aware | 87.77% | 1.327 | 767.5 | 12.3 | 69.97% | 13.45% |
| long | pd-load-aware | 84.30% | 1.270 | 809.0 | 13.2 | 63.49% | 15.96% |
| long | pd-round-robin | 75.40% | 1.135 | 1017.7 | 13.2 | 63.46% | 18.61% |
| mixed | aggregated | 99.67% | 1.804 | 383.7 | 13.9 | 32.91% | 19.64% |
| mixed | pd-kv-aware | 83.50% | 1.516 | 743.1 | 13.2 | 36.21% | 18.82% |
| mixed | pd-load-aware | 80.53% | 1.454 | 788.3 | 13.2 | 29.44% | 20.70% |
| mixed | pd-round-robin | 70.07% | 1.259 | 1083.8 | 13.2 | 29.24% | 23.34% |
| short | aggregated | 100.00% | 3.022 | 32.8 | 10.7 | 18.36% | 10.01% |
| short | pd-kv-aware | 98.80% | 2.973 | 74.4 | 13.2 | 19.04% | 11.23% |
| short | pd-load-aware | 98.20% | 2.955 | 83.0 | 13.2 | 14.49% | 13.43% |
| short | pd-round-robin | 94.73% | 2.846 | 148.0 | 13.2 | 14.44% | 14.52% |

## 关键结论

- `short`：KV-aware 相比 PD Round Robin 的 P95 TTFT 降低 **49.71%**，Goodput 提升 **4.47%**，SLO 达标率提高 **4.07%**。
- `mixed`：KV-aware 相比 PD Round Robin 的 P95 TTFT 降低 **31.44%**，Goodput 提升 **20.41%**，SLO 达标率提高 **13.43%**。
- `long`：KV-aware 相比 PD Round Robin 的 P95 TTFT 降低 **24.58%**，Goodput 提升 **16.88%**，SLO 达标率提高 **12.37%**。
- 当前 PCIe 异构画像下，Aggregated 仍取得更高的绝对 Goodput 和更低 TTFT，
  说明 PD 分离并非无条件收益；真实部署必须同时考虑 KV 传输成本、SLO 类型与负载。

## 说明

仿真使用相同四卡异构资源和相同请求轨迹比较 Aggregated、PD Round Robin、
PD Load Aware 和 PD KV Aware。延迟模型参数来自配置中的设备画像，不代表特定
线上集群。简历中不得把这些结果描述为真实 GPU 集群实测。
