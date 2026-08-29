# Real GPU validation — 2026-08-29

> A successful 2 × A800 native-NIXL rerun is documented in
> [`a800/REPORT.md`](a800/REPORT.md). This file retains the earlier 2 × RTX 4090
> negative experiment because disabled CUDA P2P is an important deployment guardrail.

## Environment

- 2 × NVIDIA GeForce RTX 4090 24 GB on one rental node
- Driver 580.142; PyTorch 2.13.0+cu132
- vLLM 0.28.0; LMCache 0.5.4; NIXL 1.4.0
- Qwen2.5-7B-Instruct, FP16, max model length 8,192
- GPU topology: `NODE`, without NVLink; CUDA peer access is disabled in both directions

Raw hardware evidence is preserved in `gpu-inventory.txt`, `gpu-topology.txt`, and
`p2p-capability.txt`.

## Valid aggregated baseline

The single-engine vLLM baseline uses LMCache's 8 GB CPU + 5 GB SSD tiered cache.
The shared-prefix workload issued 20 requests at concurrency 2, targeting 2,048
input and 128 output tokens.

| Metric | Result |
|---|---:|
| Successful requests | 20 / 20 |
| P50 TTFT | 150.71 ms |
| P95 TTFT | 276.23 ms |
| P95 TPOT | 18.67 ms |
| P95 end-to-end latency | 2,585.63 ms |
| Request throughput | 0.812 req/s |
| SLO attainment (TTFT ≤ 2 s, TPOT ≤ 80 ms) | 100% |

Per-request measurements are in `aggregated-shared-prefix.json`.

## P/D validation decision

The complete 1P1D control path was exercised: tokenization, first-token prefill,
`kv_transfer_params` injection, NIXL/UCX initialization, ZMQ completion notification,
buffer-slot admission control, and decode continuation. HTTP requests completed, but
the decode continuation failed semantic consistency checks on this host. The issue was
also reproduced with LMCache's unmodified 0.5.4 example proxy and with both Qwen2.5-7B
and Qwen2.5-1.5B, so the measurements are **not accepted as valid P/D benchmark data**.

Disabling Prefiller CPU/SSD caching removed cross-model stale-cache contamination, but
did not make the direct GPU transfer reliable. The remaining hard incompatibility is the
rental platform's disabled CUDA peer access. PDCacheServe therefore fails fast on this
topology by default. `ALLOW_NO_P2P=1` exists only for reproduction and diagnostics.

## Required rerun

Use two GPUs on a host where `torch.cuda.can_device_access_peer(0, 1)` and the reverse
direction are both true—preferably 2 × A100/A800 SXM with NVLink. Then rerun the same
correctness sentinel before collecting P/D latency and throughput. No performance claim
should be made from the rejected RTX 4090 P/D run.
