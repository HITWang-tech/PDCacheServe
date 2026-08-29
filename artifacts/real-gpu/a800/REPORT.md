# A800 native NIXL validation — 2026-08-30

## Environment

- 2 × NVIDIA A800 80 GB PCIe on one rental node;
- bidirectional CUDA Peer Access: enabled; vLLM NIXL compatibility check: passed;
- PyTorch 2.13.0, vLLM 0.28.0, LMCache 0.5.4, NIXL 1.4.0;
- Qwen2.5-7B-Instruct, FP16, maximum model length 8,192;
- aggregated baseline: one GPU with native vLLM;
- disaggregated deployment: one prefill GPU, one decode GPU, native
  `NixlConnector` over UCX, plus the PDCacheServe proxy.

Hardware inventory, topology, peer-access preflight, software versions, and NIXL
transfer metrics are preserved next to this report.

## Correctness gate

Before benchmarking, each deployment received ten concurrent deterministic chat
requests asking for the capital of France. Both native aggregated vLLM and native
NIXL P/D returned exactly `Paris` for 10/10 requests. The raw responses are in
`aggregated-native-correctness.json` and `pd-native-correctness.json`.

An earlier LMCache in-process connector trial was rejected because concurrent
requests produced cross-request KV contamination and `Double unpin` warnings. The
model and GPU were ruled out by a clean native-vLLM control run. The production
data path was therefore migrated to vLLM's native `NixlConnector`; rejected runs
are not included in the performance comparison.

## Fixed-token benchmark

Every accepted request used the same shared-prefix prompt, targeted 2,048 input
tokens (1,584 reported by the server tokenizer), forced exactly 128 output tokens
with `ignore_eos`, and used TTFT ≤ 2 s / TPOT ≤ 80 ms as the SLO. Results are
single-run engineering validation numbers, not claims about all hardware or loads.

| Concurrency | Deployment | Success | P95 TTFT | P95 TPOT | P95 E2E | Throughput | SLO attainment |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | Aggregated | 20/20 | 196.30 ms | 14.50 ms | 2,038.68 ms | 1.045 req/s | 100% |
| 2 | Native NIXL P/D | 20/20 | 410.48 ms | 14.99 ms | 2,257.95 ms | 0.991 req/s | 100% |
| 8 | Aggregated | 40/40 | 271.75 ms | 15.45 ms | 2,219.33 ms | 3.735 req/s | 100% |
| 8 | Native NIXL P/D | 40/40 | 348.62 ms | 15.27 ms | 2,278.42 ms | 3.626 req/s | 100% |

At concurrency 8, P/D reduced P95 TPOT by 1.16% while request throughput was 2.94%
lower and P95 TTFT was 28.29% higher than the one-GPU aggregated baseline. This is
the expected trade-off on a small single-node workload: P/D provides independent
prefill/decode scaling and tail-ITL isolation, but KV handoff and the extra proxy hop
do not make aggregate throughput inherently higher.

NIXL reported successful transfers, including an observed 255.12 MB/s interval;
the complete periodic transfer metrics are in `nixl-transfer-metrics.txt`.

## Verification and shutdown

- Ruff checks passed;
- 27 automated tests passed;
- both accepted benchmark files contain 100% successful requests and no errors;
- all inference services were stopped after collection;
- `nvidia-smi` reported no compute processes before the instance handoff.
