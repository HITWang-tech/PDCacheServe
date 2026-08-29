from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    arrival_ms: float
    input_tokens: int
    output_tokens: int
    prefix_hash: str
    prefix_tokens: int
    ttft_slo_ms: float
    tpot_slo_ms: float


@dataclass
class SimWorker:
    worker_id: str
    gpu_model: str
    prefill_ms_per_token: float
    decode_ms_per_token: float
    bandwidth_gbps: float
    decode_capacity: int
    available_ms: float = 0.0
    busy_ms: float = 0.0
    cache: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    cache_hit_tokens: int
    transfer_ms: float
    slo_met: bool


GPU_PROFILES: Dict[str, Tuple[float, float, float, int]] = {
    "A100-80GB": (0.030, 6.8, 160.0, 14),
    "RTX-4090-24GB": (0.047, 8.6, 24.0, 10),
    "RTX-3090-24GB": (0.080, 13.2, 20.0, 7),
}


def generate_trace(
    workload: str,
    count: int,
    seed: int,
    arrival_rate_rps: float,
    shared_prefix_ratio: float,
) -> List[TraceRequest]:
    if workload not in {"short", "long", "mixed"}:
        raise ValueError("workload must be short, long, or mixed")
    rng = random.Random(seed)
    arrival_ms = 0.0
    result: List[TraceRequest] = []
    shared_prefixes = [f"shared-{index}" for index in range(16)]
    for index in range(count):
        arrival_ms += rng.expovariate(arrival_rate_rps) * 1000.0
        if workload == "short":
            input_tokens = rng.randint(128, 512)
            output_tokens = rng.randint(64, 192)
            ttft_slo, tpot_slo = 150.0, 20.0
        elif workload == "long":
            input_tokens = rng.randint(4096, 8192)
            output_tokens = rng.randint(64, 256)
            ttft_slo, tpot_slo = 700.0, 20.0
        else:
            if rng.random() < 0.45:
                input_tokens = rng.randint(256, 1024)
                output_tokens = rng.randint(64, 256)
            else:
                input_tokens = rng.randint(2048, 8192)
                output_tokens = rng.randint(128, 512)
            ttft_slo, tpot_slo = 600.0, 20.0
        if rng.random() < shared_prefix_ratio:
            prefix_hash = rng.choice(shared_prefixes)
            prefix_tokens = max(16, int(input_tokens * rng.uniform(0.45, 0.80)))
        else:
            prefix_hash = hashlib.sha1(f"{seed}:{index}".encode()).hexdigest()[:12]
            prefix_tokens = max(16, int(input_tokens * 0.20))
        result.append(
            TraceRequest(
                request_id=f"{workload}-{seed}-{index}",
                arrival_ms=arrival_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prefix_hash=prefix_hash,
                prefix_tokens=prefix_tokens,
                ttft_slo_ms=ttft_slo,
                tpot_slo_ms=tpot_slo,
            )
        )
    return result


def _workers(prefix: str, models: Sequence[str]) -> List[SimWorker]:
    result = []
    for index, model in enumerate(models):
        prefill, decode, bandwidth, capacity = GPU_PROFILES[model]
        result.append(
            SimWorker(f"{prefix}-{index}", model, prefill, decode, bandwidth, capacity)
        )
    return result


def simulate(
    requests: Iterable[TraceRequest],
    policy: str,
    gpu_models: Sequence[str] = (
        "RTX-4090-24GB",
        "RTX-3090-24GB",
        "RTX-4090-24GB",
        "RTX-3090-24GB",
    ),
    kv_bytes_per_token: int = 131_072,
) -> Dict[str, object]:
    requests = list(requests)
    if policy not in {"aggregated", "pd-round-robin", "pd-load-aware", "pd-kv-aware"}:
        raise ValueError("unsupported policy: " + policy)
    if len(gpu_models) < 2:
        raise ValueError("at least two GPUs are required")

    if policy == "aggregated":
        results, workers = _simulate_aggregated(requests, gpu_models)
    else:
        split = max(1, len(gpu_models) // 2)
        prefillers = _workers("p", gpu_models[:split])
        decoders = _workers("d", gpu_models[split:])
        results = _simulate_pd(requests, policy, prefillers, decoders, kv_bytes_per_token)
        workers = prefillers + decoders
    return summarize(policy, requests, results, workers)


def _simulate_aggregated(
    requests: List[TraceRequest], gpu_models: Sequence[str]
) -> Tuple[List[RequestResult], List[SimWorker]]:
    workers = _workers("a", gpu_models)
    results: List[RequestResult] = []
    for request in requests:
        def projected(worker: SimWorker, current: TraceRequest = request) -> float:
            cached = min(worker.cache.get(current.prefix_hash, 0), current.prefix_tokens)
            return max(worker.available_ms, current.arrival_ms) + (
                current.input_tokens - cached
            ) * worker.prefill_ms_per_token

        worker = min(workers, key=lambda item: (projected(item), item.worker_id))
        cached_tokens = min(worker.cache.get(request.prefix_hash, 0), request.prefix_tokens)
        uncached_tokens = max(1, request.input_tokens - cached_tokens)
        start = max(worker.available_ms, request.arrival_ms)
        prefill_ms = uncached_tokens * worker.prefill_ms_per_token
        # Co-location interference: long prefills and queueing disturb token cadence.
        contention = 1.0 + min(
            0.75,
            request.input_tokens / 8192.0 * 0.55
            + max(0.0, start - request.arrival_ms) / 5000.0,
        )
        tpot = worker.decode_ms_per_token * contention
        ttft = start + prefill_ms + tpot - request.arrival_ms
        e2e = ttft + max(0, request.output_tokens - 1) * tpot
        decode_service_ms = request.output_tokens * tpot / worker.decode_capacity
        worker.available_ms = start + prefill_ms + decode_service_ms
        worker.busy_ms += prefill_ms + decode_service_ms
        worker.cache[request.prefix_hash] = max(
            worker.cache.get(request.prefix_hash, 0), request.prefix_tokens
        )
        results.append(
            RequestResult(
                request.request_id,
                ttft,
                tpot,
                e2e,
                cached_tokens,
                0.0,
                ttft <= request.ttft_slo_ms and tpot <= request.tpot_slo_ms,
            )
        )
    return results, workers


def _simulate_pd(
    requests: List[TraceRequest],
    policy: str,
    prefillers: List[SimWorker],
    decoders: List[SimWorker],
    kv_bytes_per_token: int,
) -> List[RequestResult]:
    results: List[RequestResult] = []
    p_rr = d_rr = 0
    for request in requests:
        if policy == "pd-round-robin":
            prefill = prefillers[p_rr % len(prefillers)]
            decode = decoders[d_rr % len(decoders)]
            p_rr += 1
            d_rr += 1
        elif policy == "pd-load-aware":
            prefill = min(
                prefillers,
                key=lambda item: (
                    max(item.available_ms, request.arrival_ms)
                    + request.input_tokens * item.prefill_ms_per_token,
                    item.worker_id,
                ),
            )
            decode = min(decoders, key=lambda item: (item.available_ms, item.worker_id))
        else:
            def pair_score(
                pair: Tuple[SimWorker, SimWorker], current: TraceRequest = request
            ) -> Tuple[float, str, str]:
                candidate_prefill, candidate_decode = pair
                hit = min(
                    candidate_decode.cache.get(current.prefix_hash, 0), current.prefix_tokens
                )
                uncached = max(1, current.input_tokens - hit)
                p_start = max(candidate_prefill.available_ms, current.arrival_ms)
                p_finish = p_start + uncached * candidate_prefill.prefill_ms_per_token
                if hit:
                    transfer = 0.05
                else:
                    size = current.input_tokens * kv_bytes_per_token
                    bandwidth = min(
                        candidate_prefill.bandwidth_gbps, candidate_decode.bandwidth_gbps
                    )
                    transfer = 0.15 + size * 8 / (bandwidth * 1e9) * 1000.0
                ttft = (
                    max(candidate_decode.available_ms, p_finish + transfer)
                    + candidate_decode.decode_ms_per_token
                    - current.arrival_ms
                )
                return ttft, candidate_prefill.worker_id, candidate_decode.worker_id

            prefill, decode = min(
                ((p_worker, d_worker) for p_worker in prefillers for d_worker in decoders),
                key=pair_score,
            )

        cached_tokens = min(decode.cache.get(request.prefix_hash, 0), request.prefix_tokens)
        uncached_tokens = max(1, request.input_tokens - cached_tokens)
        p_start = max(prefill.available_ms, request.arrival_ms)
        prefill_ms = uncached_tokens * prefill.prefill_ms_per_token
        p_finish = p_start + prefill_ms
        if cached_tokens and decode.cache.get(request.prefix_hash, 0) >= cached_tokens:
            transfer_ms = 0.05
        else:
            bytes_to_move = request.input_tokens * kv_bytes_per_token
            bandwidth = min(prefill.bandwidth_gbps, decode.bandwidth_gbps)
            transfer_ms = 0.15 + bytes_to_move * 8 / (bandwidth * 1e9) * 1000.0
        d_start = max(decode.available_ms, p_finish + transfer_ms)
        tpot = decode.decode_ms_per_token
        ttft = d_start + tpot - request.arrival_ms
        e2e = ttft + max(0, request.output_tokens - 1) * tpot
        prefill.available_ms = p_finish
        decode_service_ms = request.output_tokens * tpot / decode.decode_capacity
        decode.available_ms = d_start + decode_service_ms
        prefill.busy_ms += prefill_ms
        decode.busy_ms += decode_service_ms
        decode.cache[request.prefix_hash] = max(
            decode.cache.get(request.prefix_hash, 0), request.prefix_tokens
        )
        results.append(
            RequestResult(
                request.request_id,
                ttft,
                tpot,
                e2e,
                cached_tokens,
                transfer_ms,
                ttft <= request.ttft_slo_ms and tpot <= request.tpot_slo_ms,
            )
        )
    return results


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def summarize(
    policy: str,
    requests: List[TraceRequest],
    results: List[RequestResult],
    workers: List[SimWorker],
) -> Dict[str, object]:
    if not results:
        return {"policy": policy, "request_count": 0}
    start = min(item.arrival_ms for item in requests)
    finish = max(req.arrival_ms + result.e2e_ms for req, result in zip(requests, results))
    duration_seconds = max((finish - start) / 1000.0, 1e-9)
    total_prefix = sum(item.prefix_tokens for item in requests)
    total_cached = sum(item.cache_hit_tokens for item in results)
    horizon_ms = max(finish, 1.0)
    return {
        "policy": policy,
        "request_count": len(results),
        "slo_attainment": round(mean(item.slo_met for item in results), 4),
        "goodput_rps": round(sum(item.slo_met for item in results) / duration_seconds, 4),
        "throughput_rps": round(len(results) / duration_seconds, 4),
        "p50_ttft_ms": round(percentile([item.ttft_ms for item in results], 0.50), 3),
        "p95_ttft_ms": round(percentile([item.ttft_ms for item in results], 0.95), 3),
        "p95_tpot_ms": round(percentile([item.tpot_ms for item in results], 0.95), 3),
        "p95_e2e_ms": round(percentile([item.e2e_ms for item in results], 0.95), 3),
        "cache_hit_token_ratio": round(total_cached / max(total_prefix, 1), 4),
        "average_transfer_ms": round(mean(item.transfer_ms for item in results), 3),
        "average_gpu_utilization": round(
            mean(min(1.0, item.busy_ms / horizon_ms) for item in workers), 4
        ),
    }
