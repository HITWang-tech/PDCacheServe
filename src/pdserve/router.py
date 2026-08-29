from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .cache import CacheEntry, KVDirectory
from .models import CacheTier, InferenceRequest, RouteCandidate, RouteDecision, Worker, WorkerRole
from .registry import WorkerRegistry


@dataclass(frozen=True)
class RouterWeights:
    queue: float = 1.0
    ttft: float = 1.0
    tpot: float = 12.0
    transfer: float = 0.8
    utilization: float = 250.0
    cache_hit: float = 0.25


class SLORouter:
    def __init__(
        self,
        registry: WorkerRegistry,
        cache: KVDirectory,
        weights: Optional[RouterWeights] = None,
        reject_on_slo_miss: bool = True,
    ) -> None:
        self.registry = registry
        self.cache = cache
        self.weights = weights or RouterWeights()
        self.reject_on_slo_miss = reject_on_slo_miss

    def route(self, request: InferenceRequest) -> RouteDecision:
        prefillers = self.registry.healthy(WorkerRole.PREFILL)
        decoders = self.registry.healthy(WorkerRole.DECODE)
        if not prefillers or not decoders:
            return RouteDecision(request.request_id, False, "no_healthy_pd_pair")

        candidates: List[RouteCandidate] = []
        for prefill in prefillers:
            for decode in decoders:
                if not prefill.layout.compatible_with(decode.layout):
                    continue
                if prefill.layout.model_id != request.model_id:
                    continue
                cache_entry = self._best_cache(request, prefill, decode)
                candidates.append(self._candidate(request, prefill, decode, cache_entry))

        if not candidates:
            return RouteDecision(request.request_id, False, "no_compatible_pd_pair")
        candidates.sort(key=lambda item: (item.score, item.prefill_worker, item.decode_worker))
        best = candidates[0]
        meets_slo = (
            best.predicted_ttft_ms <= request.ttft_slo_ms
            and best.predicted_tpot_ms <= request.tpot_slo_ms
        )
        if self.reject_on_slo_miss and not meets_slo:
            return RouteDecision(
                request.request_id,
                False,
                "predicted_slo_miss",
                best,
                candidates[1:5],
            )
        return RouteDecision(request.request_id, True, "admitted", best, candidates[1:5])

    def _best_cache(
        self, request: InferenceRequest, prefill: Worker, decode: Worker
    ) -> Optional[CacheEntry]:
        if not request.prefix_hash:
            return None
        entries = self.cache.lookup(request.prefix_hash, prefill.layout)
        if not entries:
            return None

        def locality(entry: CacheEntry) -> tuple:
            local_decode = entry.location == decode.worker_id and entry.tier == CacheTier.GPU
            local_prefill = entry.location == prefill.worker_id and entry.tier == CacheTier.GPU
            return (not local_decode, not local_prefill, -entry.token_count)

        return min(entries, key=locality)

    def _candidate(
        self,
        request: InferenceRequest,
        prefill: Worker,
        decode: Worker,
        cache_entry: Optional[CacheEntry],
    ) -> RouteCandidate:
        cached_tokens = min(
            request.prefix_tokens,
            cache_entry.token_count if cache_entry else 0,
            request.input_tokens,
        )
        uncached_tokens = max(1, request.input_tokens - cached_tokens)
        queue_ms = prefill.queue_depth * max(
            5.0, request.input_tokens * prefill.prefill_ms_per_token
        )
        prefill_ms = uncached_tokens * prefill.prefill_ms_per_token
        transfer_ms = self._transfer_ms(prefill, decode, cache_entry, request, cached_tokens)
        predicted_ttft = queue_ms + prefill_ms + transfer_ms + decode.decode_ms_per_token
        predicted_tpot = decode.decode_ms_per_token * (
            1.0 + 0.10 * decode.active_sequences + 0.35 * decode.gpu_utilization
        )
        score = (
            self.weights.queue * (prefill.queue_depth + decode.active_sequences)
            + self.weights.ttft * predicted_ttft
            + self.weights.tpot * predicted_tpot
            + self.weights.transfer * transfer_ms
            + self.weights.utilization * (prefill.gpu_utilization + decode.gpu_utilization)
            - self.weights.cache_hit * cached_tokens
        )
        return RouteCandidate(
            prefill_worker=prefill.worker_id,
            decode_worker=decode.worker_id,
            cached_tokens=cached_tokens,
            cache_tier=cache_entry.tier if cache_entry else None,
            predicted_ttft_ms=round(predicted_ttft, 3),
            predicted_tpot_ms=round(predicted_tpot, 3),
            predicted_transfer_ms=round(transfer_ms, 3),
            score=round(score, 3),
        )

    @staticmethod
    def _transfer_ms(
        prefill: Worker,
        decode: Worker,
        cache_entry: Optional[CacheEntry],
        request: InferenceRequest,
        cached_tokens: int,
    ) -> float:
        if (
            cache_entry
            and cache_entry.location == decode.worker_id
            and cache_entry.tier == CacheTier.GPU
        ):
            return 0.05
        tokens_to_move = max(cached_tokens, request.input_tokens)
        size_bytes = tokens_to_move * prefill.layout.kv_bytes_per_token
        bandwidth_gbps = min(prefill.bandwidth_gbps, decode.bandwidth_gbps)
        if prefill.zone != decode.zone:
            bandwidth_gbps *= 0.55
        if cache_entry and cache_entry.tier == CacheTier.SSD:
            bandwidth_gbps = min(bandwidth_gbps, 6.0)
        elif cache_entry and cache_entry.tier == CacheTier.CPU:
            bandwidth_gbps = min(bandwidth_gbps, 18.0)
        return 0.15 + size_bytes * 8 / max(bandwidth_gbps * 1_000_000_000, 1) * 1000
