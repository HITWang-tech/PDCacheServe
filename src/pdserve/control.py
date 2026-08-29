from __future__ import annotations

import threading
from collections import Counter
from typing import Dict, List

from .cache import KVDirectory
from .models import InferenceRequest, RouteDecision, Worker
from .registry import WorkerRegistry
from .router import SLORouter


class ControlPlane:
    def __init__(self, registry: WorkerRegistry, cache: KVDirectory, router: SLORouter) -> None:
        self.registry = registry
        self.cache = cache
        self.router = router
        self._metrics: Counter = Counter()
        self._lock = threading.Lock()

    @classmethod
    def create(cls) -> "ControlPlane":
        registry = WorkerRegistry()
        cache = KVDirectory()
        return cls(registry, cache, SLORouter(registry, cache))

    def register_worker(self, worker: Worker) -> Worker:
        stored = self.registry.register(worker)
        with self._lock:
            self._metrics["worker_registrations"] += 1
        return stored

    def route(self, request: InferenceRequest) -> RouteDecision:
        decision = self.router.route(request)
        with self._lock:
            self._metrics["route_requests"] += 1
            self._metrics["route_admitted" if decision.admitted else "route_rejected"] += 1
            self._metrics["predicted_cache_hit_tokens"] += (
                decision.candidate.cached_tokens if decision.candidate else 0
            )
        return decision

    def fence_stale_workers(self) -> List[str]:
        workers = self.registry.fence_stale()
        removed = sum(self.cache.remove_location(worker_id) for worker_id in workers)
        with self._lock:
            self._metrics["workers_fenced"] += len(workers)
            self._metrics["cache_entries_invalidated"] += removed
        return workers

    def metrics(self) -> Dict[str, object]:
        with self._lock:
            counters = dict(self._metrics)
        return {
            **counters,
            "healthy_workers": len(self.registry.healthy()),
            "cache_entries": len(self.cache.entries()),
            "cache_usage_bytes": self.cache.usage(),
        }
