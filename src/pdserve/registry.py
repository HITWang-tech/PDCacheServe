from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Dict, List, Optional

from .models import Worker, WorkerRole


class WorkerRegistry:
    def __init__(self, heartbeat_timeout_seconds: float = 15.0, clock=time.time) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._clock = clock
        self._workers: Dict[str, Worker] = {}
        self._lock = threading.RLock()

    def register(self, worker: Worker) -> Worker:
        now = self._clock()
        with self._lock:
            stored = replace(worker, last_heartbeat=now, healthy=True)
            self._workers[worker.worker_id] = stored
            return stored

    def heartbeat(self, worker_id: str, **metrics: object) -> Worker:
        allowed = {
            "queue_depth",
            "active_sequences",
            "gpu_utilization",
            "kv_utilization",
            "prefill_ms_per_token",
            "decode_ms_per_token",
            "bandwidth_gbps",
            "metadata",
        }
        unknown = set(metrics) - allowed
        if unknown:
            raise ValueError("unknown heartbeat metrics: " + ", ".join(sorted(unknown)))
        with self._lock:
            worker = self._workers[worker_id]
            stored = replace(worker, last_heartbeat=self._clock(), healthy=True, **metrics)
            self._workers[worker_id] = stored
            return stored

    def get(self, worker_id: str) -> Optional[Worker]:
        self.fence_stale()
        with self._lock:
            return self._workers.get(worker_id)

    def healthy(self, role: Optional[WorkerRole] = None) -> List[Worker]:
        self.fence_stale()
        with self._lock:
            return [
                worker
                for worker in self._workers.values()
                if worker.healthy and (role is None or worker.role == role)
            ]

    def fence_stale(self) -> List[str]:
        now = self._clock()
        fenced: List[str] = []
        with self._lock:
            for worker_id, worker in list(self._workers.items()):
                if worker.healthy and now - worker.last_heartbeat > self.heartbeat_timeout_seconds:
                    self._workers[worker_id] = replace(worker, healthy=False)
                    fenced.append(worker_id)
        return fenced

    def all(self) -> List[Worker]:
        self.fence_stale()
        with self._lock:
            return list(self._workers.values())
