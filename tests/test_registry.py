import pytest

from pdserve.models import ModelLayout, Worker, WorkerRole
from pdserve.registry import WorkerRegistry


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def worker(worker_id="p0"):
    return Worker(worker_id, WorkerRole.PREFILL, "http://worker", "4090", ModelLayout("qwen"))


def test_stale_worker_is_fenced():
    clock = Clock()
    registry = WorkerRegistry(heartbeat_timeout_seconds=5, clock=clock)
    registry.register(worker())
    clock.now = 16
    assert registry.fence_stale() == ["p0"]
    assert registry.healthy() == []


def test_heartbeat_updates_metrics_and_recovers_health():
    clock = Clock()
    registry = WorkerRegistry(heartbeat_timeout_seconds=5, clock=clock)
    registry.register(worker())
    clock.now = 16
    registry.fence_stale()
    updated = registry.heartbeat("p0", queue_depth=3, gpu_utilization=0.7)
    assert updated.healthy
    assert updated.queue_depth == 3


def test_heartbeat_rejects_unknown_metric():
    registry = WorkerRegistry()
    registry.register(worker())
    with pytest.raises(ValueError):
        registry.heartbeat("p0", dangerous_field=True)
