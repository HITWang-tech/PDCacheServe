from pdserve.cache import KVDirectory
from pdserve.control import ControlPlane
from pdserve.models import CacheTier, InferenceRequest, ModelLayout, Worker, WorkerRole
from pdserve.registry import WorkerRegistry
from pdserve.router import SLORouter


class Clock:
    def __init__(self):
        self.now = 1.0

    def __call__(self):
        return self.now


def test_fencing_worker_invalidates_owned_cache():
    clock = Clock()
    registry = WorkerRegistry(heartbeat_timeout_seconds=5, clock=clock)
    cache = KVDirectory(clock=clock)
    plane = ControlPlane(registry, cache, SLORouter(registry, cache))
    layout = ModelLayout("qwen")
    plane.register_worker(Worker("p", WorkerRole.PREFILL, "http://p", "4090", layout))
    plane.register_worker(Worker("d", WorkerRole.DECODE, "http://d", "4090", layout))
    cache.register("prefix", layout, 10, CacheTier.GPU, "d")
    clock.now = 7
    assert set(plane.fence_stale_workers()) == {"p", "d"}
    assert cache.entries() == []
    assert plane.metrics()["workers_fenced"] == 2


def test_control_plane_records_admission_metrics():
    plane = ControlPlane.create()
    layout = ModelLayout("qwen")
    plane.register_worker(Worker("p", WorkerRole.PREFILL, "http://p", "4090", layout))
    plane.register_worker(Worker("d", WorkerRole.DECODE, "http://d", "4090", layout))
    decision = plane.route(InferenceRequest("r", "qwen", 100, 20))
    assert decision.admitted
    assert plane.metrics()["route_admitted"] == 1
