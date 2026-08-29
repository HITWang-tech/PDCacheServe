from pdserve.cache import KVDirectory
from pdserve.models import CacheTier, InferenceRequest, ModelLayout, Worker, WorkerRole
from pdserve.registry import WorkerRegistry
from pdserve.router import SLORouter


def build_router():
    layout = ModelLayout("qwen", kv_bytes_per_token=1024)
    registry = WorkerRegistry(heartbeat_timeout_seconds=1000)
    cache = KVDirectory()
    registry.register(
        Worker("p0", WorkerRole.PREFILL, "http://p0", "4090", layout, queue_depth=0)
    )
    registry.register(
        Worker("p1", WorkerRole.PREFILL, "http://p1", "3090", layout, queue_depth=5)
    )
    registry.register(
        Worker("d0", WorkerRole.DECODE, "http://d0", "4090", layout, active_sequences=2)
    )
    registry.register(
        Worker("d1", WorkerRole.DECODE, "http://d1", "3090", layout, active_sequences=0)
    )
    return SLORouter(registry, cache), cache, layout


def request(**overrides):
    values = dict(
        request_id="r1",
        model_id="qwen",
        input_tokens=1024,
        max_new_tokens=128,
        ttft_slo_ms=5000,
        tpot_slo_ms=100,
    )
    values.update(overrides)
    return InferenceRequest(**values)


def test_router_prefers_lower_load_pair_without_cache():
    router, _, _ = build_router()
    decision = router.route(request())
    assert decision.admitted
    assert decision.candidate.prefill_worker == "p0"
    assert decision.candidate.decode_worker == "d1"


def test_router_prefers_decode_local_gpu_cache():
    router, cache, layout = build_router()
    router.registry.heartbeat("d0", active_sequences=0)
    cache.register("shared", layout, 900, CacheTier.GPU, "d0")
    decision = router.route(request(prefix_hash="shared", prefix_tokens=900))
    assert decision.admitted
    assert decision.candidate.decode_worker == "d0"
    assert decision.candidate.cached_tokens == 900
    assert decision.candidate.predicted_transfer_ms == 0.05


def test_router_rejects_predicted_slo_miss():
    router, _, _ = build_router()
    decision = router.route(request(ttft_slo_ms=1, tpot_slo_ms=1))
    assert not decision.admitted
    assert decision.reason == "predicted_slo_miss"


def test_router_rejects_incompatible_layouts():
    router, _, _ = build_router()
    decision = router.route(request(model_id="other"))
    assert not decision.admitted
    assert decision.reason == "no_compatible_pd_pair"


def test_router_rejects_when_worker_pool_missing():
    registry = WorkerRegistry()
    cache = KVDirectory()
    registry.register(Worker("p", WorkerRole.PREFILL, "http://p", "4090", ModelLayout("qwen")))
    decision = SLORouter(registry, cache).route(request())
    assert not decision.admitted
    assert decision.reason == "no_healthy_pd_pair"
