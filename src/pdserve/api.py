from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("install pdcacheserve[api]") from exc

from .cache import CacheEntry
from .control import ControlPlane
from .executor import HTTPPDExecutor, SimulatedExecutor
from .models import CacheTier, InferenceRequest, ModelLayout, Worker, WorkerRole


class LayoutBody(BaseModel):
    model_id: str
    dtype: str = "float16"
    block_size: int = 16
    kv_layout: str = "standard"
    kv_bytes_per_token: int = 131_072


class WorkerBody(BaseModel):
    worker_id: str
    role: WorkerRole
    endpoint: str
    gpu_model: str
    layout: LayoutBody
    zone: str = "node-0"
    prefill_ms_per_token: float = 0.08
    decode_ms_per_token: float = 12.0
    bandwidth_gbps: float = 24.0


class HeartbeatBody(BaseModel):
    queue_depth: int = 0
    active_sequences: int = 0
    gpu_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    kv_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    prefill_ms_per_token: float = 0.08
    decode_ms_per_token: float = 12.0
    bandwidth_gbps: float = 24.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CacheBody(BaseModel):
    prefix_hash: str
    layout: LayoutBody
    token_count: int
    tier: CacheTier
    location: str
    ttl_seconds: float = 300.0


class RouteBody(BaseModel):
    request_id: str = ""
    model_id: str
    input_tokens: int = Field(ge=1)
    max_new_tokens: int = Field(default=128, ge=1)
    prefix_hash: str = ""
    prefix_tokens: int = Field(default=0, ge=0)
    ttft_slo_ms: float = Field(default=2000.0, gt=0)
    tpot_slo_ms: float = Field(default=80.0, gt=0)
    payload: Dict[str, Any] = Field(default_factory=dict)


def create_app(control: Optional[ControlPlane] = None) -> FastAPI:

    plane = control or ControlPlane.create()
    app = FastAPI(title="PDCacheServe", version="0.1.0")

    def layout(body: LayoutBody) -> ModelLayout:
        return ModelLayout(**body.model_dump())

    def inference_request(body: RouteBody) -> InferenceRequest:
        return InferenceRequest(
            request_id=body.request_id or str(uuid.uuid4()),
            model_id=body.model_id,
            input_tokens=body.input_tokens,
            max_new_tokens=body.max_new_tokens,
            prefix_hash=body.prefix_hash,
            prefix_tokens=body.prefix_tokens,
            ttft_slo_ms=body.ttft_slo_ms,
            tpot_slo_ms=body.tpot_slo_ms,
        )

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/workers")
    def register_worker(body: WorkerBody) -> Dict[str, Any]:
        worker = Worker(
            worker_id=body.worker_id,
            role=body.role,
            endpoint=body.endpoint,
            gpu_model=body.gpu_model,
            layout=layout(body.layout),
            zone=body.zone,
            prefill_ms_per_token=body.prefill_ms_per_token,
            decode_ms_per_token=body.decode_ms_per_token,
            bandwidth_gbps=body.bandwidth_gbps,
        )
        return plane.register_worker(worker).as_dict()

    @app.post("/v1/workers/{worker_id}/heartbeat")
    def heartbeat(worker_id: str, body: HeartbeatBody) -> Dict[str, Any]:
        try:
            return plane.registry.heartbeat(worker_id, **body.model_dump()).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="worker not found") from exc

    @app.get("/v1/workers")
    def workers() -> list:
        return [worker.as_dict() for worker in plane.registry.all()]

    @app.post("/v1/cache")
    def register_cache(body: CacheBody) -> Dict[str, Any]:
        entry: CacheEntry = plane.cache.register(
            body.prefix_hash,
            layout(body.layout),
            body.token_count,
            body.tier,
            body.location,
            body.ttl_seconds,
        )
        return {**entry.__dict__, "tier": entry.tier.value, "layout": entry.layout.__dict__}

    @app.post("/v1/routes")
    def route(body: RouteBody) -> Dict[str, Any]:
        return plane.route(inference_request(body)).as_dict()

    @app.post("/v1/completions")
    def completion(body: RouteBody) -> Dict[str, Any]:
        request = inference_request(body)
        decision = plane.route(request)
        if not decision.admitted:
            raise HTTPException(status_code=429, detail=decision.as_dict())
        executor = (
            HTTPPDExecutor(plane.registry)
            if os.getenv("PDSERVE_EXECUTOR") == "http"
            else SimulatedExecutor()
        )
        return executor.infer(request, decision, body.payload)

    @app.post("/v1/maintenance/fence")
    def fence() -> Dict[str, Any]:
        return {"fenced": plane.fence_stale_workers(), "at": time.time()}

    @app.get("/metrics")
    def metrics() -> Dict[str, object]:
        return plane.metrics()

    app.state.control = plane
    return app


app = create_app()
