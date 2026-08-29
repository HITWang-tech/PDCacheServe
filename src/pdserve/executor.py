from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

from .models import InferenceRequest, RouteDecision
from .registry import WorkerRegistry


class PDExecutor(Protocol):
    def infer(
        self, request: InferenceRequest, decision: RouteDecision, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...


@dataclass
class SimulatedExecutor:
    def infer(
        self, request: InferenceRequest, decision: RouteDecision, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not decision.admitted or not decision.candidate:
            raise RuntimeError("request was not admitted")
        return {
            "id": request.request_id,
            "object": "text_completion",
            "model": request.model_id,
            "choices": [{"index": 0, "text": payload.get("mock_text", "[simulated output]")}],
            "route": decision.as_dict(),
            "usage": {
                "prompt_tokens": request.input_tokens,
                "completion_tokens": request.max_new_tokens,
            },
        }


class HTTPPDExecutor:
    """Adapter for vLLM/LMCache sidecars implementing the internal PD contract."""

    def __init__(self, registry: WorkerRegistry, timeout_seconds: float = 120.0) -> None:
        self.registry = registry
        self.timeout_seconds = timeout_seconds

    def infer(
        self, request: InferenceRequest, decision: RouteDecision, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install pdcacheserve[api] for HTTP execution") from exc
        if not decision.admitted or not decision.candidate:
            raise RuntimeError("request was not admitted")
        prefill = self.registry.get(decision.candidate.prefill_worker)
        decode = self.registry.get(decision.candidate.decode_worker)
        if not prefill or not decode:
            raise RuntimeError("selected worker disappeared")
        with httpx.Client(timeout=self.timeout_seconds) as client:
            prefill_response = client.post(
                prefill.endpoint.rstrip("/") + "/internal/prefill",
                json={"request": request.__dict__, "payload": payload},
            )
            prefill_response.raise_for_status()
            transfer = prefill_response.json()
            decode_response = client.post(
                decode.endpoint.rstrip("/") + "/internal/decode",
                json={"request": request.__dict__, "payload": payload, "transfer": transfer},
            )
            decode_response.raise_for_status()
            return decode_response.json()
