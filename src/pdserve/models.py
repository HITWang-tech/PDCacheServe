from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    AGGREGATED = "aggregated"


class CacheTier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    SSD = "ssd"


@dataclass(frozen=True)
class ModelLayout:
    model_id: str
    dtype: str = "float16"
    block_size: int = 16
    kv_layout: str = "standard"
    kv_bytes_per_token: int = 131_072

    def compatible_with(self, other: "ModelLayout") -> bool:
        return (
            self.model_id == other.model_id
            and self.dtype == other.dtype
            and self.block_size == other.block_size
            and self.kv_layout == other.kv_layout
        )


@dataclass
class Worker:
    worker_id: str
    role: WorkerRole
    endpoint: str
    gpu_model: str
    layout: ModelLayout
    zone: str = "node-0"
    queue_depth: int = 0
    active_sequences: int = 0
    gpu_utilization: float = 0.0
    kv_utilization: float = 0.0
    prefill_ms_per_token: float = 0.08
    decode_ms_per_token: float = 12.0
    bandwidth_gbps: float = 24.0
    last_heartbeat: float = 0.0
    healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["role"] = self.role.value
        return result


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    model_id: str
    input_tokens: int
    max_new_tokens: int
    prefix_hash: str = ""
    prefix_tokens: int = 0
    ttft_slo_ms: float = 2_000.0
    tpot_slo_ms: float = 80.0
    priority: int = 0
    arrival_ms: float = 0.0


@dataclass(frozen=True)
class RouteCandidate:
    prefill_worker: str
    decode_worker: str
    cached_tokens: int
    cache_tier: Optional[CacheTier]
    predicted_ttft_ms: float
    predicted_tpot_ms: float
    predicted_transfer_ms: float
    score: float


@dataclass
class RouteDecision:
    request_id: str
    admitted: bool
    reason: str
    candidate: Optional[RouteCandidate] = None
    alternatives: List[RouteCandidate] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        def encode(candidate: RouteCandidate) -> Dict[str, Any]:
            result = asdict(candidate)
            result["cache_tier"] = candidate.cache_tier.value if candidate.cache_tier else None
            return result

        return {
            "request_id": self.request_id,
            "admitted": self.admitted,
            "reason": self.reason,
            "candidate": encode(self.candidate) if self.candidate else None,
            "alternatives": [encode(item) for item in self.alternatives],
        }
