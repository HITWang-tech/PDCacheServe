from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RequestMetric:
    request_id: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: str = ""


def percentile(values: List[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _prompt(target_tokens: int, shared_prefix: bool) -> str:
    prefix = (
        "You are a careful infrastructure assistant. Return a concise technical answer. " * 64
        if shared_prefix
        else ""
    )
    unit = "Explain one reliability consideration for distributed LLM inference. "
    return (prefix + unit * max(1, target_tokens // 10))[: target_tokens * 5]


def run_request(
    request_id: int,
    url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    shared_prefix: bool,
    timeout_seconds: float = 600.0,
) -> RequestMetric:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install pdcacheserve[api]") from exc
    started = time.perf_counter()
    first_token_at: Optional[float] = None
    completion_tokens = 0
    prompt_tokens = 0
    body = {
        "model": model,
        "prompt": _prompt(input_tokens, shared_prefix),
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            with client.stream("POST", url.rstrip("/") + "/v1/completions", json=body) as res:
                res.raise_for_status()
                for line in res.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    usage = event.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    choices = event.get("choices") or []
                    if choices and choices[0].get("text"):
                        completion_tokens = max(completion_tokens, 1)
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
        ended = time.perf_counter()
        first_token_at = first_token_at or ended
        generated_intervals = max(1, completion_tokens - 1)
        return RequestMetric(
            request_id=request_id,
            ttft_ms=(first_token_at - started) * 1000,
            tpot_ms=(ended - first_token_at) * 1000 / generated_intervals,
            e2e_ms=(ended - started) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=True,
        )
    except Exception as exc:  # pragma: no cover - exercised against a real endpoint
        ended = time.perf_counter()
        return RequestMetric(
            request_id=request_id,
            ttft_ms=math.nan,
            tpot_ms=math.nan,
            e2e_ms=(ended - started) * 1000,
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            error=str(exc),
        )


def summarize(
    metrics: List[RequestMetric], ttft_slo_ms: float, tpot_slo_ms: float
) -> Dict[str, float]:
    successful = [item for item in metrics if item.success]
    elapsed_s = sum(item.e2e_ms for item in successful) / 1000
    slo_met = [
        item
        for item in successful
        if item.ttft_ms <= ttft_slo_ms and item.tpot_ms <= tpot_slo_ms
    ]
    return {
        "requests": len(metrics),
        "successful": len(successful),
        "p50_ttft_ms": percentile([item.ttft_ms for item in successful], 0.50),
        "p95_ttft_ms": percentile([item.ttft_ms for item in successful], 0.95),
        "p95_tpot_ms": percentile([item.tpot_ms for item in successful], 0.95),
        "p95_e2e_ms": percentile([item.e2e_ms for item in successful], 0.95),
        "slo_attainment": len(slo_met) / len(metrics) if metrics else 0.0,
        "sequential_token_throughput": (
            sum(item.completion_tokens for item in successful) / elapsed_s if elapsed_s else 0.0
        ),
    }


def run_gpu_benchmark(
    url: str,
    model: str,
    output_path: str,
    requests: int = 20,
    concurrency: int = 2,
    input_tokens: int = 2048,
    output_tokens: int = 128,
    shared_prefix: bool = False,
    ttft_slo_ms: float = 2000.0,
    tpot_slo_ms: float = 80.0,
) -> Dict[str, object]:
    metrics: List[RequestMetric] = []
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_request,
                request_id,
                url,
                model,
                input_tokens,
                output_tokens,
                shared_prefix,
            )
            for request_id in range(requests)
        ]
        for future in as_completed(futures):
            metrics.append(future.result())
    wall_seconds = time.perf_counter() - wall_started
    summary: Dict[str, object] = summarize(metrics, ttft_slo_ms, tpot_slo_ms)
    summary["wall_seconds"] = wall_seconds
    summary["request_throughput"] = requests / wall_seconds if wall_seconds else 0.0
    summary["url"] = url
    summary["model"] = model
    summary["input_tokens_target"] = input_tokens
    summary["output_tokens_target"] = output_tokens
    summary["concurrency"] = concurrency
    summary["shared_prefix"] = shared_prefix
    result = {"summary": summary, "requests": [asdict(item) for item in metrics]}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
