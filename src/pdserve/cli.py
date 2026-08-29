from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .benchmark import run_benchmark
from .cache import KVDirectory
from .control import ControlPlane
from .models import CacheTier, InferenceRequest, ModelLayout, Worker, WorkerRole
from .registry import WorkerRegistry
from .router import SLORouter


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pdserve")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--config", default="configs/demo.json")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--output-dir", default="artifacts/benchmark")
    benchmark.add_argument("--requests", type=int, default=300)
    benchmark.add_argument("--seeds", type=int, default=10)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8200)
    return root


def _demo(config_path: str) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    registry = WorkerRegistry(heartbeat_timeout_seconds=30)
    cache = KVDirectory()
    plane = ControlPlane(registry, cache, SLORouter(registry, cache))
    for item in config["workers"]:
        layout = ModelLayout(**item.pop("layout"))
        plane.register_worker(Worker(layout=layout, role=WorkerRole(item.pop("role")), **item))
    layout = ModelLayout(**config["layout"])
    cache.register("shared-system-prompt", layout, 2048, CacheTier.GPU, "decode-0")
    request = InferenceRequest(
        request_id="demo-request",
        model_id=layout.model_id,
        input_tokens=4096,
        max_new_tokens=256,
        prefix_hash="shared-system-prompt",
        prefix_tokens=2048,
        ttft_slo_ms=2000,
        tpot_slo_ms=60,
    )
    return {"decision": plane.route(request).as_dict(), "metrics": plane.metrics()}


def main(argv: Optional[List[str]] = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "demo":
        result = _demo(args.config)
    elif args.command == "benchmark":
        result = run_benchmark(args.output_dir, args.requests, tuple(range(args.seeds)))
    else:
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("install pdcacheserve[api] before serving") from exc
        uvicorn.run("pdserve.api:app", host=args.host, port=args.port)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
