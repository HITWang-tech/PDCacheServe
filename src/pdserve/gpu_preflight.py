"""Fail-fast hardware validation for real prefill/decode deployments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_devices(device_ids: list[int]) -> dict:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - GPU deployment dependency
        raise RuntimeError("PyTorch is required for GPU preflight") from exc

    if not torch.cuda.is_available():
        return {"eligible": False, "reason": "CUDA is unavailable", "devices": []}
    count = torch.cuda.device_count()
    if any(device < 0 or device >= count for device in device_ids):
        return {
            "eligible": False,
            "reason": f"requested devices {device_ids}, but only {count} are visible",
            "devices": [],
        }
    devices = [
        {
            "id": device,
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
        for device in device_ids
    ]
    peer_matrix = {
        f"{source}->{target}": bool(torch.cuda.can_device_access_peer(source, target))
        for source in device_ids
        for target in device_ids
        if source != target
    }
    eligible = bool(peer_matrix) and all(peer_matrix.values())
    reason = (
        "bidirectional CUDA peer access is available"
        if eligible
        else "bidirectional CUDA peer access is unavailable; direct KV transfer may be incorrect"
    )
    return {
        "eligible": eligible,
        "reason": reason,
        "devices": devices,
        "peer_access": peer_matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--output")
    parser.add_argument("--allow-no-p2p", action="store_true")
    args = parser.parse_args()
    device_ids = [int(value) for value in args.devices.split(",")]
    result = inspect_devices(device_ids)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    if not result["eligible"] and not args.allow_no_p2p:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
