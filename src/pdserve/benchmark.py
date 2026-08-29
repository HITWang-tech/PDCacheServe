from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

from .simulator import generate_trace, simulate

POLICIES = ("aggregated", "pd-round-robin", "pd-load-aware", "pd-kv-aware")
WORKLOADS = {
    "short": {"arrival_rate_rps": 3.0, "shared_prefix_ratio": 0.15},
    "long": {"arrival_rate_rps": 1.5, "shared_prefix_ratio": 0.60},
    "mixed": {"arrival_rate_rps": 1.8, "shared_prefix_ratio": 0.35},
}


def run_benchmark(
    output_dir: str,
    requests_per_seed: int = 300,
    seeds: Sequence[int] = tuple(range(10)),
) -> Dict[str, object]:
    raw: List[Dict[str, object]] = []
    for workload, config in WORKLOADS.items():
        for seed in seeds:
            trace = generate_trace(
                workload,
                requests_per_seed,
                seed,
                float(config["arrival_rate_rps"]),
                float(config["shared_prefix_ratio"]),
            )
            for policy in POLICIES:
                raw.append({"workload": workload, "seed": seed, **simulate(trace, policy)})

    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for item in raw:
        grouped[(item["workload"], item["policy"])].append(item)
    aggregate = []
    metric_names = (
        "slo_attainment",
        "goodput_rps",
        "throughput_rps",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p95_tpot_ms",
        "p95_e2e_ms",
        "cache_hit_token_ratio",
        "average_transfer_ms",
        "average_gpu_utilization",
    )
    for (workload, policy), items in sorted(grouped.items()):
        aggregate.append(
            {
                "workload": workload,
                "policy": policy,
                "seed_count": len(items),
                "requests": len(items) * requests_per_seed,
                **{
                    name: round(mean(float(item[name]) for item in items), 4)
                    for name in metric_names
                },
            }
        )
    report: Dict[str, object] = {
        "kind": "discrete_event_simulation",
        "seeds": list(seeds),
        "requests_per_seed": requests_per_seed,
        "total_policy_runs": len(raw),
        "total_simulated_requests": len(raw) * requests_per_seed,
        "gpu_models": [
            "RTX-4090-24GB",
            "RTX-3090-24GB",
            "RTX-4090-24GB",
            "RTX-3090-24GB",
        ],
        "results": aggregate,
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in raw),
        encoding="utf-8",
    )
    (target / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: Dict[str, object]) -> str:
    indexed = {
        (item["workload"], item["policy"]): item
        for item in report["results"]  # type: ignore[index]
    }
    lines = [
        "# PDCacheServe 离散事件仿真报告",
        "",
        "> 本报告是控制面策略仿真，不是生产 GPU 实测。真实双卡结果需要单独执行。",
        "",
        f"- 随机种子：{len(report['seeds'])}",
        f"- 每个 workload/seed 请求数：{report['requests_per_seed']}",
        f"- 策略运行数：{report['total_policy_runs']}",
        f"- 累计策略请求：{report['total_simulated_requests']}",
        "",
        "| Workload | Policy | SLO attainment | Goodput req/s | P95 TTFT ms | "
        "P95 TPOT ms | Cache hit tokens | GPU util |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["results"]:  # type: ignore[index]
        lines.append(
            "| {workload} | {policy} | {slo_attainment:.2%} | {goodput_rps:.3f} | "
            "{p95_ttft_ms:.1f} | {p95_tpot_ms:.1f} | {cache_hit_token_ratio:.2%} | "
            "{average_gpu_utilization:.2%} |".format(**item)
        )
    lines.extend(["", "## 关键结论", ""])
    for workload in ("short", "mixed", "long"):
        baseline = indexed[(workload, "pd-round-robin")]
        optimized = indexed[(workload, "pd-kv-aware")]
        ttft_reduction = 1.0 - float(optimized["p95_ttft_ms"]) / float(
            baseline["p95_ttft_ms"]
        )
        goodput_gain = float(optimized["goodput_rps"]) / max(
            float(baseline["goodput_rps"]), 1e-9
        ) - 1.0
        slo_gain = float(optimized["slo_attainment"]) - float(baseline["slo_attainment"])
        lines.append(
            f"- `{workload}`：KV-aware 相比 PD Round Robin 的 P95 TTFT 降低 "
            f"**{ttft_reduction:.2%}**，Goodput 提升 **{goodput_gain:.2%}**，"
            f"SLO 达标率提高 **{slo_gain:.2%}**。"
        )
    lines.extend(
        [
            "- 当前 PCIe 异构画像下，Aggregated 仍取得更高的绝对 Goodput 和更低 TTFT，",
            "  说明 PD 分离并非无条件收益；真实部署必须同时考虑 KV 传输成本、SLO 类型与负载。",
        ]
    )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "仿真使用相同四卡异构资源和相同请求轨迹比较 Aggregated、PD Round Robin、",
            "PD Load Aware 和 PD KV Aware。延迟模型参数来自配置中的设备画像，不代表特定",
            "线上集群。简历中不得把这些结果描述为真实 GPU 集群实测。",
            "",
        ]
    )
    return "\n".join(lines)
