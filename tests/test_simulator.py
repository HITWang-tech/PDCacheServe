from pdserve.benchmark import run_benchmark
from pdserve.simulator import generate_trace, simulate


def test_trace_generation_is_reproducible():
    left = generate_trace("mixed", 20, 7, 1.0, 0.5)
    right = generate_trace("mixed", 20, 7, 1.0, 0.5)
    assert left == right


def test_kv_aware_policy_observes_shared_prefix_hits():
    trace = generate_trace("long", 100, 3, 0.3, 0.9)
    report = simulate(trace, "pd-kv-aware")
    assert report["cache_hit_token_ratio"] > 0.25


def test_every_policy_returns_all_requests():
    trace = generate_trace("short", 25, 1, 0.5, 0.2)
    for policy in ("aggregated", "pd-round-robin", "pd-load-aware", "pd-kv-aware"):
        assert simulate(trace, policy)["request_count"] == 25


def test_benchmark_writes_reproducible_artifacts(tmp_path):
    report = run_benchmark(str(tmp_path), requests_per_seed=20, seeds=(0, 1))
    assert report["total_policy_runs"] == 24
    assert report["total_simulated_requests"] == 480
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert len((tmp_path / "raw.jsonl").read_text().splitlines()) == 24
