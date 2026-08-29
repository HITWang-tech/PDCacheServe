import math

from pdserve.gpu_benchmark import RequestMetric, percentile, summarize


def metric(request_id, ttft, tpot, success=True):
    return RequestMetric(
        request_id=request_id,
        ttft_ms=ttft,
        tpot_ms=tpot,
        e2e_ms=ttft + tpot * 9,
        prompt_tokens=100,
        completion_tokens=10,
        success=success,
    )


def test_percentile_interpolates_and_handles_empty_input():
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([10, 20], 0.95) == 19.5
    assert math.isnan(percentile([], 0.95))


def test_summary_counts_failures_and_slo_attainment():
    result = summarize(
        [metric(0, 100, 10), metric(1, 300, 10), metric(2, 0, 0, False)],
        ttft_slo_ms=200,
        tpot_slo_ms=20,
    )
    assert result["successful"] == 2
    assert result["slo_attainment"] == 1 / 3
