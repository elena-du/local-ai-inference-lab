from local_ai_lab.metrics import TimingMetrics, summarize


def test_summary_preserves_outlier_visibility() -> None:
    result = summarize([1.0, 1.0, 1.0, 1.0, 100.0])
    assert result["median"] == 1.0
    assert result["maximum"] == 100.0
    assert result["p95"] is not None


def test_small_sample_omits_tail_percentiles() -> None:
    result = summarize([1.0, 2.0])
    assert result["p90"] is None
    assert result["p95"] is None


def test_timing_keeps_phases_separate() -> None:
    timing = TimingMetrics(model_load_seconds=1.0, cold_start_seconds=1.5, prefill_seconds=0.2, decode_seconds=0.3)
    assert timing.to_dict()["cold_start_seconds"] == 1.5
    assert timing.prefill_seconds != timing.decode_seconds
