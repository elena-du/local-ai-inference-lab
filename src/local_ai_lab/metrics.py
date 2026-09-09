from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TimingMetrics:
    model_load_seconds: float | None = None
    cold_start_seconds: float | None = None
    warm_start_seconds: float | None = None
    ttft_seconds: float | None = None
    prefill_seconds: float | None = None
    decode_seconds: float | None = None
    streaming_seconds: float | None = None
    end_to_end_seconds: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    total_tokens_per_second: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResourceMetrics:
    rss_before_bytes: int | None = None
    rss_after_bytes: int | None = None
    peak_rss_bytes: int | None = None
    cpu_percent: float | None = None
    gpu_percent: float | None = None
    npu_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentage
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {}
    return {
        "median": statistics.median(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90) if len(values) >= 3 else None,
        "p95": percentile(values, 0.95) if len(values) >= 5 else None,
        "standard_deviation": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }
