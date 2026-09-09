from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil

from local_ai_lab.adapters.registry import create_adapter
from local_ai_lab.config import ModelConfig, ROOT, Workload, load_yaml, resolve_config
from local_ai_lab.correctness import evaluate
from local_ai_lab.hardware import discover_hardware
from local_ai_lab.metrics import ResourceMetrics
from local_ai_lab.telemetry import base_metadata, sha256_file, sha256_text, write_jsonl


def run_experiment(
    adapter_id: str,
    workload: Workload,
    model: ModelConfig | None,
    samples: int,
    warmups: int,
    output: Path,
    comparison_kind: str = "single-path",
) -> list[dict[str, Any]]:
    hardware = discover_hardware()
    adapter = create_adapter(adapter_id, model)
    probe = adapter.probe()
    if not probe.available:
        raise RuntimeError(f"{adapter_id} is unavailable: {probe.status}")
    process = psutil.Process()
    records: list[dict[str, Any]] = []
    try:
        load_started = time.perf_counter()
        load_evidence = adapter.load()
        load_seconds = time.perf_counter() - load_started
        for _ in range(warmups):
            adapter.generate(workload)
        for index in range(samples):
            rss_before = process.memory_info().rss
            process.cpu_percent(None)
            result = adapter.generate(workload)
            resources = ResourceMetrics(
                rss_before_bytes=rss_before,
                rss_after_bytes=process.memory_info().rss,
                peak_rss_bytes=process.memory_info().peak_wset if hasattr(process.memory_info(), "peak_wset") else None,
                cpu_percent=process.cpu_percent(None),
            )
            if result.timing.model_load_seconds is None:
                result.timing.model_load_seconds = load_seconds
            if result.timing.end_to_end_seconds is not None:
                if warmups or index:
                    result.timing.warm_start_seconds = result.timing.end_to_end_seconds
                else:
                    result.timing.cold_start_seconds = load_seconds + result.timing.end_to_end_seconds
            record = {
                "schema_version": 1,
                "run_id": str(uuid.uuid4()),
                **base_metadata(ROOT, hardware),
                "adapter": adapter_id,
                "comparison_kind": comparison_kind,
                "adapter_description": asdict(adapter.describe()),
                "model": asdict(model) if model else {},
                "model_hash": sha256_file(Path(model.path).expanduser()) if model and model.path else None,
                "workload": workload.id,
                "prompt_hash": sha256_text(workload.prompt),
                "generation": asdict(workload.generation),
                "sample_index": index,
                "warmup_iterations": warmups,
                "cold_or_warm": "warm" if warmups or index else "cold",
                "text": result.text,
                "timing": result.timing.to_dict(),
                "resources": resources.to_dict(),
                "usage": result.usage,
                "server_metrics": result.server_metrics,
                "evidence": [item.to_dict() for item in [*load_evidence, *result.evidence]],
                "warnings": result.warnings,
                "correctness": [item.to_dict() for item in evaluate(result.text, workload.checks)],
            }
            write_jsonl(output, record)
            records.append(record)
    finally:
        adapter.close()
    return records


def run_benchmark(config_path: str) -> tuple[Path, list[dict[str, Any]]]:
    data = load_yaml(resolve_config("benchmarks", config_path))
    output = ROOT / str(data.get("output", "results/benchmark.jsonl"))
    all_records: list[dict[str, Any]] = []
    for experiment in data.get("experiments", []):
        model = ModelConfig.load(experiment["model"]) if experiment.get("model") else None
        workload = Workload.load(experiment["workload"])
        records = run_experiment(
            experiment["runtime"],
            workload,
            model,
            int(experiment.get("samples", data.get("samples", 3))),
            int(experiment.get("warmups", data.get("warmups", 1))),
            output,
            str(data.get("comparison_kind", "same-task")),
        )
        all_records.extend(records)
    return output, all_records
