from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from local_ai_lab.education.compare import explain_differences
from local_ai_lab.metrics import summarize


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_markdown_report(records: list[dict[str, Any]], path: Path) -> Path:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["adapter"], record.get("model", {}).get("id", "unknown"), record["workload"])].append(record)
    comparison_kinds = sorted({record.get("comparison_kind", "unspecified") for record in records})
    lines = [
        "# Inference comparison report",
        "",
        f"**Comparison kind:** {', '.join(comparison_kinds)}",
        "",
        "> Raw samples are authoritative. Cloud client timings include transport. Same-task is not apples-to-apples.",
        "",
        "| Runtime | Model | Workload | Samples | E2E median (s) | TTFT median (s) | Correct |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    representatives: list[dict[str, Any]] = []
    for (adapter, model, workload), items in groups.items():
        representatives.append(items[-1])
        e2e = summarize([item["timing"]["end_to_end_seconds"] for item in items if item["timing"]["end_to_end_seconds"] is not None])
        ttft = summarize([item["timing"]["ttft_seconds"] for item in items if item["timing"]["ttft_seconds"] is not None])
        correct = sum(all(check["passed"] for check in item["correctness"]) for item in items)
        lines.append(f"| {adapter} | {model} | {workload} | {len(items)} | {e2e.get('median', 'n/a')} | {ttft.get('median', 'n/a')} | {correct}/{len(items)} |")
    if len(representatives) >= 2:
        lines.extend(["", "## Why are these numbers different?", ""])
        for item in explain_differences(representatives[0], representatives[1]):
            lines.append(f"- **{item['category']}**: {item['observation']}  ")
            lines.append(f"  Possible explanation: {item['possible_explanation']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
