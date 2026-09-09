from __future__ import annotations

from typing import Any


def explain_differences(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, str]]:
    explanations: list[dict[str, str]] = []

    def add(category: str, observation: str, possible: str) -> None:
        explanations.append({"category": category, "observation": observation, "possible_explanation": possible})

    lm, rm = left.get("model", {}), right.get("model", {})
    for key, label in (("id", "model"), ("format", "model format"), ("quantization", "quantization")):
        if lm.get(key) != rm.get(key):
            add("CONFIGURED", f"Different {label}: {lm.get(key)} vs {rm.get(key)}", f"The {label} difference may affect compute, memory, and output quality.")
    le = {item["claim"]: item["value"] for item in left.get("evidence", []) if item.get("kind") == "runtime-reported"}
    re = {item["claim"]: item["value"] for item in right.get("evidence", []) if item.get("kind") == "runtime-reported"}
    for key in sorted(set(le) | set(re)):
        if le.get(key) != re.get(key):
            add("RUNTIME-REPORTED", f"{key}: {le.get(key)} vs {re.get(key)}", "Different providers or devices can change operator coverage, transfer overhead, and throughput.")
    if left.get("adapter") == "cloud" or right.get("adapter") == "cloud":
        add("OBSERVED", "At least one path uses cloud transport.", "Client-observed cloud latency includes network, queueing, and service overhead.")
    lt = left.get("timing", {}).get("end_to_end_seconds")
    rt = right.get("timing", {}).get("end_to_end_seconds")
    if lt is not None and rt is not None:
        add("OBSERVED", f"End-to-end latency: {lt:.3f}s vs {rt:.3f}s", "The measurements differ; use the configured and runtime-reported facts above before attributing a cause.")
    if not explanations:
        add("POSSIBLE EXPLANATION", "No differentiating metadata was captured.", "Collect provider profiles, model hashes, token counts, and cold/warm state before drawing conclusions.")
    return explanations
