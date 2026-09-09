from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class CheckResult:
    check: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(text: str, checks: list[dict[str, Any]]) -> list[CheckResult]:
    results = [CheckResult("non_empty", bool(text.strip()), "Response contains text" if text.strip() else "Empty response")]
    for check in checks:
        kind = check.get("type")
        if kind == "json":
            try:
                value = json.loads(text)
                expected = check.get("required_keys", [])
                missing = [key for key in expected if not isinstance(value, dict) or key not in value]
                results.append(CheckResult("json_schema", not missing, f"Missing keys: {missing}" if missing else "Valid JSON and required keys"))
            except json.JSONDecodeError as exc:
                results.append(CheckResult("json_schema", False, f"Invalid JSON: {exc.msg}"))
        elif kind == "contains":
            value = str(check.get("value", ""))
            results.append(CheckResult("contains", value.casefold() in text.casefold(), f"Expected '{value}'"))
        elif kind == "max_length":
            limit = int(check["value"])
            results.append(CheckResult("max_length", len(text) <= limit, f"{len(text)} <= {limit}"))
    return results
