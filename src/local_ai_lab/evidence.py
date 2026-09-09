from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


UNKNOWN = "unknown / runtime-managed"


class EvidenceKind(StrEnum):
    OBSERVED = "observed"
    CONFIGURED = "configured"
    RUNTIME_REPORTED = "runtime-reported"
    ASSUMPTION = "assumption"


@dataclass(slots=True)
class Evidence:
    claim: str
    value: str
    kind: EvidenceKind
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
