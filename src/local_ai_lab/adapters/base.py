from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from local_ai_lab.config import ModelConfig, Workload
from local_ai_lab.evidence import Evidence, UNKNOWN
from local_ai_lab.metrics import TimingMetrics


@dataclass(slots=True)
class ProbeResult:
    available: bool
    status: str
    version: str | None = None
    details: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(slots=True)
class PathDescription:
    id: str
    name: str
    abstraction_level: str
    stack: list[str]
    responsibilities: dict[str, str]
    you_control: list[str]
    runtime_controls: list[str]
    windows_controls: list[str]
    model_format: str
    actual_device: str = UNKNOWN
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GenerationResult:
    text: str
    timing: TimingMetrics
    evidence: list[Evidence]
    usage: dict[str, Any] = field(default_factory=dict)
    server_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


class AdapterUnavailableError(RuntimeError):
    pass


class InferenceAdapter(ABC):
    id: str

    def __init__(self, model: ModelConfig | None = None) -> None:
        self.model = model

    @abstractmethod
    def probe(self) -> ProbeResult:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> PathDescription:
        raise NotImplementedError

    def load(self) -> list[Evidence]:
        return []

    @abstractmethod
    def generate(self, workload: Workload) -> GenerationResult:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "InferenceAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
