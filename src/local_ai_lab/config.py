from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration must be a mapping: {path}")
    return data


def resolve_config(kind: str, name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate.resolve()
    name = name_or_path if name_or_path.endswith((".yaml", ".yml")) else f"{name_or_path}.yaml"
    candidate = ROOT / "configs" / kind / name
    if not candidate.exists():
        raise ConfigurationError(f"Unknown {kind.rstrip('s')} '{name_or_path}'")
    return candidate


@dataclass(slots=True)
class GenerationConfig:
    max_output_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 42
    context_length: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "GenerationConfig":
        data = data or {}
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass(slots=True)
class Workload:
    id: str
    name: str
    task: str
    prompt: str
    system_prompt: str | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    checks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, name_or_path: str) -> "Workload":
        data = load_yaml(resolve_config("workloads", name_or_path))
        required = ("id", "name", "task", "prompt")
        missing = [key for key in required if key not in data]
        if missing:
            raise ConfigurationError(f"Workload is missing: {', '.join(missing)}")
        data["generation"] = GenerationConfig.from_mapping(data.get("generation"))
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass(slots=True)
class ModelConfig:
    id: str
    format: str
    path: str | None = None
    family: str | None = None
    quantization: str | None = None
    tokenizer: str | None = None
    provider: str | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str) -> "ModelConfig":
        data = load_yaml(resolve_config("models", name_or_path))
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})
