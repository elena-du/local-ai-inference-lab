from __future__ import annotations

from local_ai_lab.adapters.base import InferenceAdapter
from local_ai_lab.adapters.cloud import CloudAdapter
from local_ai_lab.adapters.foundry_local import FoundryLocalAdapter
from local_ai_lab.adapters.llama_cpp import LlamaCppAdapter
from local_ai_lab.adapters.onnx_runtime import OnnxRuntimeAdapter
from local_ai_lab.adapters.ort_genai import OrtGenAIAdapter
from local_ai_lab.adapters.windows_ai import WindowsAIAdapter
from local_ai_lab.adapters.windows_ml import WindowsMLAdapter
from local_ai_lab.config import ModelConfig


ADAPTERS: dict[str, type[InferenceAdapter]] = {
    cls.id: cls
    for cls in (
        WindowsAIAdapter,
        FoundryLocalAdapter,
        WindowsMLAdapter,
        OnnxRuntimeAdapter,
        OrtGenAIAdapter,
        LlamaCppAdapter,
        CloudAdapter,
    )
}


def create_adapter(adapter_id: str, model: ModelConfig | None = None) -> InferenceAdapter:
    try:
        adapter = ADAPTERS[adapter_id]
    except KeyError as exc:
        raise ValueError(f"Unknown runtime '{adapter_id}'. Choose from: {', '.join(ADAPTERS)}") from exc
    return adapter(model)
