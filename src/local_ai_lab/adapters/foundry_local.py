from __future__ import annotations

import importlib.util
import os

from local_ai_lab.adapters.base import PathDescription, ProbeResult
from local_ai_lab.adapters.http_chat import OpenAICompatibleAdapter
from local_ai_lab.evidence import UNKNOWN


class FoundryLocalAdapter(OpenAICompatibleAdapter):
    id = "foundry-local"
    base_url_env = "LOCAL_AI_LAB_FOUNDRY_BASE_URL"
    api_key_env = "LOCAL_AI_LAB_FOUNDRY_API_KEY"

    def probe(self) -> ProbeResult:
        sdk = importlib.util.find_spec("foundry_local_sdk") is not None
        endpoint = super().probe()
        endpoint.details.insert(0, f"Native Python SDK installed: {sdk}")
        if not sdk and not endpoint.available:
            endpoint.status = (
                "Install foundry-local-sdk-winml for native lifecycle management, or set "
                "LOCAL_AI_LAB_FOUNDRY_BASE_URL to an active Foundry Local OpenAI-compatible endpoint."
            )
        return endpoint

    def describe(self) -> PathDescription:
        return PathDescription(
            id=self.id,
            name="Foundry Local",
            abstraction_level="Local model catalog/lifecycle SDK and inference API",
            stack=["Application", "Foundry Local Python SDK or OpenAI-compatible endpoint", "Foundry Local Core", "ONNX Runtime", "Windows ML-registered EP", UNKNOWN, "Catalog-selected optimized ONNX artifacts"],
            responsibilities={
                "model owner": "developer selects; Foundry Local manages local artifact",
                "model distribution": "Foundry Local catalog/cache",
                "runtime": "ONNX Runtime under Foundry Local Core",
                "inference engine": "Foundry Local Core and ONNX Runtime",
                "backend": "discovered/registered through Windows ML",
                "hardware": UNKNOWN,
                "hardware selection": "Foundry Local catalog variant and EP discovery",
                "tokenization": "Foundry Local runtime/model assets",
                "KV cache": "Foundry Local runtime",
                "sampling": "Foundry Local runtime",
                "prefill/decode": "Foundry Local runtime",
                "offline": "yes after required models/providers are cached",
                "application ships": "SDK/application; model delivery can be catalog-managed",
                "Windows responsibility": "Windows ML EP acquisition/registration, drivers, OS resources",
                "application responsibility": "model selection, lifecycle calls, prompts and validation",
            },
            you_control=["catalog alias/model", "prompt", "generation parameters", "cache lifecycle"],
            runtime_controls=["artifact selection/compilation", "tokenization", "KV cache", "generation"],
            windows_controls=["EP acquisition/registration, drivers, scheduling"],
            model_format="Optimized/compiled ONNX variants",
            notes=["The SDK is pre-release. X2 NPU support must be demonstrated by selected variant and EP evidence."],
        )
