from __future__ import annotations

import os

from local_ai_lab.adapters.base import PathDescription
from local_ai_lab.adapters.http_chat import OpenAICompatibleAdapter
from local_ai_lab.evidence import UNKNOWN


class LlamaCppAdapter(OpenAICompatibleAdapter):
    id = "llama-cpp"
    base_url_env = "LOCAL_AI_LAB_LLAMA_BASE_URL"
    api_key_env = "LOCAL_AI_LAB_LLAMA_API_KEY"
    default_url = "http://127.0.0.1:8080/v1"

    def describe(self) -> PathDescription:
        backend = os.getenv("LOCAL_AI_LAB_LLAMA_BACKEND", UNKNOWN)
        return PathDescription(
            id=self.id,
            name="llama.cpp server",
            abstraction_level="Local HTTP API over native inference engine",
            stack=["Application", "OpenAI-compatible llama-server API", "llama.cpp / GGML", "llama.cpp model executor", backend, "CPU/GPU/NPU only when runtime-reported", "GGUF"],
            responsibilities={
                "model owner": "application/developer",
                "model distribution": "application/developer",
                "runtime": "llama.cpp / GGML",
                "inference engine": "llama.cpp",
                "backend": backend,
                "hardware": UNKNOWN,
                "hardware selection": "server startup flags and backend availability",
                "tokenization": "llama.cpp tokenizer from GGUF",
                "KV cache": "llama.cpp",
                "sampling": "llama.cpp",
                "prefill/decode": "llama.cpp",
                "offline": "yes, after model acquisition",
                "application ships": "server/native libraries and GGUF model",
                "Windows responsibility": "drivers, scheduling, memory, networking",
                "application responsibility": "server lifecycle, model, prompts, parameters",
            },
            you_control=["GGUF model", "quantization", "context size", "offload flags", "sampling"],
            runtime_controls=["tokenization", "KV cache", "prompt evaluation", "decode"],
            windows_controls=["drivers, process and memory scheduling"],
            model_format="GGUF",
            notes=["Backend environment metadata is configured evidence until confirmed by llama.cpp startup logs."],
        )
