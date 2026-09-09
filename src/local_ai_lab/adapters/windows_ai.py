from __future__ import annotations

from local_ai_lab.adapters.base import AdapterUnavailableError, InferenceAdapter, PathDescription, ProbeResult
from local_ai_lab.config import Workload
from local_ai_lab.evidence import UNKNOWN


class WindowsAIAdapter(InferenceAdapter):
    id = "windows-ai-api"

    def probe(self) -> ProbeResult:
        return ProbeResult(
            False,
            "No official Python surface exists. Use the C#/C++ WinRT companion approach in lesson 07 and gate each API on GetReadyState().",
        )

    def describe(self) -> PathDescription:
        return PathDescription(
            id=self.id,
            name="Windows AI task-specific APIs",
            abstraction_level="High-level task API",
            stack=["Application", "Windows AI task-specific WinRT API", "Windows-managed implementation", UNKNOWN, "Windows-selected backend", "NPU where the specific API contract guarantees it", "Windows-managed system model"],
            responsibilities={
                "model owner": "Windows/Microsoft",
                "model distribution": "Windows",
                "runtime": UNKNOWN,
                "inference engine": UNKNOWN,
                "backend": "Windows-selected; application cannot choose",
                "hardware": "API-specific and readiness-gated",
                "hardware selection": "Windows",
                "tokenization": UNKNOWN,
                "KV cache": UNKNOWN,
                "sampling": "API-specific",
                "prefill/decode": UNKNOWN,
                "offline": "yes after API/model readiness",
                "application ships": "application; not the system model",
                "Windows responsibility": "model lifecycle, API implementation and supported hardware path",
                "application responsibility": "readiness, task input, output handling",
            },
            you_control=["task inputs", "limited API-specific options"],
            runtime_controls=["model lifecycle and hidden inference details"],
            windows_controls=["model distribution, compatibility and hardware selection"],
            model_format=UNKNOWN,
            notes=["This adapter intentionally does not invent Python bindings."],
        )

    def generate(self, workload: Workload):
        raise AdapterUnavailableError("Windows AI APIs require the documented C#/C++ WinRT bridge; no official Python API is available.")
