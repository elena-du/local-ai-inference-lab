from __future__ import annotations

import importlib.metadata

from local_ai_lab.adapters.base import PathDescription, ProbeResult
from local_ai_lab.adapters.onnx_runtime import OnnxRuntimeAdapter
from local_ai_lab.evidence import UNKNOWN


class WindowsMLAdapter(OnnxRuntimeAdapter):
    id = "windows-ml"

    def probe(self) -> ProbeResult:
        try:
            version = importlib.metadata.version("onnxruntime-windowsml")
        except importlib.metadata.PackageNotFoundError:
            return ProbeResult(
                False,
                "Install the supported Windows ML Python packages and matching Windows App SDK Runtime; see lesson 09.",
            )
        result = super().probe()
        result.version = version
        result.details.append("EP readiness/registration must be performed through Windows ML before session creation.")
        return result

    def describe(self) -> PathDescription:
        provider = self.model.provider if self.model and self.model.provider else "Windows ML selected/registered EP"
        return PathDescription(
            id=self.id,
            name="Windows ML (current)",
            abstraction_level="Windows platform layer plus ONNX Runtime API",
            stack=["Application", "Windows ML EP catalog + ONNX Runtime Python API", "onnxruntime-windowsml", "ORT graph executor", provider, UNKNOWN, "ONNX"],
            responsibilities={
                "model owner": "application/developer",
                "model distribution": "application; Windows ML may acquire provider packages",
                "runtime": "Windows-supported ONNX Runtime distribution",
                "inference engine": "ORT graph executor",
                "backend": provider,
                "hardware": UNKNOWN,
                "hardware selection": "application and Windows ML EP catalog/registration",
                "tokenization": "application or generative orchestration layer",
                "KV cache": "model/orchestration layer",
                "sampling": "application/orchestration layer",
                "prefill/decode": "model/orchestration layer",
                "offline": "yes after dependencies are acquired",
                "application ships": "ONNX model and framework-dependent app dependencies",
                "Windows responsibility": "supported ORT distribution, EP catalog/acquisition, DirectML, drivers",
                "application responsibility": "model, EP readiness and registration, tensors, validation",
            },
            you_control=["BYO ONNX model", "EP acquisition/registration", "session options", "inputs"],
            runtime_controls=["graph optimization, partitioning and execution"],
            windows_controls=["runtime servicing, EP catalog/acquisition, drivers"],
            model_format="ONNX",
            notes=["Unlike direct ORT, Windows ML adds Windows-managed runtime servicing and EP discovery/acquisition."],
        )
