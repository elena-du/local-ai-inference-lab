from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from typing import Any

from local_ai_lab.adapters.base import AdapterUnavailableError, GenerationResult, InferenceAdapter, PathDescription, ProbeResult
from local_ai_lab.config import Workload
from local_ai_lab.evidence import Evidence, EvidenceKind, UNKNOWN
from local_ai_lab.metrics import TimingMetrics


class OnnxRuntimeAdapter(InferenceAdapter):
    id = "onnx-runtime"

    def probe(self) -> ProbeResult:
        try:
            import onnxruntime as ort
        except ImportError:
            return ProbeResult(False, "Install the 'onnx' extra: pip install -e .[onnx]")
        return ProbeResult(
            True,
            "ONNX Runtime is importable.",
            getattr(ort, "__version__", None),
            details=[f"Available EPs: {', '.join(ort.get_available_providers())}"],
            evidence=[Evidence("available_execution_providers", ", ".join(ort.get_available_providers()), EvidenceKind.RUNTIME_REPORTED, "onnxruntime")],
        )

    def describe(self) -> PathDescription:
        provider = self.model.provider if self.model and self.model.provider else "CPUExecutionProvider (default unless configured)"
        return PathDescription(
            id=self.id,
            name="ONNX Runtime direct",
            abstraction_level="Runtime API",
            stack=["Application", "ONNX Runtime Python API", "ONNX Runtime", "ORT graph executor", provider, UNKNOWN, "ONNX"],
            responsibilities={
                "model owner": "application/developer",
                "model distribution": "application/developer",
                "runtime": "ONNX Runtime",
                "inference engine": "ORT graph executor",
                "backend": provider,
                "hardware": UNKNOWN,
                "hardware selection": "application provider order plus ORT graph partitioning",
                "tokenization": "application/model-specific",
                "KV cache": "model/application for generative graphs",
                "sampling": "application/model-specific",
                "prefill/decode": "model/application-specific",
                "offline": "yes",
                "application ships": "ONNX assets, ORT and provider dependencies",
                "Windows responsibility": "drivers and OS resource management",
                "application responsibility": "model, tensors, provider selection, preprocessing/postprocessing",
            },
            you_control=["ONNX model", "session options", "provider order/options", "input tensors"],
            runtime_controls=["graph optimization", "operator placement and execution"],
            windows_controls=["drivers, scheduling, memory"],
            model_format="ONNX",
            notes=["Generic ONNX execution is tensor inference, not an LLM generation loop."],
        )

    def load(self) -> list[Evidence]:
        if not self.model or not self.model.path:
            raise AdapterUnavailableError("Direct ONNX Runtime requires a model config with path.")
        import onnxruntime as ort

        path = Path(self.model.path).expanduser()
        if not path.exists():
            raise AdapterUnavailableError(f"ONNX model not found: {path}")
        providers: list[Any] = [self.model.provider] if self.model.provider else ["CPUExecutionProvider"]
        provider_options = {
            key: value
            for key, value in self.model.provider_options.items()
            if key != "disable_cpu_fallback"
        }
        options = [provider_options] if provider_options else None
        session_options = ort.SessionOptions()
        session_options.enable_profiling = True
        if self.model.provider_options.get("disable_cpu_fallback"):
            session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        started = time.perf_counter()
        self.session = ort.InferenceSession(str(path), sess_options=session_options, providers=providers, provider_options=options)
        self.load_seconds = time.perf_counter() - started
        actual = self.session.get_providers()
        if self.model.provider and self.model.provider not in actual:
            raise AdapterUnavailableError(f"Requested {self.model.provider}, session reports {actual}; refusing silent fallback.")
        return [Evidence("session_execution_providers", ", ".join(actual), EvidenceKind.RUNTIME_REPORTED, "InferenceSession.get_providers")]

    def run_tensors(self, inputs: dict[str, Any]) -> tuple[list[Any], TimingMetrics]:
        if not hasattr(self, "session"):
            self.load()
        started = time.perf_counter()
        outputs = self.session.run(None, inputs)
        elapsed = time.perf_counter() - started
        return outputs, TimingMetrics(model_load_seconds=self.load_seconds, end_to_end_seconds=elapsed)

    def generate(self, workload: Workload) -> GenerationResult:
        raise AdapterUnavailableError(
            "Direct ONNX Runtime accepts tensors, not prompts. Use an ORT GenAI model or the Python example in lesson 10."
        )

    def close(self) -> None:
        if hasattr(self, "session"):
            self.profile_path = self.session.end_profiling()
