from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path

from local_ai_lab.adapters.base import AdapterUnavailableError, GenerationResult, InferenceAdapter, PathDescription, ProbeResult
from local_ai_lab.config import Workload
from local_ai_lab.evidence import Evidence, EvidenceKind, UNKNOWN
from local_ai_lab.metrics import TimingMetrics


class OrtGenAIAdapter(InferenceAdapter):
    id = "ort-genai"

    def probe(self) -> ProbeResult:
        try:
            import onnxruntime_genai  # noqa: F401
        except ImportError:
            return ProbeResult(False, "Install the preview 'genai' extra: pip install -e .[genai]")
        return ProbeResult(True, "ONNX Runtime GenAI is importable (preview).", importlib.metadata.version("onnxruntime-genai"))

    def describe(self) -> PathDescription:
        return PathDescription(
            id=self.id,
            name="ONNX Runtime GenAI",
            abstraction_level="Generative model orchestration API over runtime",
            stack=["Application", "ORT GenAI Python API", "ORT GenAI + ONNX Runtime", "ORT GenAI generator / ORT graph executor", "Configured in genai_config.json", UNKNOWN, "ONNX model directory"],
            responsibilities={
                "model owner": "application/developer",
                "model distribution": "application/developer",
                "runtime": "ONNX Runtime",
                "inference engine": "ORT GenAI generator plus ORT graph executor",
                "backend": "configured in genai_config.json",
                "hardware": UNKNOWN,
                "hardware selection": "model configuration and installed providers",
                "tokenization": "ORT GenAI tokenizer assets",
                "KV cache": "ORT GenAI",
                "sampling": "ORT GenAI",
                "prefill/decode": "ORT GenAI generation loop and model graphs",
                "offline": "yes",
                "application ships": "model directory, tokenizer, genai_config.json, runtime packages",
                "Windows responsibility": "drivers and scheduling",
                "application responsibility": "model assets, prompt and generation parameters",
            },
            you_control=["model directory", "provider configuration", "prompt", "search options"],
            runtime_controls=["tokenization", "KV cache", "generation loop", "graph execution"],
            windows_controls=["drivers, scheduling, memory"],
            model_format="ONNX graph(s), genai_config.json, tokenizer assets",
            notes=["ORT GenAI is preview. A supported Python+QNN recipe must be independently qualified."],
        )

    def load(self) -> list[Evidence]:
        if not self.model or not self.model.path:
            raise AdapterUnavailableError("ORT GenAI requires a model directory in model config.")
        import onnxruntime_genai as og

        path = Path(self.model.path).expanduser()
        if not path.exists():
            raise AdapterUnavailableError(f"Model directory not found: {path}")
        started = time.perf_counter()
        self._og = og
        self._model = og.Model(str(path))
        self._tokenizer = og.Tokenizer(self._model)
        self._load_seconds = time.perf_counter() - started
        return [Evidence("model_directory", str(path), EvidenceKind.CONFIGURED, "model config")]

    def generate(self, workload: Workload) -> GenerationResult:
        if not hasattr(self, "_model"):
            self.load()
        prompt = f"{workload.system_prompt}\n{workload.prompt}" if workload.system_prompt else workload.prompt
        input_tokens = self._tokenizer.encode(prompt)
        params = self._og.GeneratorParams(self._model)
        options = {
            "max_length": len(input_tokens) + workload.generation.max_output_tokens,
            "temperature": workload.generation.temperature,
            "top_p": workload.generation.top_p,
        }
        if workload.generation.seed is not None:
            options["random_seed"] = workload.generation.seed
        params.set_search_options(**options)
        generator = self._og.Generator(self._model, params)
        generator.append_tokens(input_tokens)
        output: list[int] = []
        started = time.perf_counter()
        first_at: float | None = None
        while not generator.is_done():
            generator.generate_next_token()
            if first_at is None:
                first_at = time.perf_counter()
            output.append(generator.get_next_tokens()[0])
        finished = time.perf_counter()
        decode = finished - first_at if first_at else None
        return GenerationResult(
            text=self._tokenizer.decode(output),
            timing=TimingMetrics(
                model_load_seconds=self._load_seconds,
                ttft_seconds=first_at - started if first_at else None,
                decode_seconds=decode,
                end_to_end_seconds=finished - started,
                prompt_tokens=len(input_tokens),
                output_tokens=len(output),
                decode_tokens_per_second=len(output) / decode if output and decode else None,
                total_tokens_per_second=(len(input_tokens) + len(output)) / (finished - started),
            ),
            evidence=[Evidence("generation_orchestrator", "ONNX Runtime GenAI", EvidenceKind.RUNTIME_REPORTED, "onnxruntime_genai")],
        )
