# Architecture

This lab uses one vocabulary for every inference path:

```text
APPLICATION
    |
DEVELOPER API / SDK
    |
FRAMEWORK / RUNTIME
    |
INFERENCE ENGINE / MODEL EXECUTOR
    |
EXECUTION PROVIDER / HARDWARE BACKEND
    |
CPU | GPU | NPU
    |
MODEL REPRESENTATION / COMPILED ARTIFACT
```

Implementations sometimes collapse layers. The lab does not rename unlike components to make adapters appear symmetrical. Each adapter reports what is known, the source of that knowledge, and `unknown / runtime-managed` when the implementation does not expose a responsibility.

## Proposed repository tree

```text
local-ai-inference-lab/
|-- architecture.md
|-- README.md
|-- pyproject.toml
|-- .env.example
|-- configs/
|   |-- devices/
|   |-- models/
|   |-- workloads/
|   |-- cloud/
|   `-- benchmarks/
|-- lessons/
|-- src/local_ai_lab/
|   |-- adapters/
|   |-- benchmarking/
|   `-- education/
|-- tests/
`-- results/
```

Windows-only discovery and adapters stay isolated. Configuration, measurement, result persistence, comparison, and correctness checks remain portable.

## Adapter interface

`InferenceAdapter` owns lifecycle orchestration, not implementation details:

- `probe()` reports availability, versions, and missing requirements without throwing for an absent optional dependency.
- `describe()` returns the conceptual stack and a responsibility map.
- `load()` loads or connects to a model and returns load evidence.
- `generate()` returns text, token counts/timestamps when observable, and runtime evidence.
- `close()` releases local resources.

Adapters must not silently switch execution providers or devices. A requested and actual path mismatch is a prominent result warning. An adapter may report a configured provider as configured evidence, but only runtime output can be labeled runtime-reported.

## Dependency plan

Core dependencies are intentionally small: Typer, Rich, PyYAML, psutil, and httpx. Pytest is a development dependency.

Optional extras are isolated:

| Extra | Purpose | Notes |
|---|---|---|
| `onnx` | Direct ONNX Runtime | Windows ARM64 wheels exist; provider packages remain environment-specific. |
| `winml` | Current Windows ML | `onnxruntime-windowsml` and Windows App SDK projections; Windows only. |
| `genai` | ORT GenAI | Preview API; provider-specific composition must be qualified. |
| `foundry` | Foundry Local | Pre-release Windows ML SDK; keep separate from generic SDK variants. |

llama.cpp and cloud use an OpenAI-compatible HTTP contract and need no native Python package. llama.cpp server binaries and GGUF models are user-supplied.

## Benchmark methodology

1. Discover the real system and probe the selected adapter.
2. Print the conceptual path, controls, model format, claimed device, and evidence.
3. Warm up as configured.
4. Execute each measured sample independently and write its raw record immediately to JSONL.
5. Measure process RSS and CPU at the client; label unavailable accelerator utilization as unavailable rather than zero.
6. Preserve model-load, prefill, decode, and end-to-end metrics separately when observable.
7. Aggregate median, p50, p90, p95 (only with enough samples), standard deviation, and individual samples.
8. Run lightweight correctness checks beside performance metrics.
9. Generate Markdown from persisted results.

Same-model comparisons require matching model identity/family, quantization, workload input, context, and generation parameters. Same-task comparisons are explicitly labeled non-apples-to-apples. Cloud timings are client-observed and include transport unless the server supplies separately labeled timing.

## Snapdragon X2 Elite support matrix

This matrix describes the inspected reference machine (Windows 11 Enterprise build 28000, ARM64, Snapdragon X2 Elite, Adreno X2-85, Hexagon NPU present). Presence is not proof of use.

| Path | Python surface | Expected status on this machine | Evidence required before acceleration claim |
|---|---|---|---|
| Windows AI APIs | No official Python API; C#/C++ WinRT bridge | Scaffold only; individual API readiness is authoritative | API readiness plus API contract; low-level engine generally undisclosed |
| Foundry Local | Native Python SDK | Callable after optional install; X2 provider qualification not publicly confirmed | SDK-selected model variant, EP registration/status, runtime logs |
| Windows ML | Python projection plus `onnxruntime-windowsml` | Callable with supported Python 3.10-3.13 and Windows App SDK runtime | ready EP registration, `ort.get_ep_devices()`, session profiling |
| ONNX Runtime CPU | `InferenceSession` | Supported by ARM64 wheel | session provider list and profile |
| ONNX Runtime QNN | `InferenceSession` plus QNN plugin | Experimental qualification for X2; never assumed | CPU fallback disabled, successful run, provider and per-op profile |
| ORT GenAI | Native Python preview | ARM64 wheel available; documented Python+QNN recipe unresolved | generated runtime logs/profile and provider configuration |
| llama.cpp CPU | CLI/server | Native ARM64 build supported | server startup logs and token timings |
| llama.cpp OpenCL GPU | CLI/server | X2 family documented; local X2-85 build still must be tested | device/backend logs, layer offload, buffer allocation |
| llama.cpp Hexagon NPU | Experimental backend | Unsupported for normal MVP deployment | test-signed experimental setup and per-op profiling |
| Cloud | OpenAI-compatible HTTP | Supported with endpoint and credential | client timing plus server usage/timing fields when supplied |

## Official references

- Windows AI APIs: <https://learn.microsoft.com/windows/ai/apis/>
- Windows ML overview: <https://learn.microsoft.com/windows/ai/new-windows-ml/overview>
- Windows ML execution providers: <https://learn.microsoft.com/windows/ai/new-windows-ml/supported-execution-providers>
- Foundry Local architecture: <https://learn.microsoft.com/azure/foundry-local/concepts/foundry-local-architecture>
- Foundry Local SDK: <https://learn.microsoft.com/azure/foundry-local/reference/reference-sdk-current>
- ONNX Runtime execution providers: <https://onnxruntime.ai/docs/execution-providers/>
- QNN execution provider: <https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html>
- ONNX Runtime GenAI: <https://onnxruntime.ai/docs/genai/>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- llama.cpp OpenCL: <https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md>

Documentation and package availability were checked on 2026-09-09. Preview APIs can change; `doctor` is the source of truth for the cloned environment.
