# Local AI Inference Lab

**"Calling a model" can mean very different things.**

```text
APPLICATION
    ↓
DEVELOPER API / SDK
    ↓
FRAMEWORK / RUNTIME
    ↓
INFERENCE ENGINE / MODEL EXECUTOR
    ↓
EXECUTION PROVIDER / HARDWARE BACKEND
    ↓
CPU | GPU | NPU
    ↓
MODEL REPRESENTATION / COMPILED ARTIFACT
```

This repository is an interactive course and evidence-first benchmark harness for Windows local inference and cloud inference. Real systems may collapse layers, but an API, a runtime, an execution provider, a model file, and an NPU are not interchangeable concepts.

## Start here

Install a native Python 3.11-3.13 build. Windows ML does not support Microsoft Store Python. On ARM64, use an ARM64 Python distribution.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

python -m local_ai_lab doctor
python -m local_ai_lab hardware --output results\hardware.json
python -m local_ai_lab runtimes
python -m local_ai_lab explain windows-ml
```

`doctor` reports missing optional dependencies. A detected GPU/NPU is only OS evidence that a device exists, never proof that inference used it.

## Technology map

| Technology | What it is | Level / developer API | Runtime or engine | Typical format | Backend / EP | CPU / GPU / NPU | Hardware selection | Model ownership | Offline | Platform fit |
|---|---|---|---|---|---|---|---|---|---|---|
| Windows AI APIs | Task-specific Windows APIs | High; C#/C++ WinRT | Usually hidden | Windows-managed | Windows-managed | API-specific; supported Copilot+ APIs commonly NPU | Windows | Windows | After readiness | Windows-only; turnkey tasks |
| Foundry Local | Local catalog, lifecycle, SDK/API | High-mid; Python SDK/REST | Core over ONNX Runtime | Optimized ONNX | Windows ML-registered EP | Depends on catalog variant and verified EP | Runtime/catalog | Developer selects, service manages cache | After acquisition | Windows-first managed local models |
| Windows ML | Windows ONNX platform integration | Mid-low; WinRT EP catalog + ORT Python | Windows-supported ORT | ONNX | CPU, DirectML, acquired vendor EPs | Depends on registered EP and graph | Developer + Windows ML | Developer | Yes | Windows BYO ONNX with servicing/EP acquisition |
| ONNX Runtime | Cross-platform model runtime | Low; `InferenceSession` | ORT graph executor | ONNX | Ordered EPs | Provider-dependent | Developer order + graph partitioner | Developer | Yes | Maximum ONNX control/portability |
| ORT GenAI | Generative orchestration above ORT | Mid; generator/tokenizer API | ORT GenAI + ORT | ONNX directory/config/tokenizer | Model-configured | Package/provider-dependent | Configuration/runtime | Developer | Yes | ONNX LLM generation; preview |
| llama.cpp | Portable native LLM engine | Mid-low; CLI/server/C API | llama.cpp/GGML | GGUF | CPU, GPU backends; experimental Hexagon | Build/flags/backend-dependent | Developer flags + backend | Developer | Yes | Portable quantized local LLMs |
| Cloud inference | Remote managed service | High; HTTP/SDK | Provider-managed | Usually undisclosed | Provider-managed | Undisclosed unless provider reports it | Provider | Provider | No | Minimal local setup and elastic capacity |

Use `python -m local_ai_lab explain <path>` for the full responsibility map.

## Runnable paths

### llama.cpp server

Build/download `llama-server`, obtain a GGUF model separately, inspect its startup logs, then:

```powershell
llama-server.exe -m C:\models\model.gguf --host 127.0.0.1 --port 8080
$env:LOCAL_AI_LAB_LLAMA_BASE_URL = "http://127.0.0.1:8080/v1"
python -m local_ai_lab run --runtime llama-cpp --model llama_example --workload chat
```

Edit the model config ID and quantization first. Record backend/device and offloaded-layer logs; the HTTP response alone does not prove GPU/NPU use.

### Cloud

Copy `.env.example` values into environment variables (the CLI does not auto-load secrets):

```powershell
$env:LOCAL_AI_LAB_CLOUD_BASE_URL = "https://provider.example/v1"
$env:LOCAL_AI_LAB_CLOUD_API_KEY = "<secret>"
python -m local_ai_lab run --runtime cloud --model cloud_example --workload chat
```

Cloud timings are client-observed and include transport, queueing, and service work unless separately reported.

### ONNX Runtime and Windows ML

Direct ORT is intentionally a tensor API, not a fake prompt API. Install `.[onnx]`, configure an ONNX path/provider, then use `OnnxRuntimeAdapter.run_tensors()` with model-specific preprocessing. Windows ML uses the same ORT programming model while adding Windows runtime servicing and execution-provider discovery/acquisition. See lessons 09 and 10.

For QNN qualification, use a static-shape quantized QDQ model and disable CPU fallback. Provider availability alone does not prove complete NPU offload.

### ORT GenAI and Foundry Local

ORT GenAI is a preview optional extra. Foundry Local is pre-release and may expose a native SDK or an in-process OpenAI-compatible endpoint. `doctor` explains which surface is available. Do not install conflicting Foundry SDK variants in one environment.

## Benchmarking

```powershell
python -m local_ai_lab benchmark --config configs\benchmarks\default.yaml
python -m local_ai_lab compare llama-cpp cloud --input results\benchmark.jsonl
```

Raw JSONL is written before aggregation. The Markdown report keeps load, TTFT, decode, and end-to-end separate and displays correctness checks.

- **Same-model:** match family/revision, documented quantization, prompt, context, tokenizer behavior, and generation settings.
- **Same-task:** useful when artifacts differ, but explicitly not a pure runtime comparison.
- Keep cold and warm samples distinct.
- Never interpret different model outputs as a universal quality score.

## Repository tour

- `architecture.md`: design, dependency plan, methodology, and reference-machine support matrix.
- `configs/`: independent device expectations, models, workloads, cloud, and benchmark plans.
- `src/local_ai_lab/adapters/`: common interface and optional implementations.
- `lessons/`: guided course with experiments.
- `results/`: raw evidence and generated reports; generated files are ignored.

## Glossary

- **API:** callable contract exposed to a developer.
- **SDK:** tools, libraries, samples, and APIs for building against a platform.
- **Framework:** higher-level structure coordinating application/model behavior.
- **Runtime:** software that loads and executes model operations.
- **Inference engine:** executor implementing model evaluation and often optimizations.
- **Execution provider (EP):** ORT plugin that maps supported graph operators to a backend.
- **Hardware backend:** implementation targeting CPU, GPU, NPU, or another compute API.
- **Operator:** model operation such as matrix multiplication or attention.
- **Graph:** connected operators and tensors describing computation.
- **Tensor:** typed multidimensional array.
- **Model format:** serialized representation and associated metadata/assets.
- **ONNX:** portable computational graph/model representation.
- **GGUF:** llama.cpp ecosystem format containing model tensors and metadata.
- **Quantization:** representing values with lower precision to reduce size/work.
- **Tokenizer:** converts text to/from token IDs.
- **Prefill:** prompt processing that populates attention state.
- **Decode:** iterative output-token generation.
- **KV cache:** cached attention keys/values reused during decode.
- **Context window:** maximum token history available to a model.
- **Sampling:** selecting output tokens from model scores.
- **CPU:** general-purpose processor.
- **GPU:** parallel graphics/compute processor.
- **NPU:** neural-processing accelerator.
- **TOPS:** theoretical trillions of operations per second; not application throughput.
- **TTFT:** client- or runtime-observed time to first output token.
- **tokens/sec:** token throughput whose meaning depends on tokenizer and phase.

## Evidence rules

1. Configured is not observed.
2. Device presence is not workload placement.
3. Provider availability is not full graph offload.
4. A fallback is a result, not a detail to hide.
5. Unknown is better than invented certainty.

Current source links and lifecycle caveats are maintained in `architecture.md`.
