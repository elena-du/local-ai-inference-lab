# 02 - API vs runtime vs engine

An inference application contains several layers. These layers are easy to
confuse because one product may provide more than one of them.

- An **API** is the interface your application calls.
- A **runtime** loads a model, manages resources, and schedules model
  operations.
- An **inference engine** executes and optimizes the model's computations.
- A **backend or execution provider (EP)** connects the engine to a hardware
  path such as a CPU, GPU, or NPU.
- The **hardware** performs the selected computations.

A high-level API can hide the runtime, engine, model, and hardware selection. A
lower-level runtime API exposes more of those choices to the application
developer.

## Goals

By the end of this lesson, you should be able to:

1. explain the difference between an API, runtime, engine, backend, and hardware;
2. read the layer stack printed by the lab's `explain` command;
3. identify which decisions belong to you, the runtime, or Windows;
4. compare high-, middle-, and low-level inference paths; and
5. explain the trade-off between convenience and control.

This lesson describes three inference paths, but it does not run a model.
`READY` or `UNAVAILABLE` reports installation readiness only, not proof of model
execution.

## Before you begin

Open PowerShell in the repository root and activate the environment you used in
Lesson 1:

```powershell
.\.venv\Scripts\Activate.ps1
```

Confirm that the lab is available:

```powershell
python -m local_ai_lab --help
```

If Python reports `No module named local_ai_lab`, verify the active Python:

```powershell
python -c "import sys; print(sys.executable)"
```

The path should end with `.venv\Scripts\python.exe`. Return to the first-time
setup in Lesson 1 only if this environment has not been installed.

## One concrete example

Suppose an application must identify the object in a photo:

1. The **application** receives the photo and needs an answer such as `cat`.
2. The application calls an **API**, for example:

   ```python
   outputs = session.run(None, {"image": image_tensor})
   ```

   The API defines how the application submits input and receives output.
3. The **runtime** loads the model, checks the input, manages memory, and
   schedules the model's operations.
4. The **inference engine** optimizes and executes operations such as
   convolution and matrix multiplication. A product may combine the runtime and
   engine, so they are not always separate programs.
5. A **backend or execution provider (EP)** implements supported operations for
   a particular compute path. Examples include an ORT CPU EP or a vendor NPU EP.
6. The **hardware** performs the work assigned to it. This could be the CPU,
   GPU, NPU, or a combination if some operations fall back to another provider.
7. The runtime returns output scores through the API, and the application
   converts the highest score into the label `cat`.

The **model** is the file containing the learned parameters and computation
graph used during these steps. It is loaded by the runtime; it is not the API,
runtime, engine, EP, or hardware.

| Layer | Its job in this example | It does not prove |
|---|---|---|
| Application | Prepares the image and displays `cat` | Which device executed the model |
| API | Defines how code sends the tensor and receives scores | That the requested provider was used |
| Runtime | Loads and coordinates the model | That every operation used one device |
| Engine | Optimizes and executes model operations | That an accelerator handled every operation |
| Backend / EP | Maps supported operations to a compute path | That no operation fell back |
| Hardware | Physically performs assigned computations | That the whole model ran there |
| Model | Defines learned parameters and operations | Which runtime or device will execute it |

## How to read `explain`

The command format is:

```powershell
python -m local_ai_lab explain <inference-path>
```

Each result has five important sections:

1. **PATH** - layers from your application down to the model.
2. **YOU CONTROL** - choices exposed directly to your code.
3. **RUNTIME CONTROLS** - decisions delegated to runtime software.
4. **WINDOWS CONTROLS** - decisions delegated to the operating system.
5. **EVIDENCE / READINESS** - whether required software or configuration is
   currently available.

The responsibility table gives more detail about ownership. The line
`ACTUAL EXECUTION DEVICE: unknown / runtime-managed` is intentional: describing
an architecture does not prove which device executed a real model.

The path also names the model artifact or representation. This is not meant to
suggest that a model is physically "after" the hardware. The runtime loads the
model, and the diagram includes it so you can see who supplies it and in what
format.

## Step 1: Inspect a high-level API

Run:

```powershell
python -m local_ai_lab explain windows-ai-api
```

A shortened example is:

```text
PATH: Windows AI task-specific APIs
Application
  | v
Windows AI task-specific WinRT API
  | v
Windows-managed implementation
  | v
unknown / runtime-managed
  | v
Windows-selected backend
  | v
NPU where the specific API contract guarantees it
  | v
Windows-managed system model

YOU CONTROL
- task inputs
- limited API-specific options

WINDOWS CONTROLS
- model distribution, compatibility and hardware selection

ACTUAL EXECUTION DEVICE: unknown / runtime-managed
```

Interpretation:

- Your application asks for a task through a Windows API.
- Windows manages most implementation details, including the system model and
  supported hardware path.
- Your code gains simplicity but cannot freely choose the model format, runtime,
  or backend.
- An NPU claim is valid only where a specific API contract and readiness
  evidence guarantee it.

The readiness section may say `UNAVAILABLE` because this lab is running through
Python and the Windows AI APIs use a C# or C++ WinRT path. That is an expected
result and does not prevent you from completing this lesson.

Record:

| Question | Windows AI API answer |
|---|---|
| Who chooses the model? | Windows |
| Who chooses the hardware path? | Windows |
| Can the application select an ORT EP? | No |
| How much implementation detail is visible? | Little |

## Step 2: Inspect a managed local runtime

Run:

```powershell
python -m local_ai_lab explain foundry-local
```

A shortened example is:

```text
PATH: Foundry Local
Application
  | v
Foundry Local Python SDK or OpenAI-compatible endpoint
  | v
Foundry Local Core
  | v
ONNX Runtime
  | v
Windows ML-registered EP
  | v
unknown / runtime-managed
  | v
Catalog-selected optimized ONNX artifacts

YOU CONTROL
- catalog alias/model
- prompt
- generation parameters
- cache lifecycle

RUNTIME CONTROLS
- artifact selection/compilation
- tokenization
- KV cache
- generation

ACTUAL EXECUTION DEVICE: unknown / runtime-managed
```

Interpretation:

- Your application selects a catalog model and supplies prompts and generation
  settings.
- Foundry Local manages model artifacts and language-model details such as
  tokenization and generation.
- ONNX Runtime is still part of the stack, but your application does not create
  and configure a raw ORT session.
- Windows ML participates in EP discovery and registration.
- The actual hardware remains unknown until runtime evidence demonstrates it.

The readiness section may say `UNAVAILABLE` if the Foundry Local SDK or endpoint
is not configured. Do not install it merely to complete this comparison.

Record:

| Question | Foundry Local answer |
|---|---|
| Who selects the catalog model? | The developer |
| Who manages the local artifact? | Foundry Local |
| Who handles tokenization and generation? | Foundry Local runtime |
| Can the application directly place every operator? | No |

## Step 3: Inspect a low-level runtime API

Run:

```powershell
python -m local_ai_lab explain onnx-runtime
```

A shortened example is:

```text
PATH: ONNX Runtime direct
Application
  | v
ONNX Runtime Python API
  | v
ONNX Runtime
  | v
ORT graph executor
  | v
CPUExecutionProvider (default unless configured)
  | v
unknown / runtime-managed
  | v
ONNX

YOU CONTROL
- ONNX model
- session options
- provider order/options
- input tensors

RUNTIME CONTROLS
- graph optimization
- operator placement and execution

ACTUAL EXECUTION DEVICE: unknown / runtime-managed
```

Interpretation:

- Your application supplies the ONNX model and correctly shaped input tensors.
- Your application can request and order available EPs.
- ORT optimizes the graph and assigns supported operations to providers.
- Requesting an EP is configuration, not proof that all operations ran on its
  hardware.
- Direct ORT accepts tensors. It does not automatically provide an LLM prompt,
  tokenizer, sampling loop, or chat interface.

The readiness section can list available EPs:

```text
READY onnx-runtime: ONNX Runtime is importable.
  Available EPs: CPUExecutionProvider
```

Your list may differ. This line reports providers exposed by the installed ORT
distribution; it does not report a measured model run.

Record:

| Question | Direct ONNX Runtime answer |
|---|---|
| Who provides the model? | The developer/application |
| Who prepares input tensors? | The developer/application |
| Who requests the provider order? | The developer/application |
| Who performs graph optimization and operator placement? | ONNX Runtime |

## Step 4: Compare the paths

Complete this table using your command output:

| Decision | Windows AI API | Foundry Local | Direct ORT |
|---|---|---|---|
| Task input or prompt | Developer | Developer | Developer prepares tensors |
| Model selection | Windows-managed | Developer selects from catalog | Developer supplies ONNX |
| Model distribution | Windows | Foundry Local catalog/cache | Developer |
| Tokenization | Hidden/API-specific | Foundry Local | Developer/model-specific |
| Provider selection | Windows | Runtime/catalog/Windows ML | Developer requests provider order |
| Operator placement | Hidden | Runtime-managed | ORT-managed |
| Actual device proven by `explain` | No | No | No |
| Relative abstraction | High | Middle | Low |

Now answer these questions:

1. Which path gives the developer the fewest controls?
2. Which path makes the developer responsible for the model and input tensors?
3. Which path lets the developer request an EP order?
4. What happens to developer responsibility as abstraction becomes lower?

Expected answers:

1. Windows AI task-specific APIs.
2. Direct ONNX Runtime.
3. Direct ONNX Runtime.
4. Developer control increases, but setup, validation, and integration
   responsibility also increase.

## Common mistakes

### "`UNAVAILABLE` means the architecture description is invalid"

No. The path and responsibility sections describe the integration model. The
readiness section separately reports whether your current environment can use
that path.

### "The API is the runtime"

Not necessarily. An API is the surface your code calls. It may delegate to one
or several runtime and engine layers.

### "ONNX is the engine"

No. ONNX is a model representation. ONNX Runtime is runtime software that can
load and execute ONNX models.

### "I selected an EP, so the model ran entirely on its device"

No. Provider selection is configured intent. Runtime session reports, profiling,
and fallback checks are needed to establish what executed where.

### "A higher-level API is always better"

No. It is often easier, but it exposes fewer controls. The appropriate level
depends on whether the application values convenience, portability, detailed
provider control, custom preprocessing, or another requirement.

## Completion check

You are ready for Lesson 3 when all of the following are true:

- all three `explain` commands complete, even if one or more readiness sections
  say `UNAVAILABLE`;
- your comparison table identifies who owns model selection, tokenization,
  provider selection, and operator placement for each path;
- you can describe the stack in this order:
  **application -> API -> runtime/engine -> backend or EP -> hardware**, and
  explain that the runtime loads a model artifact;
- you can explain that higher abstraction usually provides less control and
  less integration work;
- you can explain that lower abstraction usually provides more control and more
  developer responsibility; and
- you label the actual execution device as **unknown** because `explain` did not
  run or profile a model.

Expected conclusion:

> An API is what my application calls, a runtime coordinates model execution,
> an engine performs and optimizes the computations, and a backend or EP connects
> execution to supported hardware. Different inference paths expose or hide
> these layers, so convenience and developer control vary.
