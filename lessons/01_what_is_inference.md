# 01 - What is inference?

Inference is the process of using a trained model to produce an output from an
input. Training changes the model's parameters; inference uses those parameters.

For an LLM, an application builds a prompt, a tokenizer converts text to token
IDs, model execution computes logits, and a sampler repeatedly selects output
tokens. Other models can have different inputs and outputs. For example, an
image classifier accepts an image tensor and returns class scores without a text
token loop.

## Goals

By the end of this lesson, you should be able to:

1. Run the lab's environment and hardware diagnostics.
2. distinguish a physical device from software that can use it;
3. distinguish an installed execution provider (EP) from proof that a model ran
   on that EP; and
4. label diagnostic claims as **observed**, **configured**, or **unknown**.

This lesson does not run a model yet. Its goal is to establish an accurate
baseline before later lessons make execution claims.

## Before you begin

Open PowerShell in the repository root. If you already created and installed the
lab's virtual environment, activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

You normally activate the environment once per new terminal session. You do not
need to reinstall the project each time.

For first-time setup only, create the environment and install the lab:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Confirm that the active Python belongs to the virtual environment:

```powershell
python -c "import sys; print(sys.executable)"
```

The printed path should end with `.venv\Scripts\python.exe`.

## Step 1: Run `doctor`

```powershell
python -m local_ai_lab doctor
```

`doctor` reports two different kinds of information:

- Windows-reported OS and hardware presence; and
- readiness of each optional inference path in the active Python environment.

Your exact device names and readiness results will differ. A shortened example
looks like this:

```text
OS: Microsoft Windows 11 Pro build 26100
Architecture: AMD64
CPU: Example Processor
GPU present: Example Graphics Adapter (OK)
NPU present: Example Neural Processor (OK)
Presence does not prove an inference workload used that device.

UNAVAILABLE windows-ai-api: ...
UNAVAILABLE foundry-local: ...
UNAVAILABLE windows-ml: ...
READY onnx-runtime: ONNX Runtime is importable.
  Available EPs: CPUExecutionProvider
  [runtime-reported] available_execution_providers:
  CPUExecutionProvider (onnxruntime)
UNAVAILABLE ort-genai: ...
UNAVAILABLE llama-cpp: ...
UNAVAILABLE cloud: ...
```

Interpret this example as follows:

- `GPU present` and `NPU present` are **observed hardware presence** reported by
  Windows.
- `READY onnx-runtime` means the ONNX Runtime Python package is importable in
  this environment.
- `Available EPs: CPUExecutionProvider` means this ONNX Runtime installation
  currently exposes only its CPU execution provider.
- The GPU and NPU can exist even when ORT exposes only the CPU EP. Accelerators
  require compatible drivers and provider-specific software.
- `READY` does not mean that a model has run.
- `UNAVAILABLE` is not a failure for this lesson. Most runtimes are optional,
  and different runtime distributions may require separate virtual
  environments.

Do not install every optional runtime just to make every line say `READY`.

## Step 2: Record the hardware snapshot

Run:

```powershell
python -m local_ai_lab hardware --output results\hardware.json
```

The command prints JSON and saves the same snapshot to
`results\hardware.json`. A shortened example is:

```json
{
  "architecture": "AMD64",
  "cpu": {
    "name": "Example Processor"
  },
  "devices": [
    {
      "kind": "GPU",
      "name": "Example Graphics Adapter",
      "status": "OK"
    },
    {
      "kind": "NPU",
      "name": "Example Neural Processor",
      "status": "OK"
    }
  ]
}
```

This output is evidence that the operating system reported those devices. It
does not identify which device an inference runtime selected.

If your output has no GPU or NPU entry, record that absence. Do not invent an
accelerator or treat its absence as a lesson failure.

## Step 3: Classify the evidence

Use these definitions:

| Label | Meaning | Example |
|---|---|---|
| **Observed** | Reported by the OS, runtime, or a measured run | Windows reports an NPU; ORT reports `CPUExecutionProvider` |
| **Configured** | Requested by a setting, command, or model configuration | A configuration requests `QNNExecutionProvider` |
| **Unknown** | Not demonstrated by current evidence | Which device executed model operators |

Create a small table for your machine. For example:

| Claim | Classification | Evidence |
|---|---|---|
| Windows reports a GPU | Observed | `GPU` entry in `hardware` output |
| Windows reports an NPU | Observed, or unknown if absent | `NPU` entry in `hardware` output |
| ONNX Runtime is installed | Observed | `READY onnx-runtime` |
| ORT can see the CPU EP | Observed | `Available EPs: CPUExecutionProvider` |
| ORT can use the NPU | Unknown | NPU presence alone is insufficient |
| A model ran on the GPU or NPU | Unknown | No model was run in this lesson |

Be precise: a device can be **present** while an EP is unavailable, and an EP
can be **available** while a model still runs partly or entirely on the CPU.

## Troubleshooting

### `No module named local_ai_lab`

First verify that the previously installed virtual environment is active:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

If the environment has never been installed, complete the first-time setup
above. Use the same Python environment for installation and for lab commands.

### ORT shows only `CPUExecutionProvider`

This is a valid result. The standard ONNX Runtime package may expose only the
CPU EP. Seeing a GPU or NPU in the hardware snapshot does not automatically add
an ORT provider for it. Lessons 05, 09, and 10 cover EPs, Windows ML, and direct
ORT in detail.

### Optional runtimes say `UNAVAILABLE`

This is expected unless you installed their extras and completed any external
configuration they require. Record the result; do not treat it as a failure of
this exercise.

## Completion check

You are ready for Lesson 2 when all of the following are true:

- `python -m local_ai_lab doctor` completes and you can identify at least one
  runtime as `READY` or explain why all are `UNAVAILABLE`;
- `results\hardware.json` exists and contains your OS, CPU, architecture, and
  any Windows-reported accelerator devices;
- you can explain why `NPU present` does not mean that inference used the NPU;
- you can explain why `CPUExecutionProvider` can be the only ORT EP even when a
  GPU and NPU are present; and
- your evidence table labels the actual execution device as **unknown** because
  this lesson did not run and measure a model.

Expected conclusion:

> I have recorded the hardware and runtime capabilities reported by my current
> environment. I have not yet proved that inference ran on any device.

Keep `results\hardware.json`; later lessons can compare runtime execution
evidence with this baseline.
