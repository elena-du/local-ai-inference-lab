# 03 - Model formats

ONNX describes a graph of operators and tensors consumed by ONNX Runtime and other tools. GGUF packages tensors and metadata for the llama.cpp ecosystem. Neither format is hardware. Provider compilation may produce an additional device-specific artifact.

**Exercise:** inspect the model configs. Explain why equivalent model families in GGUF and ONNX are not automatically byte-identical or performance-identical.
