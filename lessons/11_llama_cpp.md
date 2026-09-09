# 11 - llama.cpp

llama.cpp is a native LLM engine built on GGML and consuming GGUF. It owns tokenization, prompt evaluation, KV cache, sampling, and decode. Context size, quantization, threads, batch sizes, and layer offload affect results.

On Snapdragon Windows ARM64, OpenCL is the practical GPU experiment; Hexagon support is experimental and unsuitable for the normal MVP. **Exercise:** start `llama-server`, preserve startup/backend logs, run the chat workload, and reconcile server timings with client timings.
