# 08 - Foundry Local

Foundry Local combines model discovery, acquisition, caching, loading, and inference. Its Python surface calls Foundry Local Core, which uses ONNX Runtime and Windows ML execution-provider registration. This is not the same layer as ORT itself.

**Exercise:** install the Windows ML SDK variant in an isolated environment, list catalog variants, and record the selected variant/provider. Treat X2 NPU use as unknown until logs/status prove it.
