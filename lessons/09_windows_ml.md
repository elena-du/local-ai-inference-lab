# 09 - Windows ML

Current Windows ML supports Python through Windows App SDK projections plus `onnxruntime-windowsml`. It adds a Windows-supported ORT distribution and execution-provider catalog/acquisition. The application still owns the ONNX model and ORT session.

Install the matching Windows App SDK Runtime and documented packages. In Python, ensure each provider is ready and register its library explicitly before session creation; `RegisterCertifiedAsync()` alone does not register it into Python ORT. Then inspect `ort.get_ep_devices()` and profiles.

**Exercise:** run the same ONNX model through Windows ML's distribution and direct ORT CPU, recording exact versions and providers.
