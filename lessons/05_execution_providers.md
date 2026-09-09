# 05 - Execution providers

An ONNX Runtime execution provider translates supported graph regions to a backend. ORT can partition unsupported regions to another provider. Therefore, `QNNExecutionProvider` in an availability list is weaker evidence than a successful run with CPU fallback disabled plus a per-operator profile.

**Exercise:** compare the CPU and QNN model templates. Explain why static QDQ quantization and `session.disable_cpu_ep_fallback` matter during qualification.
