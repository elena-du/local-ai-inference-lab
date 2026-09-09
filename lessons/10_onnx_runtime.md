# 10 - ONNX Runtime

Direct ORT exposes `InferenceSession`, graph optimization, ordered providers, provider options, profiling, and tensor I/O.

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.enable_profiling = True
session = ort.InferenceSession("model.onnx", sess_options=options,
                               providers=["CPUExecutionProvider"])
print(session.get_providers())
outputs = session.run(None, {"input": input_tensor})
print(session.end_profiling())
```

Generic tensor execution does not supply tokenization, sampling, or a generation loop. **Exercise:** inspect the profile and map nodes to providers.
