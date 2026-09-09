# 02 - API vs runtime vs engine

An API is what code calls. A runtime loads and schedules model operations. An inference engine performs execution and optimization. A high-level API may hide all lower layers; a runtime API such as ORT exposes provider/session choices.

**Exercise:** compare `explain windows-ai-api`, `explain foundry-local`, and `explain onnx-runtime`. List controls that disappear as abstraction rises.
