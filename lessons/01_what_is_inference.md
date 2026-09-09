# 01 - What is inference?

Inference transforms an input into outputs by evaluating a trained model. For an LLM, the application builds a prompt, a tokenizer produces token IDs, model execution computes logits, and a sampler selects tokens repeatedly. For classification, the input/output may simply be tensors with no token loop.

**Exercise:** run `doctor`, then `hardware`. Mark every line as observed, configured, or unknown. Do not call the detected NPU an execution device yet.
