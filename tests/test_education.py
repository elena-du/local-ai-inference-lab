from local_ai_lab.adapters.registry import ADAPTERS, create_adapter
from local_ai_lab.education.compare import explain_differences


def test_every_adapter_has_complete_responsibility_map() -> None:
    required = {"model owner", "runtime", "backend", "hardware", "tokenization", "KV cache", "sampling", "prefill/decode", "offline"}
    for adapter_id in ADAPTERS:
        description = create_adapter(adapter_id).describe()
        assert required <= description.responsibilities.keys()
        assert len(description.stack) >= 6


def test_comparison_separates_configured_and_observed() -> None:
    left = {"adapter": "llama-cpp", "model": {"id": "a", "format": "GGUF"}, "timing": {"end_to_end_seconds": 1.0}, "evidence": []}
    right = {"adapter": "cloud", "model": {"id": "b", "format": "provider"}, "timing": {"end_to_end_seconds": 2.0}, "evidence": []}
    results = explain_differences(left, right)
    assert any(item["category"] == "CONFIGURED" for item in results)
    assert any(item["category"] == "OBSERVED" for item in results)
