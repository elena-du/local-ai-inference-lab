from local_ai_lab.correctness import evaluate


def test_json_schema_check() -> None:
    results = evaluate('{"api": "contract"}', [{"type": "json", "required_keys": ["api"]}])
    assert all(result.passed for result in results)


def test_invalid_json_is_reported() -> None:
    results = evaluate("not json", [{"type": "json", "required_keys": ["api"]}])
    assert results[-1].passed is False
    assert "Invalid JSON" in results[-1].detail
