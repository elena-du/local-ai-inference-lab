from pathlib import Path

import pytest

from local_ai_lab.config import ConfigurationError, Workload, load_yaml


def test_loads_chat_workload() -> None:
    workload = Workload.load("chat")
    assert workload.id == "chat"
    assert workload.generation.temperature == 0.0
    assert workload.checks


def test_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_yaml(path)
