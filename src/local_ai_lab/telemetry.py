from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from local_ai_lab.hardware import HardwareSnapshot


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions() -> dict[str, str]:
    names = ("local-ai-inference-lab", "onnxruntime", "onnxruntime-windowsml", "onnxruntime-genai", "foundry-local-sdk-winml", "httpx")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def base_metadata(root: Path, hardware: HardwareSnapshot) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(root),
        "hardware": hardware.to_dict(),
        "package_versions": package_versions(),
    }


def write_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=True) + "\n")
