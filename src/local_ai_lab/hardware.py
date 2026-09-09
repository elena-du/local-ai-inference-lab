from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import psutil


@dataclass(slots=True)
class DetectedDevice:
    kind: str
    name: str
    status: str | None = None
    source: str = "os-reported"


@dataclass(slots=True)
class HardwareSnapshot:
    timestamp: str
    os: dict[str, Any]
    architecture: str
    python: str
    cpu: dict[str, Any]
    memory_bytes: int
    devices: list[DetectedDevice] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _powershell_json(script: str) -> Any:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout.lstrip("\ufeff"))
    except json.JSONDecodeError:
        return None


def discover_hardware() -> HardwareSnapshot:
    devices: list[DetectedDevice] = []
    warnings: list[str] = []
    os_data: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
    }
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")

    if platform.system() == "Windows":
        info = _powershell_json(
            "Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption,Version,BuildNumber,OSArchitecture | ConvertTo-Json -Compress"
        )
        if isinstance(info, dict):
            os_data.update({key.lower(): value for key, value in info.items()})
        cpu = _powershell_json(
            "Get-CimInstance Win32_Processor | Select-Object -First 1 "
            "Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | "
            "ConvertTo-Json -Compress"
        )
        if isinstance(cpu, dict):
            cpu_name = str(cpu.get("Name") or cpu_name)
        raw_devices = _powershell_json(
            "@(Get-PnpDevice -PresentOnly | Where-Object { "
            "$_.Class -in @('Display','ComputeAccelerator') -or $_.FriendlyName -match 'NPU|Neural' } | "
            "Select-Object Class,FriendlyName,Status) | ConvertTo-Json -Compress"
        )
        if isinstance(raw_devices, dict):
            raw_devices = [raw_devices]
        for item in raw_devices or []:
            kind = "NPU" if item.get("Class") == "ComputeAccelerator" or "NPU" in str(item.get("FriendlyName")) else "GPU"
            devices.append(DetectedDevice(kind, str(item.get("FriendlyName")), item.get("Status")))
        if not any(device.kind == "NPU" for device in devices):
            warnings.append("No NPU was detected through Windows PnP; this does not prove one is absent.")

    return HardwareSnapshot(
        timestamp=datetime.now(UTC).isoformat(),
        os=os_data,
        architecture=platform.machine(),
        python=sys.version,
        cpu={"name": cpu_name, "physical_cores": psutil.cpu_count(False), "logical_cores": psutil.cpu_count(True)},
        memory_bytes=psutil.virtual_memory().total,
        devices=devices,
        warnings=warnings,
    )
