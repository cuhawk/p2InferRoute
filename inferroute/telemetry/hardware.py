"""Hardware/environment fingerprint for benchmark reproducibility.

Every benchmark artifact embeds this: OS, CPU, core count, RAM, GPU, Python and package
versions. No benchmark claim without its hardware context.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any

from inferroute import __version__
from inferroute.telemetry.gpu import GpuMonitor


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            return platform.processor() or "unknown"
    if sys.platform == "linux":
        try:
            for line in Path_read_lines("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or "unknown"
    if sys.platform == "darwin":
        return platform.processor() or "unknown"
    return platform.processor() or "unknown"


def Path_read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


def _ram_total_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def hardware_fingerprint(gpu: GpuMonitor | None = None) -> dict[str, Any]:
    gpu_monitor = gpu
    if gpu_monitor is None:
        from inferroute.telemetry.gpu import global_gpu_monitor

        gpu_monitor = global_gpu_monitor()
    gpu_status = gpu_monitor.sample()
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": {
            "name": _cpu_name(),
            "logical_cores": _cpu_count(),
            "arch": platform.machine(),
        },
        "ram_total_gb": _ram_total_gb(),
        "gpu": gpu_status.to_dict(),
        "hostname": platform.node(),
        "packages": {
            "inferroute": __version__,
            "fastapi": _package_version("fastapi"),
            "httpx": _package_version("httpx"),
            "pydantic": _package_version("pydantic"),
            "uvicorn": _package_version("uvicorn"),
        },
    }


def _cpu_count() -> int:
    try:
        import os

        return os.cpu_count() or 0
    except Exception:
        return 0
