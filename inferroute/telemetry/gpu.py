"""GPU telemetry via NVIDIA NVML (pynvml), fully optional.

On machines without an NVIDIA driver (or without pynvml) the monitor reports a clean
``unavailable`` state and everything else keeps working — the gateway must run on a
plain laptop with the mock backend. Samples are cached for ``cache_ttl_s`` to keep
multiple concurrent readers from hammering NVML.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from inferroute.observability.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import guarded by design
    import pynvml

    _NVML_IMPORT_OK = True
except Exception:  # ImportError or driver-level failure
    pynvml = None
    _NVML_IMPORT_OK = False


@dataclass(slots=True)
class GpuSample:
    index: int
    name: str
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None
    util_pct: float | None = None
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "power_limit_w": self.power_limit_w,
        }


@dataclass(slots=True)
class GpuStatus:
    available: bool
    reason: str = ""
    driver_version: str | None = None
    samples: list[GpuSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "driver_version": self.driver_version,
            "gpus": [s.to_dict() for s in self.samples],
        }


class GpuMonitor:
    """Process-wide NVML wrapper. Thread-safe; samples cached for cache_ttl_s."""

    def __init__(self, enabled: bool = True, cache_ttl_s: float = 1.0) -> None:
        self.enabled = enabled
        self.cache_ttl_s = cache_ttl_s
        self._lock = threading.Lock()
        self._init_attempted = False
        self._init_error = ""
        self._driver_version: str | None = None
        self._handles: list[Any] = []
        self._names: list[str] = []
        self._last_sample_ts = 0.0
        self._last_samples: list[GpuSample] = []

    def _ensure_init(self) -> bool:
        if not self.enabled:
            self._init_error = "gpu monitoring disabled by configuration"
            return False
        if not _NVML_IMPORT_OK:
            self._init_error = "pynvml not installed"
            return False
        if self._init_attempted:
            return bool(self._handles)
        self._init_attempted = True
        try:
            pynvml.nvmlInit()
            self._driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(self._driver_version, bytes):
                self._driver_version = self._driver_version.decode()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self._handles.append(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                self._names.append(name.decode() if isinstance(name, bytes) else str(name))
            logger.info("NVML initialized: %d GPU(s): %s", len(self._handles), self._names)
            return bool(self._handles)
        except Exception as exc:
            self._init_error = f"NVML init failed: {exc}"
            logger.info("GPU telemetry unavailable: %s", self._init_error)
            return False

    @property
    def available(self) -> bool:
        with self._lock:
            return self._ensure_init()

    def sample(self) -> GpuStatus:
        """Sample all GPUs (cached for cache_ttl_s). Never raises."""
        with self._lock:
            if not self._ensure_init():
                return GpuStatus(
                    available=False, reason=self._init_error or "no NVIDIA GPU detected"
                )
            now = time.monotonic()
            if self._last_samples and now - self._last_sample_ts < self.cache_ttl_s:
                return GpuStatus(
                    available=True,
                    driver_version=self._driver_version,
                    samples=list(self._last_samples),
                )
            samples: list[GpuSample] = []
            for i, handle in enumerate(self._handles):
                s = GpuSample(index=i, name=self._names[i])
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    s.mem_used_mb = mem.used / (1024 * 1024)
                    s.mem_total_mb = mem.total / (1024 * 1024)
                except Exception:
                    pass
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    s.util_pct = float(util.gpu)
                except Exception:
                    pass
                with contextlib.suppress(Exception):
                    s.temp_c = float(
                        pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    )
                try:
                    s.power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    s.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                except Exception:
                    pass
                samples.append(s)
            self._last_samples = samples
            self._last_sample_ts = now
            return GpuStatus(available=True, driver_version=self._driver_version, samples=samples)


_GLOBAL_MONITOR: GpuMonitor | None = None
_GLOBAL_LOCK = threading.Lock()


def global_gpu_monitor() -> GpuMonitor:
    global _GLOBAL_MONITOR
    with _GLOBAL_LOCK:
        if _GLOBAL_MONITOR is None:
            _GLOBAL_MONITOR = GpuMonitor()
        return _GLOBAL_MONITOR
