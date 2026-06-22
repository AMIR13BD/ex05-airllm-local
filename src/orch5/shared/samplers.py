"""Background resource sampler: peak RAM, peak VRAM, GPU power -> energy.

Runs a daemon thread that polls system RAM (psutil) and GPU VRAM + power (NVML) while a
benchmarked generation runs. Used as a context manager around the timed region.
"""
import threading
import time


class ResourceSampler:
    """Poll system RAM, GPU VRAM and GPU power on a background thread."""

    def __init__(self, hz: float = 10.0):
        self.interval = 1.0 / hz
        self._stop = threading.Event()
        self.peak_ram_gb = 0.0
        self.peak_vram_mib = 0.0
        self.power_samples_w: list[float] = []
        self.wall_s = 0.0
        self._t0 = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        import psutil
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001 — NVML optional (CPU runs)
            pynvml, handle = None, None
        while not self._stop.is_set():
            self.peak_ram_gb = max(self.peak_ram_gb, psutil.virtual_memory().used / 1024**3)
            if handle is not None:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.peak_vram_mib = max(self.peak_vram_mib, mem.used / 1024**2)
                self.power_samples_w.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            time.sleep(self.interval)
        if handle is not None:
            pynvml.nvmlShutdown()

    def __enter__(self) -> "ResourceSampler":
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.wall_s = time.perf_counter() - self._t0

    @property
    def avg_power_w(self) -> float:
        return sum(self.power_samples_w) / len(self.power_samples_w) if self.power_samples_w else 0.0

    @property
    def energy_wh(self) -> float:
        return self.avg_power_w * (self.wall_s / 3600.0)
