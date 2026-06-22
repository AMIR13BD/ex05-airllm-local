"""Unit test for the ResourceSampler (RAM path always; NVML if a GPU is present)."""
import time

from orch5.shared.samplers import ResourceSampler


def test_sampler_context_collects_ram_and_timing():
    with ResourceSampler(hz=50) as s:
        time.sleep(0.15)                # ensure the background thread samples >=1 time
    assert s.wall_s > 0.0
    assert s.peak_ram_gb > 0.0          # system RAM is always sampled
    assert s.avg_power_w >= 0.0         # 0.0 if no NVML/GPU
    assert s.energy_wh >= 0.0
