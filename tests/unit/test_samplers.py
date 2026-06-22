"""Unit test for the ResourceSampler (RAM path always; NVML if a GPU is present)."""
from orch5.shared.samplers import ResourceSampler


def test_sampler_context_collects_ram_and_timing():
    with ResourceSampler(hz=50) as s:
        _ = [i * i for i in range(200_000)]  # a little work to sample across
    assert s.wall_s >= 0.0
    assert s.peak_ram_gb > 0.0          # system RAM is always sampled
    assert s.avg_power_w >= 0.0         # 0.0 if no NVML/GPU
    assert s.energy_wh >= 0.0
