"""Unit tests for the cost-analysis service (pure functions)."""
import math

from orch5.services import cost


def test_usd_to_nis(cost_cfg):
    assert cost.usd_to_nis(cost_cfg, 10) == 37.0


def test_api_cost_no_cache(cost_cfg):
    # 1M input + 1M output for gpt-4o-mini = (0.15 + 0.60) USD = 0.75 USD -> *3.7 NIS
    nis = cost.api_cost_nis(cost_cfg, "gpt-4o-mini", 1_000_000, 1_000_000)
    assert math.isclose(nis, 0.75 * 3.7, rel_tol=1e-9)


def test_api_cost_with_full_cache_is_cheaper(cost_cfg):
    full = cost.api_cost_nis(cost_cfg, "claude-sonnet", 1_000_000, 100, cached_input_frac=0.0)
    cached = cost.api_cost_nis(cost_cfg, "claude-sonnet", 1_000_000, 100, cached_input_frac=1.0)
    assert cached < full  # caching the static prefix lowers cost


def test_onprem_opex(cost_cfg):
    # 1000 Wh = 1 kWh * 0.60 NIS
    assert math.isclose(cost.onprem_opex_nis(cost_cfg, 1000.0), 0.60, rel_tol=1e-9)


def test_onprem_capex(cost_cfg):
    assert cost.onprem_capex_nis(cost_cfg) == 1800 + 200 * 3


def test_cloud_cost(cost_cfg):
    # 1 hour at $0.40 -> 0.40 USD -> NIS
    assert math.isclose(cost.cloud_cost_nis(cost_cfg, 3600), 0.40 * 3.7, rel_tol=1e-9)


def test_breakeven_finite_when_api_dearer(cost_cfg):
    vol = cost.breakeven_volume(cost_cfg, api_per_req=0.01, opex_per_req=0.001)
    assert vol == (1800 + 600) / (0.01 - 0.001)


def test_breakeven_inf_when_api_cheaper(cost_cfg):
    assert cost.breakeven_volume(cost_cfg, api_per_req=0.001, opex_per_req=0.01) == float("inf")


def test_cost_table_keys(cost_cfg):
    t = cost.cost_table(cost_cfg, 256, 32, energy_wh=5.0, runtime_s=60.0)
    assert {"gpt-4o-mini", "claude-sonnet", "onprem_opex", "cloud_gpu"} <= set(t)
    assert all(v >= 0 for v in t.values())
