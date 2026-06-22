"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def cost_cfg() -> dict:
    """A self-contained cost config mirroring config/cost_config.json structure."""
    return {
        "currency": {"primary": "NIS", "nis_per_usd": 3.70},
        "electricity": {"nis_per_kwh": 0.60},
        "hardware": {"capex_nis": 1800, "lifetime_years": 3, "maintenance_nis_per_year": 200},
        "cloud_gpu": {"usd_per_hour": 0.40},
        "api_prices_usd_per_1m_tokens": {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet": {"input": 3.00, "output": 15.00},
        },
        "cache_discount": {"enabled": True, "cached_input_multiplier": 0.10},
    }


@pytest.fixture
def rate_cfg() -> dict:
    return {"services": {"default": {"requests_per_minute": 1000, "max_retries": 3,
                                     "retry_after_seconds": 0}}}
