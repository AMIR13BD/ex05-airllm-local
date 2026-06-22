"""On-prem vs API vs Cloud-GPU cost analysis (pure functions; testable).

All assumptions come from config/cost_config.json. Currency is NIS-primary with USD shown.
Break-even compares cumulative cost vs monthly request volume.
"""
from orch5.constants import SECONDS_PER_HOUR, TOKENS_PER_MILLION


def usd_to_nis(cfg: dict, usd: float) -> float:
    return usd * cfg["currency"]["nis_per_usd"]


def api_cost_nis(cfg: dict, provider: str, in_tok: int, out_tok: int,
                 cached_input_frac: float = 0.0) -> float:
    """Cost of one API request in NIS (optionally with a cached static prefix)."""
    price = cfg["api_prices_usd_per_1m_tokens"][provider]
    mult = cfg["cache_discount"]["cached_input_multiplier"] if cfg["cache_discount"]["enabled"] else 1.0
    eff_in = in_tok * ((1 - cached_input_frac) + cached_input_frac * mult)
    usd = (eff_in * price["input"] + out_tok * price["output"]) / TOKENS_PER_MILLION
    return usd_to_nis(cfg, usd)


def onprem_opex_nis(cfg: dict, energy_wh: float) -> float:
    """Per-request electricity cost in NIS (the marginal on-prem cost)."""
    return (energy_wh / 1000.0) * cfg["electricity"]["nis_per_kwh"]


def onprem_capex_nis(cfg: dict) -> float:
    """Fixed up-front + lifetime maintenance, in NIS."""
    hw = cfg["hardware"]
    return hw["capex_nis"] + hw["maintenance_nis_per_year"] * hw["lifetime_years"]


def cloud_cost_nis(cfg: dict, runtime_s: float) -> float:
    """Per-request Cloud-GPU rental cost in NIS."""
    usd = cfg["cloud_gpu"]["usd_per_hour"] * (runtime_s / SECONDS_PER_HOUR)
    return usd_to_nis(cfg, usd)


def breakeven_volume(cfg: dict, api_per_req: float, opex_per_req: float) -> float:
    """Monthly requests where on-prem (CAPEX + N*opex) beats API (N*api). inf if never."""
    denom = api_per_req - opex_per_req
    return onprem_capex_nis(cfg) / denom if denom > 0 else float("inf")


def cost_table(cfg: dict, in_tok: int, out_tok: int, energy_wh: float,
               runtime_s: float) -> dict:
    """Per-request cost (NIS) for every option, for the report's comparison table."""
    table = {p: round(api_cost_nis(cfg, p, in_tok, out_tok), 6)
             for p in cfg["api_prices_usd_per_1m_tokens"]}
    table["onprem_opex"] = round(onprem_opex_nis(cfg, energy_wh), 6)
    table["cloud_gpu"] = round(cloud_cost_nis(cfg, runtime_s), 6)
    return table
