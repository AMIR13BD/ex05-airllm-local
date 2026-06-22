"""SDK — the single entry point for ALL business logic (guidelines §4.1).

CLI / notebooks / future GUIs call this; they never reach into services directly.
"""
import json

from orch5.services import cost, env_check
from orch5.services.benchmark import RunRecord, run_airllm
from orch5.shared import config
from orch5.shared.version import __version__


class Ex05SDK:
    """Facade over env-check, benchmarking, and cost-analysis services."""

    def __init__(self):
        self.version = __version__
        self.model_cfg = config.MODEL_CFG
        self.cost_cfg = config.COST_CFG

    # --- environment ----------------------------------------------------
    def check_env(self) -> dict:
        return env_check.report()

    # --- benchmarking ---------------------------------------------------
    def resolve_model(self, key_or_id: str) -> str:
        """Accept a config key ('main'/'tiny') or a full HF repo id."""
        return self.model_cfg["models"].get(key_or_id, key_or_id)

    def benchmark(self, model: str, quant: str, device: str = "cuda",
                  prompt_key: str = "medium_reasoning", max_new_tokens: int | None = None,
                  save: bool = True) -> RunRecord:
        """Run one AirLLM benchmark configuration and (optionally) persist it."""
        model_id = self.resolve_model(model)
        prompt = self.model_cfg["prompts"].get(prompt_key, prompt_key)
        record = run_airllm(model_id, quant, device, prompt, max_new_tokens)
        if save:
            self._save(record, f"{model}_{quant}_{record.device}.json")
        return record

    def _save(self, record: RunRecord, name: str) -> None:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RESULTS_DIR / name
        path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # --- cost analysis --------------------------------------------------
    def cost_table(self, in_tok: int, out_tok: int, energy_wh: float,
                   runtime_s: float) -> dict:
        return cost.cost_table(self.cost_cfg, in_tok, out_tok, energy_wh, runtime_s)

    def breakeven(self, provider: str, in_tok: int, out_tok: int, energy_wh: float) -> dict:
        api = cost.api_cost_nis(self.cost_cfg, provider, in_tok, out_tok)
        opex = cost.onprem_opex_nis(self.cost_cfg, energy_wh)
        vol = cost.breakeven_volume(self.cost_cfg, api, opex)
        return {"provider": provider, "api_nis_per_req": round(api, 6),
                "onprem_opex_nis_per_req": round(opex, 6),
                "capex_nis": cost.onprem_capex_nis(self.cost_cfg),
                "breakeven_requests": vol}
