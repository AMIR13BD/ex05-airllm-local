"""Configuration manager.

Loads every business value from ``config/*.json`` (guidelines §7.2/§7.3 — no hardcoded
values) and resolves project paths. Import this instead of sprinkling constants in code.
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


def load_json(name: str) -> dict:
    """Load a JSON config file from config/ by file name."""
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


MODEL_CFG = load_json("model_config.json")
COST_CFG = load_json("cost_config.json")
RATE_LIMITS = load_json("rate_limits.json")

# --- resolved paths -----------------------------------------------------
AIRLLM_CACHE = Path(os.environ.get("AIRLLM_CACHE", REPO_ROOT / ".airllm_cache"))
PREPARED_DIR = REPO_ROOT / ".prepared_models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
LOGS_DIR = REPO_ROOT / "logs"

# --- convenience accessors (read-only views into the JSON) --------------
MAIN_MODEL = MODEL_CFG["models"]["main"]
TINY_MODEL = MODEL_CFG["models"]["tiny"]
QUANT_LEVELS = MODEL_CFG["quant_levels"]            # {"fp16": None, "8bit": "8bit", "4bit": "4bit"}
PROMPTS = MODEL_CFG["prompts"]
MAX_NEW_TOKENS_SMOKE = MODEL_CFG["generation"]["max_new_tokens_smoke"]
MAX_NEW_TOKENS_BENCH = MODEL_CFG["generation"]["max_new_tokens_bench"]
SAMPLER_HZ = MODEL_CFG["sampler_hz"]


def shards_for(model_id: str) -> Path:
    """Per-model AirLLM shard dir. AirLLM uses fixed split dir names, so isolating by
    model slug prevents one model's layers from colliding with another's."""
    return AIRLLM_CACHE / model_id.split("/")[-1]
