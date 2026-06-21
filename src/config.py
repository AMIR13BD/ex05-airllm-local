"""Central config for the EX05 AirLLM experiment.

Target machine: Intel i9-9900K (8C/16T) / 32 GB RAM / RTX 3070 8 GB /
WD Blue SN570 1 TB NVMe (~3,500 MB/s). Windows, Python 3.12.
"""
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Models -------------------------------------------------------------
TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # pipeline smoke test (cheap, fast)
MAIN_MODEL = "Qwen/Qwen2.5-14B-Instruct"    # main workload — DO NOT download
                                            # until the smoke test passes.

# --- AirLLM shard cache -------------------------------------------------
# Must live on a FAST, ROOMY drive. Full 14B sweep peaks at ~84 GB.
# Override with the AIRLLM_CACHE env var if you want it off the repo drive.
AIRLLM_CACHE = Path(os.environ.get("AIRLLM_CACHE", REPO_ROOT / ".airllm_cache"))

# --- Quantization levels -> AirLLM `compression` argument ---------------
QUANT_LEVELS = {
    "fp16": None,      # no compression (reference)
    "8bit": "8bit",
    "4bit": "4bit",
}

# --- Generation defaults (keep SMALL early; AirLLM decode is slow) ------
MAX_NEW_TOKENS_SMOKE = 16
MAX_NEW_TOKENS_BENCH = 32

# --- Output dirs --------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# --- Cost-analysis assumptions (Israel; NIS primary, USD shown) ---------
ELECTRICITY_NIS_PER_KWH = 0.60   # IEC residential incl. VAT (~₪0.60-0.64)
NIS_PER_USD = 3.70               # update to the current rate when reporting
GPU_TDP_W = 220                  # RTX 3070 board power (for energy estimate)
CPU_TDP_W = 95                   # i9-9900K TDP (PL1; used for CPU-run estimate)
