"""Immutable, project-wide constants (physical/format facts — not tunable config).

Anything a user might tune lives in config/*.json; this file holds only fixed facts.
"""
from enum import Enum

SECONDS_PER_HOUR = 3600
BYTES_PER_GIB = 1024**3
BYTES_PER_MIB = 1024**2
TOKENS_PER_MILLION = 1_000_000


class Engine(str, Enum):
    """How a configuration is executed."""

    AIRLLM_GPU = "airllm_gpu"      # AirLLM, layer-streamed through the RTX 3070
    AIRLLM_CPU = "airllm_cpu"      # AirLLM on CPU (the required CPU-vs-GPU point)
    BASELINE_DIRECT = "baseline"   # plain transformers .to('cuda') — expected to OOM


# Bytes per parameter by quantization level (for footprint reasoning in the report).
BYTES_PER_PARAM = {"fp16": 2.0, "8bit": 1.0, "4bit": 0.5}
