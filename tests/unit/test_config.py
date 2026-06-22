"""Unit tests for the configuration manager."""
from pathlib import Path

from orch5.shared import config


def test_load_json_reads_model_config():
    cfg = config.load_json("model_config.json")
    assert cfg["models"]["main"] == "Qwen/Qwen2.5-14B-Instruct"


def test_quant_levels_mapping():
    assert config.QUANT_LEVELS == {"fp16": None, "8bit": "8bit", "4bit": "4bit"}


def test_shards_for_is_per_model():
    a = config.shards_for("Qwen/Qwen2.5-14B-Instruct")
    b = config.shards_for("Qwen/Qwen2.5-0.5B-Instruct")
    assert a != b
    assert a.name == "Qwen2.5-14B-Instruct"
    assert isinstance(a, Path)


def test_paths_under_repo_root():
    assert config.RESULTS_DIR.parent == config.REPO_ROOT
    assert config.CONFIG_DIR.name == "config"
