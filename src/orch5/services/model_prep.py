"""Make a Hugging Face model consumable by AirLLM.

AirLLM's splitter (1) asserts a ``model.safetensors.index.json`` exists, and (2) drops a
decoder layer's tensors if that layer spans two source shards. Small single-file models
(e.g. Qwen2.5-0.5B) trip both. So we re-save such models as ONE ``model.safetensors`` plus a
hand-written index, and untie embeddings so an explicit ``lm_head`` shard exists. Natively
multi-shard models (the 14B) pass straight through.

The file-list check uses the HF API (no weight download), so it won't pull a big model early.
"""
import json
from pathlib import Path


def ensure_airllm_ready(repo_id: str, work_dir) -> str:
    """Return a model reference AirLLM can split (repo id, or a prepared local path)."""
    from huggingface_hub import HfApi

    if "model.safetensors.index.json" in HfApi().list_repo_files(repo_id):
        return repo_id  # natively multi-shard

    out = Path(work_dir) / (repo_id.split("/")[-1] + "-airllm")
    if (out / "model.safetensors.index.json").exists():
        return str(out)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16)
    if getattr(model.config, "tie_word_embeddings", False):
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(
            model.get_input_embeddings().weight.detach().clone())
    model.save_pretrained(out, max_shard_size="100GB", safe_serialization=True)
    AutoTokenizer.from_pretrained(repo_id).save_pretrained(out)
    del model

    single = out / "model.safetensors"
    with safe_open(str(single), framework="pt") as fh:
        weight_map = {k: "model.safetensors" for k in fh.keys()}
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": single.stat().st_size}, "weight_map": weight_map}, indent=2))
    return str(out)
