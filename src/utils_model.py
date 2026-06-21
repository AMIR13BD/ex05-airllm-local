"""Make a model consumable by AirLLM.

AirLLM's layer splitter has two requirements that trip up small models:

1. It asserts `model.safetensors.index.json` exists (multi-shard layout). Small models
   (e.g. Qwen2.5-0.5B) ship as a single `model.safetensors` with no index.
2. Its splitter DROPS tensors when a single decoder layer's weights span two source
   safetensors shards (observed: a layer ends up missing `input_layernorm`, leaving that
   weight on the `meta` device at inference -> "not on the expected device meta!").

So for single-file models we re-save them as ONE `model.safetensors` (max_shard_size huge =>
no per-tensor shard boundaries to straddle) and then hand-write an index.json mapping every
tensor to that one file. AirLLM then splits per-layer correctly.

`ensure_airllm_ready` checks the repo's file list over the HF API (NO weight download), so it
will NOT prematurely download the 14B during the smoke-test phase. Big models (14B) already
ship multi-shard with an index AND their layers are far smaller than the 5 GB shard size, so
no layer spans a boundary -> they pass through untouched.
"""
import json
from pathlib import Path


def ensure_airllm_ready(repo_id: str, work_dir) -> str:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id)
    if "model.safetensors.index.json" in files:
        return repo_id  # natively multi-shard; AirLLM handles it directly

    out = Path(work_dir) / (repo_id.split("/")[-1] + "-airllm")
    if (out / "model.safetensors.index.json").exists():
        return str(out)  # already prepared

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    print(f"[prep] {repo_id} is single-file; preparing AirLLM-friendly copy -> {out}")
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16)

    # Untie embeddings: AirLLM's splitter expects an explicit lm_head shard. Small Qwen
    # models tie lm_head to embed_tokens, so save_pretrained omits lm_head.weight.
    if getattr(model.config, "tie_word_embeddings", False):
        print("[prep] untying embeddings so an explicit lm_head.weight is written")
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(
            model.get_input_embeddings().weight.detach().clone())

    # Save as a SINGLE safetensors file (no shard boundary can split a layer).
    model.save_pretrained(out, max_shard_size="100GB", safe_serialization=True)
    AutoTokenizer.from_pretrained(repo_id).save_pretrained(out)
    del model

    # Hand-write the index.json AirLLM requires, pointing every tensor at the single file.
    single = out / "model.safetensors"
    with safe_open(str(single), framework="pt") as f:
        weight_map = {k: "model.safetensors" for k in f.keys()}
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": single.stat().st_size}, "weight_map": weight_map},
        indent=2))
    print(f"[prep] wrote index.json with {len(weight_map)} tensors -> single model.safetensors")
    return str(out)
