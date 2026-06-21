"""Phase-2 pipeline smoke test: tiny model + AirLLM + 4-bit compression.

Goal: prove the AirLLM plumbing end-to-end (download -> shard -> mmap ->
generate) on a CHEAP model before touching Qwen2.5-14B. Quality is irrelevant
here; we only care that it runs and produces tokens.

Run:  python src/smoke_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from utils_model import ensure_airllm_ready  # noqa: E402


def main():
    # compression: "fp16" (None) | "8bit" | "4bit"  (default 4bit)
    comp_arg = sys.argv[1] if len(sys.argv) > 1 else "4bit"
    compression = None if comp_arg == "fp16" else comp_arg

    config.AIRLLM_CACHE.mkdir(parents=True, exist_ok=True)
    prepared = ensure_airllm_ready(config.TINY_MODEL,
                                   config.REPO_ROOT / ".prepared_models")
    print(f"[smoke] model        : {config.TINY_MODEL}")
    print(f"[smoke] airllm input : {prepared}")
    print(f"[smoke] shard cache  : {config.AIRLLM_CACHE}")
    print(f"[smoke] compression  : {comp_arg}")

    import torch
    from airllm import AutoModel

    use_cuda = torch.cuda.is_available()
    print(f"[smoke] CUDA         : {use_cuda}")

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(
        prepared,
        compression=compression,
        layer_shards_saving_path=str(config.AIRLLM_CACHE),
    )
    t_load = time.perf_counter() - t0
    print(f"[smoke] load+shard   : {t_load:.1f}s")

    messages = [{"role": "user", "content": "In one sentence, what is virtual memory?"}]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = model.tokenizer(text, return_tensors="pt", return_attention_mask=False)
    input_ids = inputs["input_ids"]
    if use_cuda:
        input_ids = input_ids.cuda()

    # crude TTFT: time to produce the first new token
    t1 = time.perf_counter()
    _ = model.generate(input_ids, max_new_tokens=1, use_cache=True,
                       return_dict_in_generate=True)
    ttft = time.perf_counter() - t1
    print(f"[smoke] TTFT (~1 tok): {ttft:.2f}s")

    t2 = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=config.MAX_NEW_TOKENS_SMOKE,
                         use_cache=True, return_dict_in_generate=True)
    gen_time = time.perf_counter() - t2
    n = config.MAX_NEW_TOKENS_SMOKE
    print(f"[smoke] {n} tokens    : {gen_time:.2f}s  ({n / gen_time:.2f} tok/s)")

    print("[smoke] output:")
    print(model.tokenizer.decode(out.sequences[0]))
    print("\n[smoke] PASS — AirLLM pipeline works. Safe to scale to 14B.")


if __name__ == "__main__":
    main()
