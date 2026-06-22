"""One-off functionality demo (NOT a benchmark): ask the local 14B a normal question.

Uses the already-prepared Qwen2.5-14B 4-bit AirLLM shards on the RTX 3070 (the fastest
working setup). Single short generation; saves the real prompt + answer to
results/demo_chat.json and logs/demo_chat_transcript.md.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airllm import AutoModel  # noqa: E402

from orch5.shared import config  # noqa: E402

PROMPT = "What is a large language model? Answer in one complete simple sentence."
MAX_NEW = 40


def main() -> int:
    model_id = config.MAIN_MODEL
    shards = config.shards_for(model_id)
    model = AutoModel.from_pretrained(model_id, compression="4bit",
                                      layer_shards_saving_path=str(shards), device="cuda:0")
    messages = [{"role": "user", "content": PROMPT}]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model.tokenizer(text, return_tensors="pt", return_attention_mask=False)
    input_ids = inputs["input_ids"].cuda()

    t0 = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=MAX_NEW, use_cache=True,
                         return_dict_in_generate=True)
    elapsed = time.perf_counter() - t0

    gen_ids = out.sequences[0][input_ids.shape[-1]:]
    answer = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    rec = {"setup": "Qwen2.5-14B-Instruct via AirLLM 4-bit on RTX 3070 (GPU)",
           "command": "uv run python scripts/demo_chat.py",
           "prompt": PROMPT, "answer": answer,
           "max_new_tokens": MAX_NEW, "output_tokens": int(gen_ids.shape[-1]),
           "elapsed_s": round(elapsed, 1)}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "demo_chat.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.LOGS_DIR / "demo_chat_transcript.md").write_text(
        f"# Local chat demo (functionality check, not a benchmark)\n\n"
        f"- **Setup:** {rec['setup']}\n"
        f"- **Command:** `{rec['command']}`\n"
        f"- **max_new_tokens:** {MAX_NEW} | **elapsed:** {elapsed:.1f}s\n\n"
        f"**Prompt:**\n\n> {PROMPT}\n\n**Answer (real model output):**\n\n{answer}\n",
        encoding="utf-8")
    print("=== ANSWER ===")
    print(answer)
    print(f"=== {gen_ids.shape[-1]} tokens in {elapsed:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
