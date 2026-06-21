"""Reusable benchmark harness for one (engine, model, quant) configuration.

Captures the assignment's required metrics: TTFT, TPOT/ITL, throughput,
peak RAM, peak VRAM, total runtime, estimated energy. Saves one JSON record
per run to results/.

This is the SKELETON used by the AirLLM runs (A1-A3, C1). Wire the actual
model/generate calls in `run_airllm()` once the smoke test passes.

Run example (after smoke test):
    python src/benchmark.py --model 14b --quant 4bit --device cuda
"""
import argparse
import json
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# --------------------------------------------------------------------------
# Background samplers (RAM via psutil, VRAM + power via NVML)
# --------------------------------------------------------------------------
class ResourceSampler:
    """Polls system RAM, GPU VRAM and GPU power on a background thread."""

    def __init__(self, hz: float = 10.0):
        self.interval = 1.0 / hz
        self._stop = threading.Event()
        self.peak_ram_gb = 0.0
        self.peak_vram_mib = 0.0
        self.power_samples_w: list[float] = []
        self._t0 = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        import psutil
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            pynvml = None
            h = None
        while not self._stop.is_set():
            self.peak_ram_gb = max(self.peak_ram_gb,
                                   psutil.virtual_memory().used / 1024**3)
            if h is not None:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                self.peak_vram_mib = max(self.peak_vram_mib, mem.used / 1024**2)
                self.power_samples_w.append(
                    pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            time.sleep(self.interval)
        if h is not None:
            pynvml.nvmlShutdown()

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        self.wall_s = time.perf_counter() - self._t0

    @property
    def avg_power_w(self) -> float:
        return sum(self.power_samples_w) / len(self.power_samples_w) if self.power_samples_w else 0.0

    @property
    def energy_wh(self) -> float:
        return self.avg_power_w * (getattr(self, "wall_s", 0.0) / 3600.0)


@dataclass
class RunRecord:
    model: str
    quant: str
    device: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    tpot_ms: float = 0.0          # = ITL
    throughput_tok_s: float = 0.0
    total_runtime_s: float = 0.0
    peak_ram_gb: float = 0.0
    peak_vram_mib: float = 0.0
    avg_gpu_power_w: float = 0.0
    gpu_energy_wh: float = 0.0
    output_text: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)


def measure_generation(model, input_ids, max_new_tokens, use_cuda):
    """TTFT via a 1-token generate; TPOT from the full generate. Returns
    (ttft_s, tpot_ms, throughput, n_out, output_sequences)."""
    t = time.perf_counter()
    _ = model.generate(input_ids, max_new_tokens=1, use_cache=True,
                       return_dict_in_generate=True)
    ttft = time.perf_counter() - t

    t = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                         use_cache=True, return_dict_in_generate=True)
    total = time.perf_counter() - t

    n_out = max_new_tokens
    tpot_ms = (total - ttft) / max(n_out - 1, 1) * 1000.0
    throughput = n_out / total if total else 0.0
    return ttft, tpot_ms, throughput, n_out, out.sequences


def run_airllm(model_id, compression, device, prompt, max_new_tokens) -> RunRecord:
    import torch
    from airllm import AutoModel  # generic AutoModel -> Qwen-safe (no Class mismatch)
    from utils_model import ensure_airllm_ready

    use_cuda = (device == "cuda") and torch.cuda.is_available()
    config.AIRLLM_CACHE.mkdir(parents=True, exist_ok=True)

    rec = RunRecord(model=model_id, quant=compression or "fp16",
                    device="cuda" if use_cuda else "cpu")

    # 14B is natively multi-shard -> passes through; tiny models get an AirLLM-friendly copy.
    prepared = ensure_airllm_ready(model_id, config.REPO_ROOT / ".prepared_models")
    model = AutoModel.from_pretrained(
        prepared, compression=compression,
        layer_shards_saving_path=str(config.AIRLLM_CACHE),
    )
    messages = [{"role": "user", "content": prompt}]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = model.tokenizer(text, return_tensors="pt", return_attention_mask=False)
    input_ids = inputs["input_ids"].cuda() if use_cuda else inputs["input_ids"]
    rec.prompt_tokens = int(input_ids.shape[-1])

    with ResourceSampler() as s:
        ttft, tpot_ms, thr, n_out, seqs = measure_generation(
            model, input_ids, max_new_tokens, use_cuda)

    rec.ttft_s = ttft
    rec.tpot_ms = tpot_ms
    rec.throughput_tok_s = thr
    rec.output_tokens = n_out
    rec.total_runtime_s = s.wall_s
    rec.peak_ram_gb = s.peak_ram_gb
    rec.peak_vram_mib = s.peak_vram_mib
    rec.avg_gpu_power_w = s.avg_power_w
    rec.gpu_energy_wh = s.energy_wh
    rec.output_text = model.tokenizer.decode(seqs[0])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tiny", "14b"], default="tiny")
    ap.add_argument("--quant", choices=list(config.QUANT_LEVELS), default="4bit")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--prompt", default="Explain what an operating system page fault is.")
    ap.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS_BENCH)
    args = ap.parse_args()

    model_id = config.MAIN_MODEL if args.model == "14b" else config.TINY_MODEL
    compression = config.QUANT_LEVELS[args.quant]

    rec = run_airllm(model_id, compression, args.device, args.prompt, args.max_new_tokens)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = config.RESULTS_DIR / f"{args.model}_{args.quant}_{args.device}.json"
    fname.write_text(json.dumps(asdict(rec), indent=2, ensure_ascii=False))
    print(json.dumps(asdict(rec), indent=2, ensure_ascii=False))
    print(f"\nsaved -> {fname}")


if __name__ == "__main__":
    main()
