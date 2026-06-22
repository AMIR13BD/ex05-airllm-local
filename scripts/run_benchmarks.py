"""Run the EX05 benchmark suite. Resumable: skips configs whose result JSON exists.

Configs (docs/PLAN.md §2): B0 baseline (direct load, expect OOM), A1 fp16, A2 8bit,
A3 4bit (all AirLLM/GPU), C1 4bit CPU. Each result -> results/<name>.json; log -> logs/.

Run:  PYTHONPATH=src .venv\\Scripts\\python.exe -u scripts\\run_benchmarks.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from orch5.sdk import Ex05SDK  # noqa: E402
from orch5.shared import config  # noqa: E402

LOG = config.LOGS_DIR / "benchmark.log"
CONFIGS = [
    {"id": "B0", "kind": "baseline"},
    {"id": "A1", "kind": "airllm", "model": "main", "quant": "fp16", "device": "cuda"},
    {"id": "A2", "kind": "airllm", "model": "main", "quant": "8bit", "device": "cuda"},
    {"id": "A3", "kind": "airllm", "model": "main", "quant": "4bit", "device": "cuda"},
    # CPU comparison MUST be fp16: bitsandbytes 4/8-bit is CUDA-only. Few tokens — 14B on
    # CPU is extremely slow; per-token metrics (TTFT/TPOT) are still comparable.
    {"id": "C1", "kind": "airllm", "model": "main", "quant": "fp16", "device": "cpu",
     "max_new_tokens": 4},
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_baseline(out: Path) -> None:
    """Direct transformers load onto the 8 GB GPU — expected CUDA OOM (the evidence)."""
    import torch
    from transformers import AutoModelForCausalLM
    model_id = config.MAIN_MODEL
    rec = {"config": "B0_baseline_direct_gpu", "model": model_id,
           "engine": "transformers from_pretrained(device_map=cuda)"}
    t0 = time.time()
    try:
        AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                             device_map="cuda")
        rec["status"] = "loaded_unexpectedly"
    except Exception as exc:  # noqa: BLE001 — the OOM IS the result
        rec["status"] = "failed_as_expected"
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)[:600]
    rec["elapsed_s"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    log(f"B0 baseline: {rec['status']} ({rec.get('error_type', '')}) in {rec['elapsed_s']}s")


def main() -> int:
    sdk = Ex05SDK()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        if cfg["kind"] == "baseline":
            name = "baseline_b0.json"
        else:
            name = f"{cfg['model']}_{cfg['quant']}_{cfg['device']}.json"
        out = config.RESULTS_DIR / name
        if out.exists():
            log(f"{cfg['id']} skip (exists: {name})")
            continue
        log(f"{cfg['id']} START -> {name}")
        t0 = time.time()
        try:
            if cfg["kind"] == "baseline":
                run_baseline(out)
            else:
                rec = sdk.benchmark(cfg["model"], cfg["quant"], cfg["device"], save=True,
                                    max_new_tokens=cfg.get("max_new_tokens"))
                log(f"{cfg['id']} DONE: TTFT={rec.ttft_s:.2f}s TPOT={rec.tpot_ms:.0f}ms "
                    f"thr={rec.throughput_tok_s:.3f} tok/s peakVRAM={rec.peak_vram_mib:.0f}MiB "
                    f"peakRAM={rec.peak_ram_gb:.1f}GB in {(time.time()-t0)/60:.1f}min")
        except Exception as exc:  # noqa: BLE001 — record and continue
            log(f"{cfg['id']} ERROR: {type(exc).__name__}: {str(exc)[:300]}")
    log("SUITE PASS COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
