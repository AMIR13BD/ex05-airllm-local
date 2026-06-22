"""Run a single benchmark configuration (used directly and as a subprocess by the suite).

For CPU runs, launch with CUDA_VISIBLE_DEVICES="" so AirLLM places ALL layers on the CPU
(otherwise it auto-selects the visible GPU and mismatches the CPU input tensor).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from orch5.sdk import Ex05SDK  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="main")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    a = ap.parse_args()
    rec = Ex05SDK().benchmark(a.model, a.quant, a.device, max_new_tokens=a.max_new_tokens,
                              save=True)
    print(json.dumps({"device": rec.device, "ttft_s": rec.ttft_s, "tpot_ms": rec.tpot_ms,
                      "throughput_tok_s": rec.throughput_tok_s, "peak_ram_gb": rec.peak_ram_gb,
                      "total_runtime_s": rec.total_runtime_s}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
