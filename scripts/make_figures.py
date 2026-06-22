"""Render tables + figures from results/ into figures/ (+ results/comparison.md, cost_summary.json).

Run AFTER the suite:  PYTHONPATH=src uv run python scripts/make_figures.py
Safe with partial results (skips what's missing).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from orch5.services import analysis, cost  # noqa: E402
from orch5.shared import config  # noqa: E402

FIG = config.FIGURES_DIR
FIG.mkdir(parents=True, exist_ok=True)


def bar_charts(rows):
    for key, label in analysis.METRICS:
        pairs = [(r["config"], r[label]) for r in rows if r.get(label) is not None]
        if not pairs:
            continue
        names = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        plt.figure(figsize=(7, 4))
        plt.bar(names, vals, color="#4c72b0")
        plt.ylabel(label)
        plt.title(f"{label} - Qwen2.5-14B via AirLLM")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(FIG / f"metric_{key}.png", dpi=130)
        plt.close()


def breakeven_chart(results):
    rec = results.get("main_4bit_cuda") or results.get("main_8bit_cuda")
    if not rec:
        return
    opex = cost.onprem_opex_nis(config.COST_CFG, rec.get("gpu_energy_wh", 0.0))
    capex = cost.onprem_capex_nis(config.COST_CFG)
    vols = list(range(0, 200001, 2000))
    plt.figure(figsize=(7, 4.5))
    plt.plot(vols, [capex + opex * v for v in vols], lw=2, label="On-prem (CAPEX + energy)")
    for prov in config.COST_CFG["api_prices_usd_per_1m_tokens"]:
        api = cost.api_cost_nis(config.COST_CFG, prov, rec.get("prompt_tokens", 45),
                                rec.get("output_tokens", 32))
        plt.plot(vols, [api * v for v in vols], ls="--", label=f"API: {prov}")
    plt.xlabel("Requests (cumulative)")
    plt.ylabel("Total cost (NIS)")
    plt.title("On-prem vs API break-even (NIS)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "breakeven.png", dpi=130)
    plt.close()


def roofline_chart():
    peak = 20e12  # RTX 3070 ~20 TFLOPS FP16
    ai = np.logspace(-2, 3, 200)
    plt.figure(figsize=(7, 4.5))
    for label, bw in {"VRAM 448 GB/s": 448e9, "NVMe 3.5 GB/s": 3.5e9}.items():
        plt.loglog(ai, np.minimum(peak, bw * ai), label=f"roof: {label}")
    plt.scatter([80], [peak * 0.6], s=90, marker="o", label="Prefill (compute-bound)")
    plt.scatter([1], [448e9], s=90, marker="s", label="Decode (VRAM-bound)")
    plt.scatter([1], [3.5e9], s=110, marker="^", color="red", label="AirLLM (disk-bound)")
    plt.xlabel("Arithmetic intensity (FLOPs/byte)")
    plt.ylabel("Attainable FLOP/s")
    plt.title("Roofline - RTX 3070; AirLLM sits on the disk slope")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "roofline.png", dpi=130)
    plt.close()


def cost_summary(results) -> dict:
    rec = results.get("main_4bit_cuda", {})
    in_tok = rec.get("prompt_tokens", 45)
    out_tok = rec.get("output_tokens", 32)
    opex = cost.onprem_opex_nis(config.COST_CFG, rec.get("gpu_energy_wh", 0.0))
    out = {"based_on": "main_4bit_cuda", "onprem_opex_nis_per_req": round(opex, 6),
           "capex_nis": cost.onprem_capex_nis(config.COST_CFG), "providers": {}}
    for prov in config.COST_CFG["api_prices_usd_per_1m_tokens"]:
        api = cost.api_cost_nis(config.COST_CFG, prov, in_tok, out_tok)
        be = cost.breakeven_volume(config.COST_CFG, api, opex)
        out["providers"][prov] = {
            "api_nis_per_req": round(api, 6),
            # on-prem energy/req already exceeds API price -> never breaks even (valid JSON)
            "breakeven_requests": "never" if be == float("inf") else round(be),
        }
    (config.RESULTS_DIR / "cost_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    results = analysis.load_results()
    rows = analysis.comparison_rows(results)
    (config.RESULTS_DIR / "comparison.md").write_text(
        analysis.markdown_table(rows), encoding="utf-8")
    bar_charts(rows)
    breakeven_chart(results)
    roofline_chart()
    cost_summary(results)
    print(f"wrote comparison.md ({len(rows)} runs), cost_summary.json, figures -> {FIG}")


if __name__ == "__main__":
    main()
