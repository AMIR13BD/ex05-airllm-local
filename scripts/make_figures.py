"""Render comparison tables + figures from results/ into figures/ (+ results/comparison.md).

Run AFTER the benchmark suite: PYTHONPATH=src .venv\\Scripts\\python.exe scripts\\make_figures.py
Safe to run with partial results (skips what's missing).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from orch5.services import analysis, cost  # noqa: E402
from orch5.shared import config  # noqa: E402

FIG = config.FIGURES_DIR
FIG.mkdir(parents=True, exist_ok=True)


def bar_charts(rows):
    for key, label in analysis.METRICS:
        labels = [r["config"] for r in rows if r.get(label) is not None]
        vals = [r[label] for r in rows if r.get(label) is not None]
        if not vals:
            continue
        plt.figure(figsize=(7, 4))
        plt.bar(labels, vals, color="#4c72b0")
        plt.ylabel(label)
        plt.title(f"{label} by configuration (Qwen2.5-14B via AirLLM)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(FIG / f"metric_{key}.png", dpi=130)
        plt.close()


def breakeven_chart(results):
    rec = results.get("main_4bit_cuda") or results.get("main_8bit_cuda")
    if not rec:
        return
    in_tok = rec.get("prompt_tokens", 256)
    out_tok = rec.get("output_tokens", 32)
    energy = rec.get("gpu_energy_wh", 0.0)
    opex = cost.onprem_opex_nis(config.COST_CFG, energy)
    capex = cost.onprem_capex_nis(config.COST_CFG)
    vols = list(range(0, 200001, 2000))
    plt.figure(figsize=(7, 4.5))
    plt.plot(vols, [capex + opex * v for v in vols], label="On-prem (CAPEX + energy)", lw=2)
    for prov in config.COST_CFG["api_prices_usd_per_1m_tokens"]:
        api = cost.api_cost_nis(config.COST_CFG, prov, in_tok, out_tok)
        plt.plot(vols, [api * v for v in vols], label=f"API: {prov}", ls="--")
    plt.xlabel("Requests (cumulative)")
    plt.ylabel("Total cost (NIS)")
    plt.title("On-prem vs API break-even (NIS)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "breakeven.png", dpi=130)
    plt.close()


def main():
    results = analysis.load_results()
    rows = analysis.comparison_rows(results)
    (config.RESULTS_DIR / "comparison.md").write_text(
        analysis.markdown_table(rows), encoding="utf-8")
    bar_charts(rows)
    breakeven_chart(results)
    print(f"wrote comparison.md ({len(rows)} runs) + figures to {FIG}")


if __name__ == "__main__":
    main()
