"""Aggregate benchmark result JSONs into comparison tables (pure, testable).

Figure rendering lives in scripts/make_figures.py; this module only shapes data.
"""
import json
from pathlib import Path

from orch5.shared import config

# (json key, human label) for the comparison table / charts
METRICS = [
    ("ttft_s", "TTFT (s)"),
    ("tpot_ms", "TPOT/ITL (ms)"),
    ("throughput_tok_s", "Throughput (tok/s)"),
    ("peak_vram_mib", "Peak VRAM (MiB)"),
    ("peak_ram_gb", "Peak RAM (GB)"),
    ("total_runtime_s", "Runtime (s)"),
    ("gpu_energy_wh", "GPU energy (Wh)"),
]
_SKIP = {"prepare_status.json"}


def load_results(results_dir=None) -> dict:
    """Load every benchmark result JSON (skips status files)."""
    path = Path(results_dir or config.RESULTS_DIR)
    out = {}
    for f in sorted(path.glob("*.json")):
        if f.name in _SKIP:
            continue
        out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out


def comparison_rows(results: dict) -> list[dict]:
    """One row per AirLLM run (records with a 'quant' field; baseline excluded)."""
    rows = []
    for name, rec in results.items():
        if "quant" not in rec:
            continue
        row = {"config": name}
        row.update({label: rec.get(key) for key, label in METRICS})
        rows.append(row)
    return rows


def markdown_table(rows: list[dict]) -> str:
    """Render comparison rows as a GitHub-flavored Markdown table."""
    if not rows:
        return "_no results yet_"
    cols = ["config"] + [label for _, label in METRICS]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        cells = [str(r.get(c, "")) for c in cols]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])
