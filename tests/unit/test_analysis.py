"""Unit tests for the analysis aggregation service."""
import json

from orch5.services import analysis


def _write(tmp_path, name, rec):
    (tmp_path / name).write_text(json.dumps(rec), encoding="utf-8")


def test_load_results_skips_status(tmp_path):
    _write(tmp_path, "prepare_status.json", {"x": 1})
    _write(tmp_path, "main_4bit_cuda.json", {"quant": "4bit", "ttft_s": 2.0})
    out = analysis.load_results(tmp_path)
    assert "main_4bit_cuda" in out and "prepare_status" not in out


def test_comparison_rows_excludes_baseline(tmp_path):
    _write(tmp_path, "baseline_b0.json", {"status": "failed_as_expected"})
    _write(tmp_path, "main_fp16_cuda.json",
           {"quant": "fp16", "ttft_s": 3.0, "throughput_tok_s": 0.2})
    rows = analysis.comparison_rows(analysis.load_results(tmp_path))
    assert len(rows) == 1 and rows[0]["config"] == "main_fp16_cuda"


def test_markdown_table_has_header_and_row(tmp_path):
    _write(tmp_path, "main_8bit_cuda.json", {"quant": "8bit", "ttft_s": 1.0})
    md = analysis.markdown_table(analysis.comparison_rows(analysis.load_results(tmp_path)))
    assert "| config |" in md and "main_8bit_cuda" in md


def test_markdown_table_empty():
    assert analysis.markdown_table([]) == "_no results yet_"
