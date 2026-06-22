"""Unit tests for RunRecord and the SDK's pure (non-GPU) methods."""
from orch5.sdk import Ex05SDK
from orch5.services.benchmark import RunRecord


def test_runrecord_to_dict_roundtrip():
    rec = RunRecord(model="m", quant="4bit", device="cuda", ttft_s=1.5, output_tokens=32)
    d = rec.to_dict()
    assert d["model"] == "m" and d["quant"] == "4bit" and d["ttft_s"] == 1.5
    assert "throughput_tok_s" in d and "extra" in d


def test_sdk_resolve_model_key_and_passthrough():
    sdk = Ex05SDK()
    assert sdk.resolve_model("main") == "Qwen/Qwen2.5-14B-Instruct"
    assert sdk.resolve_model("org/some-model") == "org/some-model"  # full id passthrough


def test_sdk_cost_table_nonnegative():
    sdk = Ex05SDK()
    table = sdk.cost_table(256, 32, energy_wh=5.0, runtime_s=60.0)
    assert all(v >= 0 for v in table.values())


def test_sdk_breakeven_structure():
    sdk = Ex05SDK()
    res = sdk.breakeven("claude-sonnet", 1000, 200, energy_wh=5.0)
    assert res["provider"] == "claude-sonnet"
    assert res["capex_nis"] > 0
    assert "breakeven_requests" in res
