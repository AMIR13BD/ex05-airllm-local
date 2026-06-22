"""Integration smoke test: tiny model end-to-end through the SDK + AirLLM.

Exercises model_prep (re-save single-file model), the AirLLM split/load/generate path, and
the resource samplers. Requires a CUDA GPU and downloads Qwen2.5-0.5B (cached after first run).
Skipped automatically when CUDA is unavailable.
"""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA GPU")


def test_tiny_airllm_generates(tmp_path):
    from orch5.sdk import Ex05SDK

    sdk = Ex05SDK()
    rec = sdk.benchmark("tiny", "4bit", "cuda", prompt_key="short_factual",
                        max_new_tokens=4, save=False)
    assert rec.output_tokens == 4
    assert rec.device == "cuda"
    assert rec.ttft_s > 0
    assert rec.throughput_tok_s > 0
    assert rec.peak_vram_mib > 0
    assert len(rec.output_text) > 0
