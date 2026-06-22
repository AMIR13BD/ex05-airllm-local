"""Benchmark one (model, quant, device) AirLLM configuration.

Captures the assignment's required metrics: TTFT, TPOT/ITL, throughput, peak RAM, peak VRAM,
total runtime, estimated GPU energy. Returns a RunRecord; persistence is the SDK's job.
"""
import time
from dataclasses import asdict, dataclass, field

from orch5.shared import config
from orch5.shared.samplers import ResourceSampler
from orch5.services.model_prep import ensure_airllm_ready


@dataclass
class RunRecord:
    model: str
    quant: str
    device: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    tpot_ms: float = 0.0           # inter-token latency == ITL
    throughput_tok_s: float = 0.0
    total_runtime_s: float = 0.0
    peak_ram_gb: float = 0.0
    peak_vram_mib: float = 0.0
    avg_gpu_power_w: float = 0.0
    gpu_energy_wh: float = 0.0
    output_text: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _measure(model, input_ids, max_new_tokens):
    """TTFT via a 1-token generate; TPOT from the full generate."""
    t = time.perf_counter()
    model.generate(input_ids, max_new_tokens=1, use_cache=True, return_dict_in_generate=True)
    ttft = time.perf_counter() - t

    t = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, use_cache=True,
                         return_dict_in_generate=True)
    total = time.perf_counter() - t
    tpot_ms = (total - ttft) / max(max_new_tokens - 1, 1) * 1000.0
    throughput = max_new_tokens / total if total else 0.0
    return ttft, tpot_ms, throughput, max_new_tokens, out.sequences


def run_airllm(model_id: str, quant: str, device: str, prompt: str,
               max_new_tokens: int | None = None) -> RunRecord:
    """Run one AirLLM benchmark configuration and return its measured RunRecord."""
    import torch
    from airllm import AutoModel

    compression = config.QUANT_LEVELS[quant]
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS_BENCH
    use_cuda = (device == "cuda") and torch.cuda.is_available()
    shards = config.shards_for(model_id)
    shards.mkdir(parents=True, exist_ok=True)

    rec = RunRecord(model=model_id, quant=quant, device="cuda" if use_cuda else "cpu")
    prepared = ensure_airllm_ready(model_id, config.PREPARED_DIR)
    model = AutoModel.from_pretrained(prepared, compression=compression,
                                      layer_shards_saving_path=str(shards))

    messages = [{"role": "user", "content": prompt}]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model.tokenizer(text, return_tensors="pt", return_attention_mask=False)
    input_ids = inputs["input_ids"].cuda() if use_cuda else inputs["input_ids"]
    rec.prompt_tokens = int(input_ids.shape[-1])

    with ResourceSampler(hz=config.SAMPLER_HZ) as s:
        ttft, tpot_ms, thr, n_out, seqs = _measure(model, input_ids, max_new_tokens)

    rec.ttft_s, rec.tpot_ms, rec.throughput_tok_s = ttft, tpot_ms, thr
    rec.output_tokens, rec.total_runtime_s = n_out, s.wall_s
    rec.peak_ram_gb, rec.peak_vram_mib = s.peak_ram_gb, s.peak_vram_mib
    rec.avg_gpu_power_w, rec.gpu_energy_wh = s.avg_power_w, s.energy_wh
    rec.output_text = model.tokenizer.decode(seqs[0])
    return rec
