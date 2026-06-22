# EX05 — Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

> **Deep-dive technical report.** Running **Qwen2.5-14B-Instruct** (FP16 ≈ 29 GB — far too
> big for the GPU) on an **i9-9900K / 32 GB RAM / RTX 3070 (8 GB) / WD SN570 NVMe** box, by
> streaming it one layer at a time with **AirLLM**, comparing **FP16 / 8-bit / 4-bit**
> quantization, and analyzing the result both technically and economically (on-prem vs API).

This README **is** the report. See [`docs/PRD.md`](docs/PRD.md), [`docs/PLAN.md`](docs/PLAN.md),
[`docs/TODO.md`](docs/TODO.md), [`docs/PROMPTS.md`](docs/PROMPTS.md).

---

## 1. Hardware & model justification

| Component | Spec | Role in the experiment |
|---|---|---|
| GPU | RTX 3070, **8 GB** (confirmed `nvidia-smi`; WMI's 4 GB is a known bug) | the 8 GB wall the direct run hits |
| CPU | i9-9900K, 8C/16T | CPU comparison run; prefill matmuls |
| RAM | 32 GB | caps the FP16 working set |
| Storage | WD Blue SN570 **1 TB NVMe** (~3.5 GB/s) | **the real AirLLM bottleneck** — weights stream from here every token |

**Why Qwen2.5-14B-Instruct:** FP16 ≈ 29 GB cannot fit 8 GB VRAM (so a direct run *must*
fail), it stresses 32 GB RAM, finishes in the time budget, and Qwen is AirLLM-`AutoModel`
friendly. (32B would blow the time budget; 8B fits in 32 GB RAM, weakening the story.)

## 2. Environment & the "don't use the newest libraries" lesson

AirLLM 2.11 broke against every current release; the project is pinned to a coherent
2024-era stack (single source of truth: [`pyproject.toml`](pyproject.toml) + `uv.lock`):

| Library | "latest" (broke) | pinned | failure |
|---|---|---|---|
| torch | 2.6.0 | **2.4.1+cu124** | streamed weights stranded on `meta` device |
| transformers | 5.x / 4.46 | **4.44.2** | ≥4.45 refactored Qwen2's forward |
| optimum | 2.2 | **1.23.3** | 2.x removed `bettertransformer` (airllm import) |
| tokenizers / hub | 0.20 / 1.x | **0.19.1 / 0.24.7** | newer `tokenizer.json` format |
| numpy / datasets | 2.4 / 5.0 | **1.26.4 / 2.21.0** | pyarrow/numpy2 access-violation segfault |

Three AirLLM model quirks were also handled in `src/orch5/services/model_prep.py` (needs a
multi-shard `index.json`; single-file models must be re-saved as one file so no layer spans a
shard boundary; small Qwen models need untied embeddings for an explicit `lm_head`).

## 3. Experiment

Engine matrix (see `docs/PLAN.md` §2): **B0** direct `transformers.to('cuda')` (baseline,
expected OOM) · **A1/A2/A3** AirLLM FP16/8-bit/4-bit on GPU · **C1** AirLLM FP16 on CPU.
Fixed prompt (a 45-token "explain page faults"), 32 new tokens (4 for the slow CPU run).
Metrics captured per run: TTFT, TPOT/ITL, throughput, peak RAM, peak VRAM, runtime, GPU
energy (NVML), plus the generated text. Reproduce: see §8.

## 4. Results

**Baseline B0 (direct load):** ❌ `CUDA out of memory` — *"Tried to allocate … GPU 0 has a
total capacity of 8.00 GiB of which 0 bytes is free … 22.46 GiB is allocated by PyTorch."*
The model tried to put **22.46 GB on an 8 GB GPU**. This is the bottleneck (RQ1): **VRAM**.

**AirLLM sweep (GPU), Qwen2.5-14B, 32 output tokens:**

| Config | TTFT (s) | TPOT/ITL (s/tok) | Throughput (tok/s) | Peak VRAM (MiB) | Peak RAM (GB) | Runtime (min) | Energy (Wh) | Avg GPU power (W) |
|---|---|---|---|---|---|---|---|---|
| A1 FP16 | 36.9 | 33.4 | 0.030 | 3 581 | 17.7 | 18.6 | 11.5 | 37 |
| A2 8-bit | 23.6 | 11.5 | 0.084 | 4 353 | 16.0 | 6.8 | 4.7 | 42 |
| A3 4-bit | 14.8 | 9.0 | 0.109 | 5 091 | 17.1 | 6.1 | 4.8 | 57 |
| C1 FP16 **(CPU)** | 570.7 | 620.0 | 0.0016 | — | 18.9 | 50.0* | — | — |

\* C1 generated only **4 tokens** (14B on CPU is ~18× slower than GPU); per-token metrics
(TTFT/TPOT) are still comparable. CPU TPOT **620 s/token** vs GPU FP16 **33 s/token** — the CPU
is bottlenecked on *both* disk streaming **and** slow CPU matmuls, while the GPU only waits on
disk. This is the assignment's CPU-vs-GPU comparison.

Per-metric charts:

| | |
|---|---|
| ![TTFT](figures/metric_ttft_s.png) | ![TPOT](figures/metric_tpot_ms.png) |
| ![Throughput](figures/metric_throughput_tok_s.png) | ![Peak VRAM](figures/metric_peak_vram_mib.png) |

Full machine-readable table: [`results/comparison.md`](results/comparison.md); raw per-run JSON
in [`results/`](results/).

**Output quality** stayed coherent at every quant level (all correctly explain a page fault),
so the accuracy "red line" (RQ3) is **not reached at 4-bit** for these prompts.

## 5. Analysis — why the numbers look like this (concept linkage)

- **Peak VRAM is 3.5–5 GB, never near 8 GB** → AirLLM holds only ~one layer + KV-cache on the
  GPU at a time. That is exactly why a model whose weights are 29 GB *runs at all* on 8 GB.
- **The bottleneck moved from VRAM-bandwidth to DISK-bandwidth.** Normal decode is
  memory-bound (re-reads weights from VRAM each token); AirLLM re-reads **all weights from the
  NVMe** each token. So TPOT is seconds, not milliseconds — the price of "infinite VRAM."
  This is the OS **virtual-memory / paging** analogy: SafeTensors flat buffer → `mmap` → page
  faults pull the needed layer from disk, then evict it.
- **Quantization helps by shrinking the per-token disk read.** FP16 streams ~28 GB/token
  (33.4 s); 4-bit streams ~3.8 GB/token (9.0 s). Not perfectly proportional — per-layer Python
  overhead and 4/8-bit dequantization add a floor — but the direction is unambiguous.
- **GPU avg power is only 37–57 W** (TDP 220 W) → the GPU idles waiting on disk. Concrete proof
  the system is **I/O-bound, not compute-bound** (Roofline: far left on the disk slope, not the
  compute roof). Prefill (TTFT) is the compute-heavy GEMM phase; Decode (TPOT) is the
  memory/disk-movement phase — here both are disk-dominated.
- **Peak VRAM rises with more quantization** (3 581 → 5 091 MiB) — bitsandbytes keeps
  dequantization workspace on the GPU, a small counter-intuitive cost of compression.

### Research questions
- **RQ1 (bottleneck):** VRAM — the 8 GB GPU vs a 22.46 GB load; identified by the baseline OOM.
- **RQ2 (AirLLM):** Layer-by-layer mmap streaming; reallocates so peak VRAM ≈ 1 layer + KV
  (3.5–5 GB). Same idea as OS paging — disk as backing store.
- **RQ3 (quantization):** ↓ footprint, ↓ TTFT/TPOT, ↑ throughput, quality preserved at 4-bit
  for these prompts (red line not hit).
- **RQ4 (Prefill/Decode):** TTFT ↔ prefill (+KV build); TPOT/ITL ↔ decode. Both disk-bound here.
- **RQ5 (price of local):** GPU 0.03–0.11 tok/s (9–33 s/token); **CPU 0.0016 tok/s
  (620 s/token)**. Usable for offline/batch, never interactive — the cost of running an
  otherwise-impossible model is latency.
- **RQ6 (on-prem vs API):** **API is cheaper at every volume here** (see §6) — AirLLM's
  inefficiency makes on-prem worthwhile only for privacy/data-control, not cost.

## 6. Cost analysis — on-prem vs API vs Cloud GPU (₪)

Assumptions in [`config/cost_config.json`](config/cost_config.json) (₪0.60/kWh; CAPEX ₪1 800 +
maintenance, amortized 3 y; published Claude/GPT-4o-mini token prices; Prompt/Context-Caching
modeled). Computed in `src/orch5/services/cost.py`; summary in
[`results/cost_summary.json`](results/cost_summary.json).

**Per-request cost (NIS), based on the 4-bit GPU run (45 in / 32 out tokens, 4.82 Wh):**

| Option | NIS / request |
|---|---|
| On-prem **energy only** (OPEX) | **0.00290** |
| API — GPT-4o-mini | 0.000096 |
| API — Claude Haiku | 0.00061 |
| API — Claude Sonnet | 0.00228 |

**Break-even: never.** On-prem's *energy alone* (₪0.0029/req) already exceeds every API's
per-request price — before adding the ₪2 400 CAPEX. So `on-prem = 2400 + 0.0029·N` stays above
`API = price·N` for all N (break-even volume = ∞). See `figures/breakeven.png`:

![break-even](figures/breakeven.png)

**Why:** AirLLM trades efficiency for the *ability to run at all* — long runtimes burn more
energy per request than an API charges. Prompt/Context Caching only widens the gap (cached
input tokens bill ~10× cheaper). **Recommendation:** for this 14B-on-8 GB scenario, use an
**API** for cost/latency; choose **on-prem** only when **data privacy / no-egress** is
mandatory (the lecture's core on-prem argument). A Cloud-GPU rental that *fits* the model would
beat both on throughput but also keeps data off-box.

## 7. Original extension — quantization-quality sweep + Roofline
The FP16/8-bit/4-bit sweep (§4) **is** the quantization study (footprint ↓, latency ↓, quality
held). The Roofline below places the three regimes on the RTX 3070's roofs: Prefill near the
compute roof, Decode on the VRAM-bandwidth slope, and **AirLLM far down on the NVMe-disk
slope** — a visual proof that AirLLM is disk-I/O-bound (matching the 37–57 W idle-GPU evidence).

![roofline](figures/roofline.png)

## 8. Reproduce
```powershell
uv sync                                   # creates .venv from pyproject + uv.lock
uv run python -m orch5.main env           # verify CUDA + versions
uv run python scripts/prepare_models.py   # download + split 14B (resumable)
uv run python scripts/run_benchmarks.py   # baseline + AirLLM sweep (resumable)
uv run python scripts/make_figures.py     # tables + figures
uv run pytest --cov=src                    # tests (>=85%)
```
Set secrets via `.env` (copy `.env-example`); never commit it. HF token optional for Qwen2.5.
