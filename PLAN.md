# PLAN — EX05 Technical Plan (tailored to i9-9900K / 32 GB / RTX 3070 8 GB / SSD)

This is the phased execution plan. It implements the goals in `PRD.md` and feeds the ordered
checklist in `TODO.md`. Numbers below are *planning estimates*; the report uses **measured**
values.

---

## 0. Hardware profile (the constraints we design around)

| Component | Spec | Why it matters here |
|-----------|------|---------------------|
| CPU | i9-9900K, 8C/16T, 3.6 GHz (5.0 turbo), AVX2 | CPU AirLLM path + Prefill matmuls; sets `OMP_NUM_THREADS`. |
| RAM | 32 GB | Caps the FP16 working set; FP16-14B weights (~29 GB) ≈ whole RAM ⇒ swap risk = the memory-bound story. |
| GPU | RTX 3070, **8 GB** GDDR6, ~448 GB/s, Ampere CC 8.6, ~20 TFLOPS FP16 (Tensor Cores) | 8 GB is the wall the direct run hits; AirLLM streams one layer at a time through it. |
| Storage | ~1 TB SSD, **~193 GB free** (**confirm NVMe vs SATA**) | **The real AirLLM bottleneck.** Every decoded token re-reads all layer shards from disk. 193 GB free comfortably covers the ~84 GB sweep. |

**Pre-flight facts to capture (Phase 1):**
- `nvidia-smi` → confirm 8 GB VRAM, driver/CUDA version (ignore the WMI 4 GB bug).
- Disk type + sequential read (CrystalDiskMark / `Get-PhysicalDisk` / `fio`).
- CPU/RAM via `wmic` / `lscpu` / Task Manager.
- **Free space gate:** confirm free space (`Get-PSDrive` / `dir`) **before** downloading —
  need **~85 GB** for the full sweep; you have **~193 GB** ✓. Keep ~20 GB clear for the
  Windows pagefile.
- **Where shards live:** the fastest, roomiest drive via `layer_shards_saving_path`. (Native
  Windows here; if you ever switch to WSL2, put shards on **native ext4**, never `/mnt/c`.)

---

## 1. Model strategy

### Main model — `Qwen/Qwen2.5-14B-Instruct`
- 14.7B params, 48 decoder layers, hidden 5120, GQA, SafeTensors (mmap-friendly — good).
- Footprint per quant level (weights only, approx):

  | Level | AirLLM `compression` | Weight size | Fits 8 GB VRAM directly? |
  |-------|----------------------|-------------|--------------------------|
  | FP16  | `None`               | ~29 GB      | ❌ (this is the point) |
  | 8-bit | `'8bit'`             | ~15 GB      | ❌ |
  | 4-bit | `'4bit'`             | ~9 GB       | ❌ (still > 8 GB as a whole; AirLLM streams it) |

  With AirLLM the **whole** model never sits in VRAM — peak VRAM ≈ *largest single layer*
  (~0.5–0.7 GB) **+ KV-cache**, which is why it runs at all on 8 GB.

### Storage budget (you have ~193 GB free — comfortable)
AirLLM keeps a **separate** layer-sharded copy *in addition to* the HF source download, and
the FP16 source must be downloaded before any quant level can be built. Budget for both on
the same drive:

| Item | Size |
|------|------|
| HF source download (FP16, mandatory) | ~29 GB |
| AirLLM shards — FP16 | ~29 GB |
| AirLLM shards — 8-bit | ~15 GB |
| AirLLM shards — 4-bit | ~9 GB |
| Tiny verification model | ~2 GB |
| **Peak with all sets resident** | **~84 GB** |
| **Free space available** | **~193 GB ✓** |

Headroom is fine. Two rules still apply: (1) **preflight free-space gate** before any
download; (2) if disk ever gets tight, the HF source can be deleted *after* all shard sets
are built (inference reads only the shards), and shard sets can be built/measured one at a
time. Keep ~20 GB clear for the Windows pagefile (FP16's ~29 GB working set vs 32 GB RAM).

### Pipeline-verification model — tiny Qwen2.5 (0.5B or 1.5B)
- Run first, with **4-bit** + `max_new_tokens` ~16, only to prove the AirLLM "plumbing"
  (download → shard → mmap → generate) works end-to-end. Quality is irrelevant here.

### Why not 32B / why not 8B
- 32B (FP16 ~65 GB) → heaviest story but per-token disk reads + ~65 GB download risk
  blowing the 11 h budget. 8B (FP16 ~16 GB) → fits in 32 GB RAM, so the "doesn't fit"
  narrative is weak. **14B is the sweet spot.** (8B is the documented fallback if the SSD
  turns out to be SATA.)

---

## 2. Experiment matrix

Fixed prompt set (e.g., 3 prompts: short factual, medium reasoning, long-context summarize)
and fixed `max_new_tokens` per tier. Configurations to run:

| ID | Engine / device | Model | Quant | Purpose |
|----|-----------------|-------|-------|---------|
| B0 | HF `transformers` → `.to('cuda')` direct | 14B | FP16 | **Baseline**: expect CUDA OOM (compute/mem wall). |
| B1 | Ollama or HF direct, CPU/offload | 14B | — | Baseline: "runs" but unbearably slow / swaps. |
| V0 | AirLLM, **GPU** | tiny | 4-bit | Pipeline smoke-test. |
| A1 | AirLLM, **GPU** | 14B | FP16 | Main run (no compression). |
| A2 | AirLLM, **GPU** | 14B | 8-bit | Quant comparison. |
| A3 | AirLLM, **GPU** | 14B | 4-bit | Quant comparison (lightest). |
| C1 | AirLLM, **CPU** | 14B | 4-bit | Required CPU-vs-GPU comparison point. |

> Keep token counts modest (e.g., prefill prompt ~256 tokens, decode 16–64 tokens). AirLLM
> decode is slow; you need *clean* numbers, not many tokens.

---

## 3. Metrics & how each is measured

| Metric | Definition | Measurement method |
|--------|-----------|--------------------|
| **TTFT** (Time To First Token) | request → first output token | `time.perf_counter()` around prefill / first streamed token (`TextIteratorStreamer`). Reflects **Prefill** (compute + KV-cache build). |
| **TPOT / ITL** | mean ms per output token after the first | `(t_last − t_first)/(n_out − 1)`; also log per-token deltas. Reflects **Decode** (memory/disk movement). |
| **Throughput** | output tokens/sec | `n_out / decode_time`. |
| **Peak RAM** | max process RSS + system used | `psutil` sampler thread (e.g., 10 Hz) over the run; record process + `virtual_memory()`. |
| **Peak VRAM** | max GPU memory used | `torch.cuda.max_memory_allocated()` and/or `pynvml`/`nvidia-smi` polling. |
| **Total runtime** | wall clock per config | `perf_counter` start→end. |
| **Estimated energy/power** | Wh and avg W | Integrate GPU `power.draw` via `pynvml.nvmlDeviceGetPowerUsage` over time; add CPU estimate (≈ TDP×utilization, RAPL if available). Energy → cost in §6. |
| **Quality** (extension) | accuracy proxy per quant | Perplexity on a fixed small text (e.g., a WikiText-2 snippet) + a short rubric on fixed prompts. |

All raw samples saved to `results/*.json|csv` (consistent units) so figures are reproducible.

**A reusable `benchmark()` harness** (in `src/`) wraps any (engine, model, quant) config,
runs the prompt set, and emits one results record per run.

---

## 4. The science to write up (concept-linkage)

- **Prefill = compute-bound (GEMM):** whole prompt processed in parallel, one matrix×matrix
  pass; bounded by FLOPs/Tensor Cores. Shows up as **TTFT**.
- **Decode = memory-bound (GEMV):** one token at a time, must re-read all weights from
  memory each step; bounded by bandwidth. Shows up as **TPOT/ITL**.
- **AirLLM = disk-I/O-bound:** because weights are streamed from SSD per layer, the true
  bottleneck moves *below* RAM bandwidth to **disk bandwidth** — the OS **paging / mmap**
  analogy: `SafeTensors` flat byte buffer → `mmap` → page faults pull only the needed layer
  into VRAM, then evict. This is exactly virtual memory's "pretend you have 1 TB of RAM."
- **VRAM role:** why 8 GB blocks the direct load but suffices for one layer + KV-cache.
- **Roofline (extension):** arithmetic intensity (FLOPs/byte) on x, attainable FLOP/s on y;
  Prefill near the compute roof, Decode on the bandwidth slope, **AirLLM far left on the
  *disk* slope** — visual proof of which resource is the limiter.

---

## 5. Software stack & key snippets

```bash
# Phase 1 — environment (do NOT use the bleeding-edge Python)
uv venv --python 3.11 .venv
source .venv/bin/activate            # (Windows: .venv\Scripts\activate)
uv pip install airllm transformers accelerate bitsandbytes \
               torch --index-url <cuda-build> \
               psutil pynvml matplotlib pandas numpy
export HF_TOKEN=...                   # never hardcode
```

```python
# AirLLM — generic AutoModel for Qwen (avoids Class mismatch), shards on a FAST drive
from airllm import AutoModel
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    compression="4bit",               # None | "8bit" | "4bit"
    layer_shards_saving_path="/data/airllm_cache",   # fast NVMe, not C:/ , not /mnt/c
)
# generate with low max_new_tokens for early runs; stream for TTFT/ITL timestamps
```

```python
# Baseline that should FAIL on 8 GB (capture the OOM as evidence)
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct",
                                         torch_dtype="float16").to("cuda")  # expect CUDA OOM
```

---

## 6. Cost analysis plan (on-prem vs API vs Cloud GPU) — ₪ primary, USD shown

**Assumptions to state explicitly (all tunable in one config block):**
- Electricity: **₪0.60/kWh** (Israel residential, incl. VAT ≈ ₪0.60–0.64). USD ≈ $0.16.
- Hardware CAPEX: marginal GPU cost (RTX 3070 ≈ ₪1,500–2,000) **or** full PC; amortized over
  ~3 years. Add small OPEX for maintenance.
- Measured **energy per request** (from §3) → on-prem OPEX/request.
- API price per token (input+output) for 2–3 references, e.g. **Claude Haiku/Sonnet** and
  **GPT-4o-mini**; compute per-request and per-N-requests cost.
- **Cloud GPU:** rent ≈ equivalent GPU per-hour × runtime/request.

**Outputs:**
1. Cost-per-request table (API vs on-prem vs Cloud GPU).
2. **Break-even graph**: cumulative cost (y) vs usage volume in requests/month (x); the
   crossover where on-prem beats API. `matplotlib`, saved to `figures/`.
3. **Prompt/Context Caching nuance:** providers cache the static system-prompt prefix
   (PagedAttention-style) and bill those tokens far cheaper → for repetitive long-context
   prompts this **shifts the break-even point**. Model one "with caching" API line.
4. A reasoned recommendation: which scenarios favor API, which favor on-prem (incl. privacy
   / data-security, not just price).

---

## 7. Phased timeline (maps to assignment §11, total 6.5–11 h)

| Phase | Work | Est. wall time | Active time |
|-------|------|----------------|-------------|
| **1. Setup & downloads** | `uv` venv, install, HF auth, capture hardware facts (nvidia-smi, disk type), download tiny + 14B models, let AirLLM shard them. Mostly passive (tens-of-GB download + sharding = heavy disk op). | 1.5–3 h | ~15 min |
| **2. Runs & measurement** | Smoke-test (V0), baseline OOM (B0/B1), AirLLM A1–A3 + CPU C1, write the `benchmark()` + measurement scripts. First run is slow (RAM fills → mmap streams from SSD per layer, per quant). Be patient. | 3–5 h | 30–45 min |
| **3. Data processing, comparison & cost analysis** | Aggregate raw numbers → tables + graphs (TTFT/TPOT/throughput/RAM/VRAM), build Roofline + quality sweep, run the break-even/cost computation. | 1–1.5 h | ~30 min |
| **4. Report / README** | Assemble code + figures into an ordered academic report; explain numbers via theory (compute- vs memory-bound), critical pass, reproduce instructions. | 1–1.5 h | ~1 h |

> The active "cognitive" time is the lever: good experiment design + precise scripts +
> critical reading of results. The rest is the machine grinding through disk I/O.

---

## 8. Definition of done (acceptance)

- All §2 configs attempted; baseline failure captured with the actual error.
- All §3 metrics measured for A1–A3 (+ C1) and tabulated/graphed.
- Roofline + quantization-quality sweep produced.
- Break-even graph + cost tables with explicit assumptions.
- README embeds every table/figure/screenshot and re-run steps; HF token not committed.
- Each research question (RQ1–RQ6) answered with evidence + theory.
