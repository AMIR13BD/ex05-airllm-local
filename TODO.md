# TODO — EX05 ordered checklist

Work top-to-bottom. **Start small, verify the pipeline, then scale.** Each box maps to
`PLAN.md`. Don't skip the pre-flight checks — they prevent the two classic failures
(no disk space; shards on a slow drive).

---

## ✅ Progress (updated 2026-06-21)

**DONE — env + pipeline verified on the tiny model (Qwen2.5-0.5B), fp16 AND 4-bit.**
TTFT ~2–3 s, ~0.4–0.5 tok/s — the expected slow AirLLM streaming profile.

**Working version stack (pinned — newer libs all broke AirLLM 2.11):**
`torch==2.4.1+cu124`, `transformers==4.44.2`, `tokenizers==0.19.1`,
`huggingface_hub==0.24.7`, `optimum==1.23.3`, `accelerate==0.34.2`,
`bitsandbytes==0.44.1`, `numpy==1.26.4`, `datasets==2.21.0`. (Why each: see `requirements.txt`.)

**AirLLM gotchas handled (in `src/utils_model.py`):** single-file small models need (a) an
`index.json` and (b) re-saving as ONE `model.safetensors` so no decoder layer spans a shard
boundary (AirLLM's splitter silently drops a spanning layer's weights → `meta` device error);
small Qwen models also need embeddings untied so an explicit `lm_head` shard exists. The 14B
is natively multi-shard and passes through untouched.

**Hardware confirmed:** RTX 3070 8192 MiB (nvidia-smi), WD Blue SN570 1 TB **NVMe**, 193 GB free.

**NEXT:** Phase 3+ on the 14B — download (~29 GB), baseline OOM, AirLLM fp16/8-bit/4-bit + CPU.

---

## Phase 0 — Pre-flight (do before anything else)
- [ ] Confirm GPU is really **8 GB**: run `nvidia-smi` (ignore the Windows WMI 4 GB bug). Note driver + CUDA version.
- [ ] Confirm storage type: **NVMe vs SATA** (`Get-PhysicalDisk` / CrystalDiskMark / `fio`). Record sequential read MB/s.
- [ ] **Free-space gate:** confirm free space (`Get-PSDrive C` / `dir`). Need **~85 GB** for the full 14B sweep; you have **~193 GB** ✓. Keep ~20 GB clear for the pagefile.
- [ ] Pick the **shard drive**: the fastest, roomiest drive; set `layer_shards_saving_path` there (native Windows; if WSL2 → native **ext4**, NOT `/mnt/c`).
- [ ] Record CPU/RAM facts (`wmic` / `lscpu`) for the report's hardware table.
- [ ] Decide run environment (native Windows vs WSL2 + CUDA) and stick to it.

## Phase 1 — Environment & downloads (≈1.5–3 h, mostly passive)
- [ ] Create isolated env: `uv venv --python 3.11` (avoid bleeding-edge Python).
- [ ] Install: `airllm transformers accelerate bitsandbytes torch(+CUDA) psutil pynvml matplotlib pandas numpy`.
- [ ] Set `HF_TOKEN` via **environment variable** (never hardcode / never commit).
- [ ] Init repo skeleton: `README.md`, `requirements.txt`/`pyproject.toml`, `src/ experiments/ results/ reports/ figures/`.
- [ ] Add `.gitignore` (exclude `.venv/`, model caches, tokens, large shards).
- [ ] Download **tiny** model (Qwen2.5-0.5B/1.5B-Instruct).
- [ ] **Re-confirm free space** right before the big download (≥ ~85 GB).
- [ ] Download **main** model `Qwen/Qwen2.5-14B-Instruct`; let AirLLM shard it (point `layer_shards_saving_path` at the fast drive). Budget ~84 GB peak (29 GB source + 29/15/9 GB shard sets).

## Phase 2 — Prove the pipeline (small + aggressive quant FIRST)
- [ ] **V0:** AirLLM on **GPU**, tiny model, `compression="4bit"`, `max_new_tokens≈16` — confirm it generates. Use `AutoModel` (Qwen → avoid `Class mismatch`).
- [ ] Verify shards landed on the intended drive; verify peak VRAM stays under 8 GB.
- [ ] Only once V0 works, proceed to the 14B runs.

## Phase 3 — Baseline (must fail / crawl — that's the evidence)
- [ ] **B0:** direct `transformers` 14B FP16 → `.to('cuda')`. Capture the **CUDA OOM** (screenshot + error text).
- [ ] **B1:** direct/CPU or Ollama 14B — observe unbearable slowness / swap thrash. Note RAM pressure.
- [ ] Write up the bottleneck (memory wall vs compute wall) and *how* you identified it (RQ1).

## Phase 4 — Build the measurement harness
- [ ] Write `src/benchmark.py`: wraps (engine, model, quant), runs the fixed prompt set, returns one record.
- [ ] Instrument **TTFT** (first-token timestamp via `TextIteratorStreamer`).
- [ ] Instrument **TPOT/ITL** (per-token deltas) + **throughput** (tok/s).
- [ ] Instrument **peak RAM** (`psutil` sampler thread) + **peak VRAM** (`torch.cuda.max_memory_allocated` / `pynvml`).
- [ ] Instrument **runtime** + **energy** (integrate GPU `power.draw`; estimate CPU power).
- [ ] Persist every raw number to `results/` (JSON/CSV, consistent units).

## Phase 5 — Main AirLLM runs (GPU) + CPU comparison
- [ ] **A1:** AirLLM GPU, 14B, FP16 (`compression=None`).
- [ ] **A2:** AirLLM GPU, 14B, **8-bit**.
- [ ] **A3:** AirLLM GPU, 14B, **4-bit**.
- [ ] **C1:** AirLLM **CPU**, 14B, 4-bit — the required CPU-vs-GPU comparison point.
- [ ] (Be patient: first decode tokens are slow as layers cold-load from SSD. Log per-layer timing.)
- [ ] Capture a **quality** sample per quant level (fixed prompts) for RQ3.

## Phase 6 — Extension: quantization-quality sweep + Roofline
- [ ] Compute a quality proxy per level (perplexity on a fixed small text + short rubric).
- [ ] Plot quality vs quant level; identify the accuracy **"red line."**
- [ ] Build the **Roofline** chart: place Prefill (compute-bound) & Decode (memory-bound), mark AirLLM on the **disk-I/O** slope.

## Phase 7 — Data processing & figures (≈1–1.5 h)
- [ ] Aggregate `results/` → comparison **tables** (TTFT, TPOT/ITL, throughput, peak RAM, peak VRAM, runtime, energy).
- [ ] Generate **bar/line graphs** per metric across quant levels (+ GPU vs CPU).
- [ ] Save all figures to `figures/`.

## Phase 8 — Cost analysis & break-even (₪ primary, USD shown)
- [ ] Config block: ₪0.60/kWh, CAPEX + amortization, measured Wh/request, API token prices (Claude + GPT-4o-mini), Cloud GPU $/h.
- [ ] Compute cost/request for **API**, **on-prem**, **Cloud GPU**.
- [ ] Plot **break-even graph** (cumulative cost vs requests/month) → find crossover.
- [ ] Add a **"with Prompt/Context Caching"** API line and explain how it shifts break-even.
- [ ] Write the reasoned recommendation (price **and** privacy/data-security).

## Phase 9 — Report / README (≈1–1.5 h)
- [ ] Write `README.md` as the deep-dive report: hardware spec + model justification; experiment description, phases, tools.
- [ ] Summary of findings: baseline vs AirLLM vs quantization; answer **RQ1–RQ6** with evidence + theory.
- [ ] **Concept-linkage** section: Prefill/Decode ↔ TTFT/TPOT, VRAM role, mmap/paging analogy, Roofline.
- [ ] Cost-analysis summary + recommendation.
- [ ] Embed **all** tables, graphs, screenshots inline; add clear **reproduce** instructions.
- [ ] Document the original extension.
- [ ] Final critical pass: are claims tied to theory? compute- vs memory-bound correctly attributed? fix gaps.

## Phase 10 — Submit
- [ ] Verify HF token not committed; repo is clean, consistent, easy to navigate.
- [ ] Push to GitHub; confirm README renders with all figures.

---

### Guardrails (assignment "Don't")
- ❌ Don't pick a model with no chance of running even under AirLLM.
- ❌ Don't store the HF token in plaintext / in code.
- ❌ Don't present raw numbers without analysis, graphs, and concept-linkage.
- ❌ Don't skip the economic analysis — it's a graded core part.
- ❌ Don't over-build — keep the experiment focused, clean, well-analyzed.
