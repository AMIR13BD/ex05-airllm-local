# PRD — EX05: Running a Massive LLM Locally (AirLLM, Quantization & Benchmarking)

**Course:** L08 — On-Premises LLM Deployment (Dr. Yoram Segal) · **Assignment:** EX05 · **Version:** 1.0
**Author:** (you) · **Target hardware:** Intel i9-9900K · 32 GB RAM · RTX 3070 (8 GB) · ~1 TB SSD (~193 GB free)

---

## 1. Purpose & one-line summary

Prove, in a practical and justified way, that I can take an LLM that is **too big to run
directly** on my hardware, make it run anyway using **AirLLM (layer-by-layer streaming) +
quantization**, **measure** it with industry-standard performance metrics, and **analyze**
the results both technically (where is the bottleneck?) and economically (on-prem vs API).

The graded artifact is a **deep-dive technical report** (in a GitHub repo with full code),
**not** a model whose answer quality is good. Understanding *why* the numbers look the way
they do is the whole point.

> Core narrative for *my* machine: a 14B model in FP16 needs ~29 GB of weights. My GPU has
> only **8 GB VRAM**, so a direct GPU load **cannot fit** and must fail (CUDA OOM). AirLLM
> sidesteps this by keeping the model on disk and streaming one transformer layer at a time
> through the 8 GB VRAM — trading **latency** (disk I/O per token) for the **ability to run
> at all**.

---

## 2. Goals (what "done" looks like)

| # | Goal | Done when… |
|---|------|-----------|
| G1 | Document hardware + justify model choice | Exact CPU/GPU/RAM/VRAM/storage spec recorded; reasoned argument for picking a model **too big** for direct execution but feasible in the time budget. |
| G2 | Establish a failing/crawling **baseline** | A documented direct-run attempt (Ollama or HF `transformers` → `.to('cuda')`) that OOMs or is unbearably slow. Bottleneck named (memory vs compute). |
| G3 | Make the model run via **AirLLM + quantization** | Same model produces tokens using AirLLM with FP16 / 8-bit / 4-bit compression; resource reallocation explained. |
| G4 | **Measure** standardized metrics | TTFT, ITL/TPOT, throughput (tok/s), peak RAM, peak VRAM, total runtime, estimated energy/power — captured as raw numbers, tabulated and graphed per quant level. |
| G5 | **Economic** on-prem vs API analysis | Cost-per-request curves + a **break-even graph** (on-prem vs API, with optional Cloud GPU line); all assumptions explicit and reproducible. |
| G6 | **Concept linkage** | Results tied to Prefill (compute-bound) vs Decode (memory-bound), VRAM role, mmap/virtual-memory & paging analogy, Roofline. |
| G7 | ≥1 **original extension** | A **quantization-quality sweep + Roofline plot** delivered (see §6). |
| G8 | Reproducible deliverable | GitHub repo: code, scripts, README with all tables/graphs/screenshots embedded, clear re-run instructions. |

---

## 3. Deliverables (graded outputs)

1. **Full code** of the experiment + run/measurement scripts.
2. **Deep-dive technical report** (must live as a Markdown/README file in the repo) covering:
   hardware documentation, baseline, AirLLM + quantization findings.
3. **README.md** of the project — the report itself, with **all** graphs, tables and
   screenshots embedded inline (must be readable by an external reader).
4. **Comparative tables + graphs** of the performance metrics.
5. **Cost-analysis section**: on-prem vs API (+ optional Cloud GPU), including the
   **break-even graph** and every assumption.
6. **Concept-linkage analysis** connecting numbers → inference theory.
7. **Documentation of the original extension(s).**

Recommended repo layout: `README.md`, `requirements.txt`/`pyproject.toml`, `src/`,
`experiments/`, `results/`, `reports/`, `figures/`.

---

## 4. Research questions the report MUST answer

(From the assignment §4 — these are graded explicitly.)

- **RQ1 — Bottleneck:** What bottleneck blocked the direct run — memory (VRAM/RAM) or
  compute? How did I identify it?
- **RQ2 — AirLLM mechanism:** How does AirLLM change resource allocation, and how does it
  relate to the OS virtual-memory / paging mechanism?
- **RQ3 — Quantization impact:** Effect of quantization on memory footprint, speed, and
  output quality. Where is the "red line" of acceptable accuracy?
- **RQ4 — Prefill vs Decode:** How do Prefill and Decode show up in *my* measurements? How
  do they map to TTFT (compute load) vs TPOT/ITL (memory-movement load)?
- **RQ5 — The price of local:** What is the latency/throughput cost of running a big model
  on modest hardware?
- **RQ6 — On-prem vs API:** When is local economically worthwhile, and when is an external
  API preferable?

---

## 5. Constraints & assumptions

**Hardware (fixed):**
- CPU: Intel Core i9-9900K, 8 cores / 16 threads, 3.6 GHz base (Z390 platform).
- RAM: 32 GB.
- GPU: NVIDIA RTX 3070, **8 GB** GDDR6 (Ampere, CC 8.6), ~448 GB/s bandwidth.
  - ⚠️ Windows WMI may misreport VRAM as 4 GB (32-bit `AdapterRAM` overflow). **Confirm 8 GB
    with `nvidia-smi`** and state this in the report.
- Storage: ~1 TB SSD, **~193 GB free** (confirmed after cleanup) — **must confirm NVMe vs
  SATA** (`wmic diskdrive get model,...` / `Get-PhysicalDisk`). This is the single biggest
  performance factor for AirLLM (NVMe ~3,500 MB/s vs SATA ~550 MB/s ⇒ ~6× difference in
  per-token decode time).
- **Storage budget:** the full 14B FP16/8-bit/4-bit sweep peaks at **~84 GB** on disk
  (29 GB HF source + 29/15/9 GB shard sets + ~2 GB tiny model). 193 GB free is comfortable;
  keep ~20 GB clear for the Windows pagefile. **Run a preflight free-space check before any
  download/shard** and point `layer_shards_saving_path` at the fastest, roomiest drive.

**Software / process constraints (assignment "Do/Don't"):**
- Use an isolated virtual environment (`uv` recommended). Do **not** use the newest Python
  release (many libs lag); target a well-supported version (e.g., 3.10/3.11).
- Start **small** (tiny model + aggressive quantization) to verify the pipeline before
  scaling to the full model. Use a low `max_new_tokens` for early checks.
- Ensure enough free disk **before** downloading large models.
- Point AirLLM `layer_shards_saving_path` to a **fast, dedicated drive** — not the OS drive.
- For Qwen-family models, use AirLLM's generic `AutoModel` (avoids `Class mismatch`).
- Measure consistently; **save every raw number** for the graphs.
- **Never** commit the Hugging Face token in plaintext.
- **Don't** present raw numbers without analysis/graphs/concept-linkage; **don't** skip the
  cost analysis (it is a core part of the grade); keep the experiment focused and clean.

**Decisions locked for this project:**
- **Main model:** `Qwen/Qwen2.5-14B-Instruct` (FP16 ≈ 29 GB ⇒ won't fit 8 GB VRAM; stresses
  32 GB RAM). Pipeline-verification model: a tiny Qwen2.5 (0.5B/1.5B) with 4-bit.
- **AirLLM device:** GPU-primary on the RTX 3070 + one CPU run for the required CPU-vs-GPU
  comparison.
- **Quantization levels:** FP16 (no compression) / 8-bit / 4-bit (AirLLM `compression`).
- **Cost basis:** Israel — ₪ primary (≈ ₪0.60/kWh), USD shown alongside; include a Cloud
  GPU rental line in the break-even graph.
- **Original extension:** quantization-quality sweep (perplexity/quality per level) +
  compute-vs-memory **Roofline** plot.

**Time budget:** 6.5–11 h end-to-end (mostly passive download + physical compute time);
~2–3 h of active work. See PLAN.md for the phase breakdown.

---

## 6. Original extension (required, §5.7)

**Quantization-quality sweep + Roofline plot.**
- For each quant level (FP16 / 8-bit / 4-bit): measure a **quality** signal (perplexity on a
  small fixed text sample and/or a rubric on fixed prompts) alongside the speed/memory
  metrics, to locate the accuracy "red line."
- Build a **Roofline-style** chart placing Prefill (compute-bound, high arithmetic
  intensity) and Decode (memory-bound, low intensity) on the i9/3070's roofline, and mark
  where AirLLM actually sits (**disk-I/O bound**, far left) — a visual proof of which
  resource limits the system in each regime.

---

## 7. Out of scope

- Production serving / multi-user concurrency, distributed/disaggregated serving (discussed
  only as theory for concept-linkage).
- Achieving competitive answer quality or "winning" benchmark numbers.
- Turning this into a finished product — it stays a focused, well-analyzed experiment.

---

## 8. Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| SATA (not NVMe) SSD | AirLLM decode painfully slow | Confirm early; if SATA, reduce `max_new_tokens`, prefer 4-bit, consider 8B fallback. |
| Running AirLLM inside WSL2 with shards on `/mnt/c` | drvfs I/O kills performance | Keep shards on native Linux ext4 (or run natively on Windows). |
| 14B sweep is ~84 GB on disk (source + 3 shard sets) | Disk overrun | ~193 GB free is enough; still run the preflight gate, download once, reuse shards. |
| Decode hangs for minutes (cold layer loads) | Looks like a crash | Expected — be patient; low token counts; log per-layer timing. |
| HF token leak / model license | Compliance | Use env var; check Qwen2.5 license; never hardcode. |
