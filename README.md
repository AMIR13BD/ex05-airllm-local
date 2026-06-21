# EX05 — Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

> Deep-dive technical report. Running **Qwen2.5-14B-Instruct** (too big for 8 GB VRAM) on an
> **i9-9900K / 32 GB / RTX 3070 (8 GB) / NVMe** box via **AirLLM** layer streaming +
> quantization, with full performance benchmarking and an on-prem-vs-API cost analysis.

This README **is** the report — every table, graph, and screenshot is embedded inline.
See [`docs/PRD.md`](docs/PRD.md), [`docs/PLAN.md`](docs/PLAN.md), [`docs/TODO.md`](docs/TODO.md)
for requirements, plan, and progress, and [`docs/PROMPTS.md`](docs/PROMPTS.md) for the prompt log.

## 1. Hardware & model justification
- CPU: Intel i9-9900K (8C/16T) · RAM: 32 GB · GPU: RTX 3070 **8 GB** (confirmed via
  `nvidia-smi`; Windows WMI 4 GB report is a known bug) · Storage: WD Blue SN570 1 TB **NVMe**.
- Why Qwen2.5-14B: FP16 ≈ 29 GB ⇒ cannot fit 8 GB VRAM, stresses 32 GB RAM; Qwen family is
  AutoModel-friendly for AirLLM. _(Full justification: see PRD/PLAN.)_

## 2. Experiment & tooling
- Environment: Python 3.12, isolated venv. Stack: AirLLM, transformers, accelerate,
  bitsandbytes, torch (CUDA). Measurement: psutil (RAM), pynvml (VRAM + power).
- Configs: baseline direct-load (expected OOM) → AirLLM FP16/8-bit/4-bit (GPU) → CPU run.
- _(TODO: fill in once runs complete.)_

## 3. Results — baseline vs AirLLM vs quantization
_(TODO: comparative tables + graphs: TTFT, TPOT/ITL, throughput, peak RAM/VRAM, runtime, energy.)_

## 4. Quantization-quality sweep + Roofline (extension)
_(TODO.)_

## 5. On-prem vs API cost analysis + break-even
_(TODO: break-even graph, assumptions, Prompt/Context Caching nuance, recommendation.)_

## 6. Concept linkage
_(TODO: Prefill/Decode ↔ TTFT/TPOT, VRAM role, mmap/paging analogy, Roofline, answers to RQ1–RQ6.)_

## 7. Reproduce
```bash
# 1) create env (Windows PowerShell, Python 3.12)
py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
# 2) install
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt   # pinned stack — newer libs break AirLLM 2.11
# 3) verify, then smoke-test, then scale
python src/check_env.py
python src/smoke_test.py
python src/benchmark.py --model 14b --quant 4bit --device cuda
```
Set `HF_TOKEN` via environment variable (never commit it).
