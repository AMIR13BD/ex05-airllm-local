# PRD — AirLLM Layer-Streaming Inference Pipeline (central mechanism)

Per submission-guidelines §2.3: a dedicated PRD for the project's central mechanism — running
a model whose weights (≈29 GB FP16) far exceed VRAM (8 GB) by streaming it layer-by-layer.

## 1. Mechanism & theory
A transformer is a stack of N decoder layers. Inference normally requires **all** weights
resident in VRAM. AirLLM instead keeps weights **on disk** (SafeTensors, one file per layer)
and, for each forward pass, **loads one layer → runs it → frees it → loads the next**:

```
for layer in [embed, L0, L1, …, L47, norm, lm_head]:
    weights = mmap_load(layer_shard)      # SafeTensors flat buffer -> mmap -> page-in
    move_to_device(weights, gpu)          # only THIS layer occupies VRAM
    hidden = layer(hidden)                # compute
    free(weights)                         # evict -> VRAM back to ~1 layer
```

This is the OS **virtual-memory / paging** analogy: disk is the backing store, `mmap` page
faults pull the needed layer, and the working set is one layer + KV-cache — so a 29 GB model
runs in <6 GB VRAM. The cost is **latency**: every generated token re-reads the whole model
from disk, moving the bottleneck from VRAM-bandwidth to **disk-bandwidth**.

Quantization (8-bit / 4-bit via bitsandbytes) shrinks each layer shard, reducing bytes
streamed per token → lower TTFT/TPOT, at a possible accuracy cost.

## 2. Requirements
- **Inputs:** HF repo id (or local path) of a causal-LM; quant level ∈ {fp16, 8bit, 4bit};
  device ∈ {cuda, cpu}; prompt; max_new_tokens.
- **Outputs:** generated text + a `RunRecord` (TTFT, TPOT/ITL, throughput, peak RAM/VRAM,
  runtime, GPU energy).
- **Functional:** must run a model that OOMs on direct load; peak VRAM must stay < 8 GB;
  results persisted as JSON; resumable splits (skip already-split layers).
- **Non-functional:** per-model shard isolation; no hardcoded paths/values (config-driven);
  files ≤150 LOC; reproducible via uv.

## 3. Constraints, alternatives & decisions
- **bitsandbytes 4/8-bit is CUDA-only** → the CPU comparison run uses **fp16** (`device="cpu"`).
- **AirLLM auto-selects cuda:0** → must pass `device` explicitly for CPU runs.
- **Splitter requires `model.safetensors.index.json`** and drops layers that span two source
  shards → small single-file models are re-saved as one file + hand-written index, with
  embeddings untied (explicit `lm_head`). The 14B is natively multi-shard → passes through.
- **Pinned 2024-era library stack** (torch 2.4.1, transformers 4.44.2, …) — newer releases
  break AirLLM 2.11 (see README §2).
- Alternative engines rejected: Ollama/llama.cpp (GGUF) hide the layer-streaming mechanism
  the assignment asks us to study; vLLM needs the model to fit in VRAM.

## 4. Success criteria & edge cases
- **Success:** baseline OOMs; AirLLM produces coherent tokens at all 3 quant levels with peak
  VRAM < 8 GB; quantization monotonically lowers latency. *(All met — see README §4.)*
- **Edge cases handled:** insufficient disk (preflight gate + AirLLM's own space check);
  shard-cache collision across models (per-model dirs); tied embeddings; layer-spanning
  shards; CPU/GPU device mismatch; cold first-token latency (expected, not a hang).
