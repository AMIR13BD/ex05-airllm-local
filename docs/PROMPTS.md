# Prompt Engineering Log

Per the submission guidelines (§8.3), this logs the significant prompts that drove this
AI-assisted (Vibe Coding) project, with context and outcomes. Newest last.

| # | Context / Goal | Prompt (summary) | Outcome |
|---|----------------|------------------|---------|
| 1 | Kickoff | "Read the EX05 homework + lecture PDFs; create PRD/PLAN/TODO tailored to my hardware (i9-9900K / 32 GB / RTX 3070 8 GB / SSD); recommend a model too big for 8 GB VRAM but finishable in 6–11 h." | PRD/PLAN/TODO created; 4 scoping questions → Qwen2.5-14B, GPU+CPU, Israel ₪ cost basis, quant-quality+Roofline extension. |
| 2 | Storage reality | "Only ~47 GB free — is the 14B plan safe?" then "freed space, now ~193 GB." | Showed AirLLM keeps source + shard copies (~84 GB sweep); confirmed 14B safe at 193 GB; added preflight disk gate. |
| 3 | Environment | "Continue with environment setup; is Python 3.12 OK or do I need 3.11?" | 3.12 confirmed; native-Windows venv driven from WSL via interop; verified RTX 3070 8 GB + CUDA. |
| 4 | Pipeline bring-up | "Verify the tiny-model smoke test works before the 14B." | Hit and fixed 5 version incompatibilities (torch/transformers/optimum/tokenizers/numpy) + 3 AirLLM model quirks (index.json, single-file re-save, untied lm_head). fp16 & 4-bit smoke PASS. |
| 5 | GitHub | "Create a repo, commit & push step by step." | Installed gh (userspace), authed, created `ex05-airllm-local` (private), first commit pushed. |
| 6 | Submission compliance | "Read submission guidelines; do option 1 (start 14B download) then option 2 (benchmark harness); don't fake results; keep secrets/weights/caches out of git; logs to files." | Started 14B download/split in background; restructuring to mandated layout (docs/, config/, src/<pkg>, uv). |

## Working-style rules adopted (from the user)
- Never fabricate benchmark numbers; measure or report failure.
- Never commit `HF_TOKEN`, `.env`, model weights, HF cache, or AirLLM shards.
- Long runs must be resumable; save logs/results to files after each step.
- Keep large download/benchmark logs in files; summarize milestones/errors only.
