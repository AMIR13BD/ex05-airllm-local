"""Download + AirLLM-split Qwen2.5-14B-Instruct for fp16 / 8bit / 4bit.

Long, mostly-passive job (HF download ~29 GB + per-layer splits ~29/15/9 GB).
- Resumable: AirLLM skips layers already split; HF download resumes partial files.
- Logs to logs/prepare_models.log (full detail) AND stdout (milestones).
- Writes results/prepare_status.json after each step so progress is queryable.

Run (background):  .venv\\Scripts\\python.exe -u scripts\\prepare_models.py
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-14B-Instruct"
CACHE = Path(os.environ.get("AIRLLM_CACHE", REPO / ".airllm_cache"))
QUANTS = [("fp16", None), ("8bit", "8bit"), ("4bit", "4bit")]
MIN_FREE_GB = 90          # full sweep peaks ~84 GB on disk; refuse to start below this

LOG_FILE = REPO / "logs" / "prepare_models.log"
STATUS = REPO / "results" / "prepare_status.json"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(state: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> int:
    free_gb = shutil.disk_usage(CACHE.anchor or "C:\\").free / 1024**3
    log(f"free disk on cache drive: {free_gb:.1f} GB (need >= {MIN_FREE_GB})")
    if free_gb < MIN_FREE_GB:
        log("ABORT: insufficient free disk; not starting downloads.")
        return 2

    CACHE.mkdir(parents=True, exist_ok=True)
    state = {"model": MODEL, "cache": str(CACHE), "quants": {}}
    write_status(state)

    from airllm import AutoModel  # heavy import; prints bnb/cache notices

    for name, comp in QUANTS:
        t0 = time.time()
        log(f"=== preparing {MODEL} [{name}] (download source if needed + split) ===")
        state["quants"][name] = {"status": "running", "started": time.strftime("%H:%M:%S")}
        write_status(state)
        try:
            model = AutoModel.from_pretrained(
                MODEL, compression=comp, layer_shards_saving_path=str(CACHE))
            del model
        except Exception as exc:  # noqa: BLE001 — record and continue to next quant
            log(f"ERROR while preparing {name}: {type(exc).__name__}: {exc}")
            state["quants"][name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            write_status(state)
            continue
        mins = (time.time() - t0) / 60
        log(f"=== {name} split READY in {mins:.1f} min ===")
        state["quants"][name] = {"status": "ready", "minutes": round(mins, 1)}
        write_status(state)

    log("ALL REQUESTED SPLITS PROCESSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
