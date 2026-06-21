"""Phase-1 environment check. Verifies the whole stack BEFORE any model download.

Run:  python src/check_env.py
"""
import importlib
import platform
import sys


def _ver(mod_name, attr="__version__"):
    try:
        m = importlib.import_module(mod_name)
        return getattr(m, attr, "installed (no __version__)")
    except Exception as e:  # noqa: BLE001
        return f"MISSING ({type(e).__name__}: {e})"


def main():
    print("=" * 60)
    print("EX05 environment check")
    print("=" * 60)
    print(f"Python   : {sys.version.split()[0]}  ({platform.platform()})")

    # --- core packages ---
    for mod in [
        "torch", "transformers", "accelerate", "safetensors",
        "huggingface_hub", "bitsandbytes", "airllm",
        "psutil", "pynvml", "matplotlib", "pandas", "numpy", "datasets",
    ]:
        print(f"{mod:16s}: {_ver(mod)}")

    # --- torch / CUDA detail ---
    try:
        import torch
        print("-" * 60)
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            p = torch.cuda.get_device_properties(i)
            print(f"  device        : {p.name}")
            print(f"  capability    : {p.major}.{p.minor}")
            print(f"  total VRAM    : {p.total_memory / 1024**3:.2f} GiB")
            print(f"  torch CUDA    : {torch.version.cuda}")
    except Exception as e:  # noqa: BLE001
        print(f"torch CUDA check failed: {e}")

    # --- pynvml: confirms VRAM + power telemetry works (needed for metrics) ---
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        print("-" * 60)
        print(f"NVML VRAM used : {mem.used / 1024**2:.0f} / {mem.total / 1024**2:.0f} MiB")
        print(f"NVML power draw: {power:.1f} W")
        pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001
        print(f"NVML check failed (VRAM/power telemetry): {e}")

    print("=" * 60)
    print("If torch.cuda.is_available() is True and no package is MISSING,")
    print("the environment is ready for the tiny smoke test.")
    print("=" * 60)


if __name__ == "__main__":
    main()
