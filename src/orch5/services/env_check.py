"""Environment verification: package versions, CUDA device, NVML telemetry."""
import importlib
import platform
import sys

_PACKAGES = ["torch", "transformers", "accelerate", "safetensors", "huggingface_hub",
             "bitsandbytes", "airllm", "psutil", "pynvml", "matplotlib", "pandas",
             "numpy", "datasets"]


def _ver(mod_name: str) -> str:
    try:
        return getattr(importlib.import_module(mod_name), "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__})"


def collect() -> dict:
    """Return a structured snapshot of the environment (testable, no printing)."""
    info: dict = {"python": sys.version.split()[0], "platform": platform.platform(),
                  "packages": {m: _ver(m) for m in _PACKAGES}}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": props.name,
                           "vram_gib": round(props.total_memory / 1024**3, 2),
                           "capability": f"{props.major}.{props.minor}",
                           "torch_cuda": torch.version.cuda}
    except Exception as exc:  # noqa: BLE001
        info["cuda_error"] = str(exc)
    return info


def report() -> dict:
    """Collect and pretty-print the environment snapshot; return it too."""
    info = collect()
    print("=" * 56)
    print(f"Python   : {info['python']}  ({info['platform']})")
    for mod, ver in info["packages"].items():
        print(f"{mod:16s}: {ver}")
    print("-" * 56)
    print(f"cuda_available: {info.get('cuda_available')}")
    if "gpu" in info:
        g = info["gpu"]
        print(f"  {g['name']} | {g['vram_gib']} GiB | cc {g['capability']} | cuda {g['torch_cuda']}")
    print("=" * 56)
    return info
