"""CLI entry point. Thin argument parsing over the SDK (no business logic here)."""
import argparse
import json

from orch5.sdk import Ex05SDK
from orch5.shared import config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orch5", description="EX05 AirLLM benchmarking CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="verify the environment (versions, CUDA, NVML)")

    b = sub.add_parser("bench", help="run one AirLLM benchmark configuration")
    b.add_argument("--model", default="main", help="config key (main/tiny) or HF repo id")
    b.add_argument("--quant", default="4bit", choices=list(config.QUANT_LEVELS))
    b.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    b.add_argument("--prompt", default="medium_reasoning", help="prompt key or literal text")
    b.add_argument("--max-new-tokens", type=int, default=None)

    c = sub.add_parser("cost", help="per-request cost table (NIS)")
    c.add_argument("--in-tok", type=int, required=True)
    c.add_argument("--out-tok", type=int, required=True)
    c.add_argument("--energy-wh", type=float, required=True)
    c.add_argument("--runtime-s", type=float, required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sdk = Ex05SDK()
    if args.cmd == "env":
        sdk.check_env()
    elif args.cmd == "bench":
        rec = sdk.benchmark(args.model, args.quant, args.device, args.prompt,
                            args.max_new_tokens)
        print(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False))
    elif args.cmd == "cost":
        print(json.dumps(sdk.cost_table(args.in_tok, args.out_tok, args.energy_wh,
                                        args.runtime_s), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
