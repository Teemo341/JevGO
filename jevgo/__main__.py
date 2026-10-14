"""JevGO frozen-model experiments and model protocol checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

from jevgo.core import JevGOError
from jevgo.factory import MODEL_KEYS, check_model, run_from_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    default_assets = Path(os.environ.get("JEVGO_ASSETS_ROOT", str(Path(__file__).resolve().parents[1])))
    run = commands.add_parser("run", help="Run B0/B1/B2/B3 from a YAML configuration")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--assets-root", type=Path, default=default_assets)
    run.add_argument("--method", choices=("B0", "B1", "B2", "B3"))
    run.add_argument("--model", choices=MODEL_KEYS)
    run.add_argument("--base-url")
    check = commands.add_parser("model-check", help="Call model protocols and save requests, responses and costs")
    check.add_argument("--model", choices=MODEL_KEYS, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--assets-root", type=Path, default=default_assets)
    check.add_argument("--base-url")
    check.add_argument("--max-model-len", type=int, default=4096)
    check.add_argument("--image", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Configuration must be a YAML object")
            if args.method:
                payload.setdefault("run", {})["method"] = args.method
            if args.model:
                payload.setdefault("model", {})["backend"] = args.model
            if args.base_url:
                payload.setdefault("model", {})["base_url"] = args.base_url
            result = run_from_dict(payload, assets_root=args.assets_root, output_dir=args.output)
            print(result.model_dump_json())
            return 1 if result.error else 0
        config = {"backend": args.model}
        if args.model != "deterministic":
            config["max_model_len"] = args.max_model_len
        if args.base_url:
            config["base_url"] = args.base_url
        result = check_model(config, assets_root=args.assets_root, output_dir=args.output, image_path=args.image)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ok"] else 1
    except (OSError, ValueError, TypeError, KeyError, JevGOError, yaml.YAMLError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
