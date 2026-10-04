#!/usr/bin/env python3
"""Use this entrypoint for all server and Kaggle executions."""
from __future__ import annotations

import argparse
import json

from mirage.config import load_config
from mirage.server_runner import run_server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_server(load_config(args.config, seed=args.seed), resume=args.resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
