#!/usr/bin/env python3
"""Run a fully configured MIRAGE experiment on the server or Kaggle."""
from __future__ import annotations

import argparse
import json

from mirage.config import load_config
from mirage.runner import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config, seed=args.seed), resume=args.resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
