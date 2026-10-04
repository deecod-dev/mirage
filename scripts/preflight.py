#!/usr/bin/env python3
"""Server-only non-experiment preflight; it never downloads models or datasets."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from datetime import datetime, timezone

from mirage.artifacts import atomic_json
from mirage.config import load_config
from mirage.runtime import Runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    runtime = Runtime.from_environment(config.runtime)
    runtime.ensure_layout()
    runtime.assert_free_space()
    packages = ["torch", "transformers", "sentence_transformers", "datasets", "faiss", "sklearn"]
    report = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "platform": platform.platform(), "config_hash": config.hash,
              "runtime": config.runtime, "packages_present": {p: importlib.util.find_spec(p) is not None for p in packages}}
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_devices"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        report["cuda_names"] = [torch.cuda.get_device_name(i) for i in range(report["cuda_devices"])]
    except ImportError:
        report["cuda_available"] = False
        report["cuda_devices"] = 0
    atomic_json(runtime.results / "preflight" / f"preflight_{config.hash}.json", report)
    print(json.dumps(report, indent=2))
    if not all(report["packages_present"].values()) or not report["cuda_available"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
