"""Portable runtime paths and conservative safety checks."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Runtime:
    name: str
    root: Path
    cache: Path
    artifacts: Path
    results: Path
    use_4bit: bool

    @classmethod
    def from_environment(cls, name: str = "server") -> "Runtime":
        root = Path(os.environ.get("MIRAGE_ROOT", Path.cwd())).resolve()
        if name not in {"server", "kaggle_t4"}:
            raise ValueError(f"unsupported runtime: {name}")
        return cls(name, root, Path(os.environ.get("HF_HOME", root / ".hf_cache")),
                   root / "artifacts", root / "results", name == "kaggle_t4")

    def ensure_layout(self) -> None:
        for path in (self.cache, self.artifacts, self.results):
            path.mkdir(parents=True, exist_ok=True)

    def assert_free_space(self, minimum_gb: int = 20) -> None:
        free = shutil.disk_usage(self.root).free / 1024**3
        if free < minimum_gb:
            raise RuntimeError(f"Only {free:.1f}GB free at {self.root}; need {minimum_gb}GB")
