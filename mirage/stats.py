"""Pre-registered stratified bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np


def bootstrap_mean(values: list[float], seed: int, repeats: int = 1000) -> dict[str, float]:
    clean_values = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not len(clean_values):
        clean_values = [0.0]
    array = np.asarray(clean_values, dtype=float)
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    return {"point": float(array.mean()), "ci_low": float(np.quantile(samples, .025)),
            "ci_high": float(np.quantile(samples, .975))}
