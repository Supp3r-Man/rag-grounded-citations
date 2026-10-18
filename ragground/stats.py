"""Small statistics helpers: Wilson interval and a question-level bootstrap."""
from __future__ import annotations

import math
import random
from typing import Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ratio_ci(groups: Sequence[tuple[float, float]], n_boot: int = 1000,
             seed: int = 0) -> tuple[float, float, float]:
    """(estimate, lo, hi) for sum(num)/sum(den), resampling whole groups.
    Groups are questions, so claims from the same question stay together."""
    def ratio(gs):
        den = sum(d for _, d in gs)
        return sum(n for n, _ in gs) / den if den else 0.0

    est = ratio(groups)
    if not groups:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    vals = sorted(ratio([groups[rng.randrange(len(groups))] for _ in groups])
                  for _ in range(n_boot))
    return (est, vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1])
