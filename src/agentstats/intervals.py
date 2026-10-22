"""
Confidence intervals. Each returns an Interval that records its method, so a report can say what was done.

Which interval fits depends on how the eval set was built (see claims.py):
  - run_level (Student's t over complete runs): hand-picked sets; supports "on this eval set".
  - per_case (t over per-case mean pass rates): cases randomly sampled and independent; supports "on the
    population the cases came from". Miller 2024, arXiv 2411.00640, with t (N-1 df) instead of 1.96 so small
    sets aren't overconfident.
  - clustered (cluster-robust standard error over groups of related cases): cases sampled, but related within
    groups (same feature). Miller 2024 §2.3; CR0 sandwich with a G/(G-1) correction and t with G-1 df.
  - wilson: one run only, per-case binary outcomes.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from scipy import stats


@dataclass
class Interval:
    estimate: float
    low: float
    high: float
    method: str
    n: int
    se: float | None = None
    df: float | None = None
    note: str = ""
    z: float | None = None

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def to_json(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _t(conf: float, df: float) -> float:
    return float(stats.t.ppf(0.5 + conf / 2, df))


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def t_interval(values: Sequence[float], conf: float = 0.95, method: str = "t") -> Interval:
    """Mean of independent observations with a Student's t interval (df = n - 1)."""
    n = len(values)
    m = mean(values)
    if n < 2:
        return Interval(m, float("nan"), float("nan"), method, n, note="needs at least 2 observations")
    se = sd(values) / math.sqrt(n)
    half = _t(conf, n - 1) * se
    note = "zero spread: every observation equal; the interval understates uncertainty" if se == 0 else ""
    return Interval(m, m - half, m + half, method, n, se=se, df=n - 1, note=note)


def wilson(successes: int, n: int, conf: float = 0.95) -> Interval:
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), "wilson", 0)
    z = float(stats.norm.ppf(0.5 + conf / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), "wilson", n,
                    se=math.sqrt(p * (1 - p) / n))


def clustered(values: Sequence[float], clusters: Sequence[str], conf: float = 0.95) -> Interval:
    """Mean with a cluster-robust standard error: observations in the same cluster may be correlated."""
    n = len(values)
    groups: dict[str, list[float]] = {}
    for v, c in zip(values, clusters, strict=True):
        groups.setdefault(c, []).append(v)
    g = len(groups)
    m = mean(values)
    if g < 2:
        return Interval(m, float("nan"), float("nan"), "clustered", n, note="needs at least 2 clusters")
    meat = sum(sum(v - m for v in vs) ** 2 for vs in groups.values())
    se = math.sqrt(meat * g / (g - 1)) / n
    half = _t(conf, g - 1) * se
    return Interval(m, m - half, m + half, "clustered", n, se=se, df=g - 1, note=f"{g} clusters")


def bootstrap(units: Sequence, statistic: Callable[[Sequence], float], conf: float = 0.95,
              n_boot: int = 4000, seed: int = 0, method: str = "bootstrap") -> Interval:
    """Percentile bootstrap, resampling whole units (cases), so per-case correlation is kept."""
    rng = random.Random(seed)
    est = statistic(units)
    draws = []
    for _ in range(n_boot):
        sample = [units[rng.randrange(len(units))] for _ in units]
        v = statistic(sample)
        if not math.isnan(v):
            draws.append(v)
    if not draws:
        return Interval(est, float("nan"), float("nan"), method, len(units))
    draws.sort()
    lo = draws[int((1 - conf) / 2 * (len(draws) - 1))]
    hi = draws[int((1 + conf) / 2 * (len(draws) - 1))]
    return Interval(est, lo, hi, method, len(units), se=sd(draws))


def quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
