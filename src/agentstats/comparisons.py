"""
Is B different from A? Each test returns an Interval for the difference B - A (positive = B better).

  - welch: complete runs as observations (hand-picked eval sets). Welch's t: unequal run-to-run variance.
  - paired: per-case differences in mean pass rate on the same cases (cases sampled and independent).
    Each case is its own control, so case difficulty cancels out.
  - paired_clustered: paired differences with a cluster-robust standard error.
  - mcnemar: one run each, binary outcomes on the same cases; exact test on the discordant pairs.

Smallest detectable difference: about (t_{1-a/2} + z_{power}) x SE; with a = 0.05 and power 0.8, ~2.8 x SE.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from scipy import stats

from agentstats.intervals import Interval, clustered, mean, sd, t_interval


def welch(a_runs: Sequence[float], b_runs: Sequence[float], conf: float = 0.95) -> Interval:
    ka, kb = len(a_runs), len(b_runs)
    delta = mean(b_runs) - mean(a_runs)
    if ka < 2 or kb < 2:
        return Interval(delta, float("nan"), float("nan"), "welch", ka + kb, note="needs at least 2 runs per arm")
    va, vb = sd(a_runs) ** 2 / ka, sd(b_runs) ** 2 / kb
    se = math.sqrt(va + vb)
    if se == 0:
        return Interval(delta, delta, delta, "welch", ka + kb, se=0.0,
                        note="no run-to-run variation in either arm; the interval understates uncertainty")
    df = (va + vb) ** 2 / ((va ** 2 / (ka - 1) if ka > 1 else 0) + (vb ** 2 / (kb - 1) if kb > 1 else 0))
    half = float(stats.t.ppf(0.5 + conf / 2, df)) * se
    return Interval(delta, delta - half, delta + half, "welch", ka + kb, se=se, df=df)


def paired(diffs: Sequence[float], conf: float = 0.95) -> Interval:
    iv = t_interval(diffs, conf, method="paired")
    nonzero = sum(1 for d in diffs if d != 0)
    if nonzero and nonzero < 10:
        iv.note = (iv.note + "; " if iv.note else "") + f"only {nonzero} cases differ; the interval may be optimistic"
    return iv


def paired_clustered(diffs: Sequence[float], clusters: Sequence[str], conf: float = 0.95) -> Interval:
    iv = clustered(diffs, clusters, conf)
    iv.method = "paired_clustered"
    return iv


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (same cases, one run each)."""
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return {"b_only": b_only, "a_only": a_only, "discordant": n, "p_value": p}


def smallest_detectable(iv: Interval, conf: float = 0.95, power: float = 0.8) -> float | None:
    if iv.se is None or math.isnan(iv.se) or iv.se == 0:
        return None
    crit = float(stats.t.ppf(0.5 + conf / 2, iv.df)) if iv.df else float(stats.norm.ppf(0.5 + conf / 2))
    return (crit + float(stats.norm.ppf(power))) * iv.se
