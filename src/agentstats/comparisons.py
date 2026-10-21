"""
Is B different from A? Each test returns an Interval for the difference B - A (positive = B better).

  - welch: complete runs as observations (hand-picked eval sets). Welch's t: unequal run-to-run variance.
  - paired: per-case differences in mean pass rate on the same cases (cases sampled and independent).
    Each case is its own control, so case difficulty cancels out.
  - paired_clustered: paired differences with a cluster-robust standard error.
  - mcnemar: one run each, binary outcomes on the same cases; exact test on the discordant pairs.

  - fixed_cases: the cases are fixed (a hand-picked set) and the noise is the agent's own randomness, measured
    from repeated trials of each case. Delta = mean over cases of (p_B,i - p_A,i). Variance under "nothing
    changed": each case's pooled rate p_i from all its trials, Var = sum p_i (1 - p_i) (1/k_A + 1/k_B) / N^2,
    with the small-sample correction n_i / (n_i - 1) (n_i = k_A + k_B): without it, a rate estimated from a few
    trials understates its variance and the test raises too many false alarms. This is the diagnosis gate v2
    test from agent-platform (app/evals/gate_v2.py) plus that correction. Cases that pass (or fail) every trial on
    both sides contribute nothing; the cases that sometimes pass carry the signal. With one trial per arm it
    reduces to McNemar's test (normal form). Valid when trials are independent across cases: a run-wide shock (a
    provider's bad hour) breaks that, which analysis.py checks for.

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


def fixed_cases(a_counts: dict[str, tuple[int, int]], b_counts: dict[str, tuple[int, int]], conf: float = 0.95,
                correction: bool = True) -> Interval:
    """a_counts / b_counts: case -> (trials, passes), on the same cases."""
    cases = sorted(set(a_counts) & set(b_counts))
    n = len(cases)
    t = v = 0.0
    for c in cases:
        ka, sa = a_counts[c]
        kb, sb = b_counts[c]
        t += sb / kb - sa / ka
        p = (sa + sb) / (ka + kb)
        var = p * (1 - p) * (1 / ka + 1 / kb)
        if correction:
            var *= (ka + kb) / (ka + kb - 1)
        v += var
    delta = t / n
    se = math.sqrt(v) / n
    if se == 0:
        return Interval(delta, delta, delta, "fixed_cases", n, se=0.0, z=0.0 if t == 0 else math.copysign(math.inf, t),
                        note="every case gave the same result on every trial; nothing varies")
    zc = float(stats.norm.ppf(0.5 + conf / 2))
    return Interval(delta, delta - zc * se, delta + zc * se, "fixed_cases", n, se=se, z=delta / se)


def overdispersion(run_rates: Sequence[float], counts: dict[str, tuple[int, int]]) -> dict | None:
    """Do whole runs vary more than per-case randomness predicts? Predicted variance of a run's pass rate:
    sum over cases of p_i (1 - p_i) / N^2 (per-case rates from all trials, small-sample corrected). Observed: the
    sample variance of the run pass rates. (K - 1) s^2 / sigma^2 ~ chi-square(K - 1) if runs only differ by
    per-case noise; a small p-value means run-wide shocks, and the fixed-cases test is too optimistic."""
    k = len(run_rates)
    if k < 2:
        return None
    n = len(counts)
    pred = sum((s / t) * (1 - s / t) * (t / (t - 1)) for t, s in counts.values() if t >= 2) / n ** 2
    obs = sd(run_rates) ** 2
    if pred == 0:
        return {"ratio": math.inf if obs > 0 else 1.0, "p_value": 0.0 if obs > 0 else 1.0, "runs": k}
    stat = (k - 1) * obs / pred
    return {"ratio": obs / pred, "p_value": float(stats.chi2.sf(stat, k - 1)), "runs": k}


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
