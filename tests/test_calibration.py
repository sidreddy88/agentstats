"""
Calibration: do the statistics keep the error rates the README promises? Simulated under known truths, seeded.

These guard the package's core claim. A refactor that quietly changes a false-alarm rate (the first fixed-cases
test raised false alarms 6.8% of the time at a nominal 5%, before its small-sample correction) should fail here,
not in someone's CI weeks later. Bounds are loose enough to be stable across seeds and strict enough to catch a
real change (sims chosen so one standard error is about 0.5 points).
"""
from __future__ import annotations

import random

import pytest

from agentstats import sizing
from agentstats.comparisons import fixed_cases, overdispersion
from agentstats.intervals import wilson
from agentstats.records import from_dict


def _agent_like_rates(rng: random.Random, n: int) -> list[float]:
    """Per-case pass rates shaped like a real agent's: mostly near 1, some flaky, a few near 0."""
    return [rng.choice([0.97, 0.97, 0.97, 0.9, 0.75, 0.5, 0.1]) for _ in range(n)]


def _baseline_rows(rng: random.Random, n: int = 100, trials: int = 4):
    rows = []
    for i, p in enumerate(_agent_like_rates(rng, n)):
        for t in range(1, trials + 1):
            rows.append(from_dict({"case": f"c{i}", "run": t, "passed": rng.random() < p}))
    return rows


@pytest.mark.parametrize("p, n", [(0.1, 50), (0.5, 50), (0.9, 50), (0.97, 200)])
def test_wilson_interval_covers_the_true_rate(p, n):
    rng = random.Random(1)
    sims = 2000
    hits = 0
    for _ in range(sims):
        iv = wilson(sum(rng.random() < p for _ in range(n)), n, 0.95)
        hits += iv.low <= p <= iv.high
    # Wilson dips a little below nominal at some (p, n) because pass/fail is discrete (e.g. ~93% at p=0.5, n=50).
    assert hits / sims >= 0.92, f"coverage {hits / sims:.3f} at p={p}, n={n}"


def test_drop_gate_false_alarm_rate_under_no_change():
    """no_significant_drop at a 90% interval fails an unchanged PR about 5% of the time (one-sided).

    12,000 simulations (SE ~0.2 points). Measured when written: 5.1% with the small-sample correction, 6.7% with it
    turned off, so the 6% upper bound is what catches that bug coming back."""
    base = _baseline_rows(random.Random(2))
    fails = [sizing.simulate(base, trials=2, rule="drop", conf=0.90, shifts=(0.0,), sims=4000, seed=seed)
             ["rows"][0]["FAIL"] for seed in (3, 11, 29)]
    fail = sum(fails) / len(fails)
    assert 0.035 <= fail <= 0.060, f"false alarms {fail:.4f} (per seed {fails})"


def test_drop_gate_still_catches_a_real_drop():
    r = sizing.simulate(_baseline_rows(random.Random(2)), trials=2, rule="drop", conf=0.90, shifts=(1.5,),
                        sims=400, seed=4)
    assert r["rows"][0]["FAIL"] >= 0.8


def test_fixed_cases_interval_covers_the_true_difference():
    """Two arms with the same per-case rates: the 95% interval for the difference covers 0 about 95% of the time."""
    rng = random.Random(5)
    rates = _agent_like_rates(rng, 100)
    sims, hits = 1500, 0
    for _ in range(sims):
        a = {f"c{i}": (4, sum(rng.random() < p for _ in range(4))) for i, p in enumerate(rates)}
        b = {f"c{i}": (2, sum(rng.random() < p for _ in range(2))) for i, p in enumerate(rates)}
        iv = fixed_cases(a, b, 0.95)
        hits += iv.low <= 0 <= iv.high
    assert hits / sims >= 0.92, f"coverage {hits / sims:.3f}"


def test_shock_check_rarely_fires_without_a_shock():
    rng = random.Random(6)
    rates = _agent_like_rates(rng, 100)
    sims, fired = 600, 0
    for _ in range(sims):
        runs = [[rng.random() < p for p in rates] for _ in range(3)]
        run_rates = [sum(r) / len(r) for r in runs]
        counts = {f"c{i}": (3, sum(r[i] for r in runs)) for i in range(len(rates))}
        res = overdispersion(run_rates, counts)
        fired += res["p_value"] < 0.05
    assert fired / sims <= 0.08, f"shock check fired {fired / sims:.3f} with no shock"
