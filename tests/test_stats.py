"""Every interval and test checked against an independent reference (scipy / statsmodels) or exact values."""
import math
import random

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.proportion import proportion_confint

from agentstats import comparisons, intervals, reliability

rng = random.Random(7)
RUNS_A = [0.78, 0.84, 0.81, 0.86, 0.80]
RUNS_B = [0.85, 0.88, 0.83, 0.90]


@pytest.mark.parametrize("c,n", [(0, 10), (3, 10), (48, 100), (100, 100), (7, 13)])
def test_wilson_matches_statsmodels(c, n):
    iv = intervals.wilson(c, n)
    lo, hi = proportion_confint(c, n, alpha=0.05, method="wilson")
    assert iv.low == pytest.approx(lo, abs=1e-9) and iv.high == pytest.approx(hi, abs=1e-9)


def test_t_interval_matches_scipy():
    iv = intervals.t_interval(RUNS_A)
    lo, hi = stats.t.interval(0.95, len(RUNS_A) - 1, loc=np.mean(RUNS_A), scale=stats.sem(RUNS_A))
    assert (iv.low, iv.high) == pytest.approx((lo, hi))


def test_t_interval_needs_two():
    assert math.isnan(intervals.t_interval([0.5]).low)


def test_welch_matches_scipy():
    iv = comparisons.welch(RUNS_A, RUNS_B)
    res = stats.ttest_ind(RUNS_B, RUNS_A, equal_var=False)
    ci = res.confidence_interval(0.95)
    assert iv.estimate == pytest.approx(np.mean(RUNS_B) - np.mean(RUNS_A))
    assert (iv.low, iv.high) == pytest.approx((ci.low, ci.high))
    assert iv.df == pytest.approx(res.df)


def test_paired_matches_scipy():
    a = [rng.random() for _ in range(40)]
    b = [x + rng.gauss(0.05, 0.1) for x in a]
    iv = comparisons.paired([y - x for x, y in zip(a, b, strict=True)])
    ci = stats.ttest_rel(b, a).confidence_interval(0.95)
    assert (iv.low, iv.high) == pytest.approx((ci.low, ci.high))


def test_mcnemar_matches_statsmodels():
    a = [True] * 30 + [False] * 20 + [True] * 6 + [False] * 7
    b = [True] * 30 + [False] * 20 + [False] * 6 + [True] * 7
    ours = comparisons.mcnemar(a, b)
    table = [[30, 6], [7, 20]]
    assert ours["b_only"] == 7 and ours["a_only"] == 6
    assert ours["p_value"] == pytest.approx(sm_mcnemar(table, exact=True).pvalue)


def test_clustered_matches_statsmodels_cr0_with_group_correction():
    values = [rng.random() for _ in range(60)]
    groups = [f"g{i % 7}" for i in range(60)]
    iv = intervals.clustered(values, groups)
    codes = np.array([int(g[1:]) for g in groups])
    fit = sm.OLS(np.array(values), np.ones(len(values))).fit(
        cov_type="cluster", cov_kwds={"groups": codes, "use_correction": False})
    g = 7
    assert iv.se == pytest.approx(fit.bse[0] * math.sqrt(g / (g - 1)))
    assert iv.df == g - 1


def test_clustered_wider_than_naive_when_groups_correlate():
    values, groups = [], []
    for g in range(10):
        level = rng.random()
        for _ in range(8):
            values.append(level + rng.gauss(0, 0.02))
            groups.append(f"g{g}")
    assert intervals.clustered(values, groups).se > intervals.t_interval(values).se * 2


def test_pass_at_k_and_hat_k_exact():
    assert reliability.pass_at_k(5, 2, 1) == pytest.approx(0.4)
    assert reliability.pass_hat_k(5, 2, 1) == pytest.approx(0.4)
    assert reliability.pass_at_k(5, 2, 2) == pytest.approx(1 - 3 / 10)
    assert reliability.pass_hat_k(5, 2, 2) == pytest.approx(1 / 10)
    assert reliability.pass_hat_k(3, 3, 3) == 1.0
    with pytest.raises(ValueError):
        reliability.pass_at_k(2, 1, 3)


def test_reliability_summary_classifies_cases():
    s = reliability.summarize({"a": (3, 3), "b": (3, 0), "c": (3, 2)})
    assert s["k_max"] == 3
    assert s["cases"] == {"always": ["a"], "never": ["b"], "flaky": ["c"]}
    assert s["pass_at"][1]["estimate"] == pytest.approx((1 + 0 + 2 / 3) / 3)
    assert s["pass_hat"][3]["estimate"] == pytest.approx(1 / 3)


def test_smallest_detectable_is_about_2_8_se_for_large_df():
    iv = intervals.Interval(0.0, -0.1, 0.1, "paired", 1000, se=0.01, df=10_000)
    assert comparisons.smallest_detectable(iv) == pytest.approx(0.028, abs=5e-4)


def test_bootstrap_covers_estimate():
    units = list(range(50))
    iv = intervals.bootstrap(units, lambda us: sum(us) / len(us))
    assert iv.low < 24.5 < iv.high


def _gate_v2_z(pr, base):
    """agent-platform's diagnosis gate v2 test (app/evals/gate_v2.py paired_test), verbatim maths."""
    t = v = 0.0
    for c in pr:
        a, k = sum(pr[c]), len(pr[c])
        b, m = sum(base[c]), len(base[c])
        t += a / k - b / m
        p = (a + b) / (k + m)
        v += p * (1 - p) * (1 / k + 1 / m)
    return t / math.sqrt(v)


def _counts(v):
    return {c: (len(x), sum(x)) for c, x in v.items()}


def test_fixed_cases_without_correction_reproduces_gate_v2():
    r = random.Random(5)
    base = {f"c{i}": [r.random() < 0.6 for _ in range(4)] for i in range(300)}
    pr = {f"c{i}": [r.random() < 0.55 for _ in range(2)] for i in range(300)}
    iv = comparisons.fixed_cases(_counts(base), _counts(pr), correction=False)
    assert iv.z == pytest.approx(_gate_v2_z(pr, base))


def test_fixed_cases_one_trial_each_is_mcnemar_normal_form():
    a = {f"c{i}": [x] for i, x in enumerate([True] * 30 + [False] * 20 + [True] * 6 + [False] * 7)}
    b = {f"c{i}": [x] for i, x in enumerate([True] * 30 + [False] * 20 + [False] * 6 + [True] * 7)}
    iv = comparisons.fixed_cases(_counts(a), _counts(b))
    assert iv.z == pytest.approx((7 - 6) / math.sqrt(13))


def _false_alarm(correction, sims=1500, n=400, k=2, m=4, seed=11):
    """No change: both sides drawn from each case's own rate (most cases near 0 or 1, as in real agent evals)."""
    r = random.Random(seed)
    rates = [r.choice([0.02, 0.98, 0.98, 0.5, 0.2, 0.8, 0.98, 0.02]) for _ in range(n)]
    hits = 0
    for _ in range(sims):
        base = {f"c{i}": (m, sum(r.random() < p for _ in range(m))) for i, p in enumerate(rates)}
        pr = {f"c{i}": (k, sum(r.random() < p for _ in range(k))) for i, p in enumerate(rates)}
        z = comparisons.fixed_cases(base, pr, correction=correction).z
        hits += z is not None and z < -1.645
    return hits / sims


def test_small_sample_correction_brings_false_alarms_to_nominal():
    corrected, raw = _false_alarm(True), _false_alarm(False)
    assert 0.03 <= corrected <= 0.07
    assert raw > corrected


# ---- few failures: the normal approximation breaks; Newcombe's interval takes over ----------------------------

def test_no_failures_on_either_side_is_not_a_zero_width_interval():
    from agentstats.comparisons import fixed_cases
    allpass = {f"c{i}": (3, 3) for i in range(40)}
    iv = fixed_cases(allpass, allpass)
    assert iv.low < 0 < iv.high and iv.high - iv.low > 0.05                 # 120 vs 120 passes: about +-3.1 points
    assert iv.low == pytest.approx(-0.0310, abs=0.001) and "newcombe" in iv.method and "0 failures" in iv.note


def test_newcombe_matches_published_values():
    # Newcombe 1998, Table II example (56/70 vs 48/80): difference 0.2, method 10 interval 0.0524 to 0.3339
    from agentstats.intervals import newcombe
    iv = newcombe(48, 80, 56, 70)
    assert iv.estimate == pytest.approx(0.2) and iv.low == pytest.approx(0.0524, abs=5e-4)
    assert iv.high == pytest.approx(0.3339, abs=5e-4)


def test_enough_failures_keeps_the_paired_fixed_cases_test():
    from agentstats.comparisons import fixed_cases
    a = {f"c{i}": (3, 3 if i % 3 else 1) for i in range(40)}               # 28 failures: plenty
    assert fixed_cases(a, a).method == "fixed_cases"


def test_coverage_when_failures_are_rare():
    """True per-case pass rate 0.99 on both sides, 40 cases x 3 trials: most runs see 0-3 failures. The interval must
    still cover the true difference (0) at about the nominal rate; the old zero-width interval didn't."""
    import random

    from agentstats.comparisons import fixed_cases
    rng = random.Random(0)
    covered = trials = 0
    for _ in range(2000):
        def arm():
            return {f"c{i}": (3, sum(rng.random() < 0.99 for _ in range(3))) for i in range(40)}
        iv = fixed_cases(arm(), arm())
        trials += 1
        covered += iv.low <= 0 <= iv.high
    assert covered / trials >= 0.95


def test_rare_failures_with_a_real_gap_keep_nominal_coverage():
    """Base 0.99 vs candidate 0.95 per case (a real 4-point gap), 40 cases x 3 trials: failures are rare, and the old
    normal-approximation interval covered the truth only ~92% of the time (overconfident exactly where a model switch
    is decided). The few-failures fallback must hold the nominal 95%."""
    import random

    from agentstats.comparisons import fixed_cases
    rng = random.Random(1)
    covered = 0
    for _ in range(2000):
        a = {f"c{i}": (3, sum(rng.random() < 0.99 for _ in range(3))) for i in range(40)}
        b = {f"c{i}": (3, sum(rng.random() < 0.95 for _ in range(3))) for i in range(40)}
        iv = fixed_cases(a, b)
        covered += iv.low <= -0.04 <= iv.high
    assert covered / 2000 >= 0.95
