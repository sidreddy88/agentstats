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
