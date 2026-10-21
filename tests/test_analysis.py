import json
import random

import pytest

from agentstats import analysis, checks, cli
from agentstats.records import Attempt, load_jsonl, write_jsonl

rng = random.Random(3)


def make(arm, runs, cases, rate, tags=None, infra_every=0, cost=0.1):
    out = []
    for r in range(1, runs + 1):
        for i in range(cases):
            infra = "Timeout" if infra_every and (i + r) % infra_every == 0 else None
            out.append(Attempt(case=f"c{i}", passed=None if infra else rng.random() < rate, arm=arm, run=r,
                               infra_error=infra, tags=(tags(i) if tags else []), metrics={"cost_usd": cost}))
    return out


def test_curated_multi_run_uses_fixed_cases():
    s = analysis.summarize(make("a", 4, 40, 0.7))
    assert s["pass_rate"]["method"] == "fixed_cases"
    assert s["claim"] == "on this eval set"
    assert "per_case" in s["not_used"] and "run_level" in s["not_used"]
    assert s["reliability"]["k_max"] == 4


def test_curated_single_run_is_wilson_and_says_indicative():
    s = analysis.summarize(make("a", 1, 40, 0.7))
    assert s["pass_rate"]["method"] == "wilson"
    assert "indicative" in s["claim"]


def test_sampled_uses_per_case():
    s = analysis.summarize(make("a", 3, 40, 0.7), design="sampled")
    assert s["pass_rate"]["method"] == "per_case"


def test_clustered_with_clean_groups():
    s = analysis.summarize(make("a", 3, 40, 0.7, tags=lambda i: [f"feature:f{i % 5}"]), design="clustered",
                           cluster_by="feature")
    assert s["pass_rate"]["method"] == "clustered"


def test_clustered_with_overlapping_groups_falls_back():
    s = analysis.summarize(make("a", 3, 40, 0.7, tags=lambda i: [f"feature:f{i % 5}", f"feature:g{i % 3}"]),
                           design="clustered", cluster_by="feature")
    assert s["pass_rate"]["method"] in ("fixed_cases", "run_level")
    assert any("more than one group" in n for n in s["notes"])


def test_infra_failures_excluded_and_flagged():
    s = analysis.summarize(make("a", 3, 40, 1.0, infra_every=5))
    assert s["pass_rate"]["estimate"] == 1.0
    assert s["infra"]["count"] > 0 and s["infra"]["kinds"] == {"Timeout": s["infra"]["count"]}
    assert any("infrastructure failures" in n for n in s["notes"])


def test_runs_with_different_cases_use_shared_cases():
    atts = make("a", 1, 40, 0.5) + [a for a in make("a", 2, 60, 0.5) if a.run == 2]
    s = analysis.summarize(atts)
    assert s["cases"] == 40
    assert any("different cases" in n for n in s["notes"])


def test_compare_curated_fixed_cases_detects_big_gain():
    c = analysis.compare(make("a", 5, 80, 0.4), make("b", 5, 80, 0.8))
    assert c["method"] == "fixed_cases" and c["verdict"] == "BETTER"


def shocked(arm, runs, cases, seed):
    """Every case has its own rate, and each run adds a shared shock (a provider's good or bad hour)."""
    r = random.Random(seed)
    rates = [r.random() for _ in range(cases)]
    out = []
    for run in range(1, runs + 1):
        shock = r.choice([-0.35, 0.35])
        for i, p in enumerate(rates):
            out.append(Attempt(case=f"c{i}", passed=r.random() < min(max(p + shock, 0), 1), arm=arm, run=run))
    return out


def test_run_wide_shock_falls_back_to_runs():
    s = analysis.summarize(shocked("a", 6, 200, 1))
    assert s["pass_rate"]["method"] == "run_level"
    assert any("run-wide shock" in n for n in s["notes"])
    c = analysis.compare(shocked("a", 6, 200, 1), shocked("b", 6, 200, 2))
    assert c["method"] == "welch"


def test_compare_no_difference_when_same():
    random.seed(1)
    c = analysis.compare(make("a", 5, 80, 0.6), make("b", 5, 80, 0.6))
    assert c["verdict"] in ("NO DETECTABLE DIFFERENCE", "BETTER", "WORSE")
    assert c["delta"]["low"] < 0.15


def test_compare_single_run_uses_mcnemar():
    a = [Attempt(case=f"c{i}", passed=i < 50) for i in range(100)]
    b = [Attempt(case=f"c{i}", passed=i < 52 or i in (90, 91)) for i in range(100)]
    c = analysis.compare(a, b)
    assert c["method"] == "mcnemar" and c["mcnemar"]["b_only"] == 4
    assert c["verdict"] == "NO DETECTABLE DIFFERENCE"


def test_compare_min_effect_ship_rule():
    c = analysis.compare(make("a", 5, 80, 0.3), make("b", 5, 80, 0.9), min_effect=0.05)
    assert c["ship"] is True


def test_compare_drops_unmatched_cases():
    c = analysis.compare(make("a", 3, 50, 0.5), make("b", 3, 40, 0.5))
    assert c["cases"] == 40 and len(c["dropped"]["only_a"]) == 10


def test_checks_rules():
    calls = [{"name": "get_order_details", "args": {"order_id": 42}, "step": 1},
             {"name": "issue_refund", "args": {}, "step": 2}]
    rules = [{"name": "looked_up", "called": "get_order_details"}, {"name": "no_refund", "not_called": "issue_refund"},
             {"name": "lookup_first", "before": ["get_order_details", "issue_refund"]},
             {"name": "right_order", "args": {"tool": "get_order_details", "match": {"order_id": "42"}}}]
    assert checks.run_checks(rules, calls) == {"looked_up": True, "no_refund": False, "lookup_first": True,
                                               "right_order": True}
    assert checks.check({"name": "x", "before": ["a", "b"]}, [{"name": "b"}, {"name": "a"}]) is False


def test_jsonl_roundtrip_and_cli(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(make("a", 3, 30, 0.5), a)
    write_jsonl(make("b", 3, 30, 0.9), b)
    assert len(load_jsonl(a)) == 90
    assert cli.main(["summary", str(a)]) == 0
    assert "Pass rate" in capsys.readouterr().out
    assert cli.main(["compare", "--a", str(a), "--b", str(b), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "BETTER"


@pytest.mark.parametrize("design", ["curated", "sampled", "clustered"])
def test_every_design_runs_end_to_end(design):
    atts = make("a", 3, 30, 0.6, tags=lambda i: [f"feature:f{i % 3}"])
    s = analysis.summarize(atts, design=design, cluster_by="feature")
    assert 0 <= s["pass_rate"]["estimate"] <= 1
