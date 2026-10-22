import json
import random

import pytest

from agentstats import cli, gate, policy, sample
from agentstats.records import Attempt, write_jsonl


def make(runs, cases, rate, seed, arm="x", cost=0.1, latency=10.0, infra_every=0, checks=None, tags=None):
    rng = random.Random(seed)
    out = []
    for r in range(1, runs + 1):
        for i in range(cases):
            infra = "QuotaError" if infra_every and i % infra_every == 0 else None
            out.append(Attempt(case=f"c{i}", passed=None if infra else rng.random() < rate, arm=arm, run=r,
                               infra_error=infra, checks=dict(checks or {}), tags=tags(i) if tags else [],
                               metrics={"cost_usd": cost * (0.8 + 0.4 * rng.random()),
                                        "latency_s": latency * (0.8 + 0.4 * rng.random())}))
    return out


def pol(**require):
    return policy.parse({"design": "curated", "t": {"runs": 3, "require": require}})


def ev(p, cand, base=None):
    return gate.evaluate(p.tiers["t"], p, cand, base)


def row(g, name):
    return next(r for r in g["rows"] if r["requirement"].startswith(name))


def test_underpowered_run_is_inconclusive_never_pass():
    p = policy.parse({"t": {"runs": 1, "require": {"no_regression": 0.02}}})
    g = gate.evaluate(p.tiers["t"], p, make(1, 50, 0.6, 2), make(1, 50, 0.6, 1))
    assert row(g, "no_regression")["result"] == gate.INCONCLUSIVE
    assert g["exit_code"] == 2


def test_no_regression_passes_with_enough_runs_and_margin():
    p = pol(no_regression=0.10)
    g = ev(p, make(5, 200, 0.6, 2), make(5, 200, 0.6, 1))
    assert row(g, "no_regression")["result"] == gate.PASS
    assert g["overall"] == gate.PASS and g["exit_code"] == 0


def test_no_regression_fails_on_a_big_drop():
    g = ev(pol(no_regression=0.02), make(5, 200, 0.3, 2), make(5, 200, 0.8, 1))
    assert row(g, "no_regression")["result"] == gate.FAIL
    assert g["exit_code"] == 1


def test_inconclusive_gives_a_runs_estimate():
    g = ev(pol(no_regression=0.05), make(3, 60, 0.6, 2), make(3, 60, 0.6, 1))
    r = row(g, "no_regression")
    if r["result"] == gate.INCONCLUSIVE:
        assert "runs" in r["detail"]


def test_infra_unhealthy_exit_3():
    g = ev(pol(max_infra=0.05, min_pass_rate=0.1), make(3, 40, 0.9, 1, infra_every=4))
    assert g["overall"] == gate.UNHEALTHY and g["exit_code"] == 3


def test_rules_and_min_pass_rate():
    bad = make(3, 30, 0.9, 1, checks={"lookup_first": True})
    bad[5].checks["lookup_first"] = False
    g = ev(pol(rules=1.0, min_pass_rate=0.5), bad)
    assert row(g, "rules")["result"] == gate.FAIL and "c5:lookup_first" in row(g, "rules")["detail"]
    assert row(g, "min_pass_rate")["result"] == gate.PASS


def test_cost_ratio_pass_and_fail():
    base = make(3, 80, 0.6, 1, cost=0.10)
    cheap = ev(pol(max_cost_ratio=1.5), make(3, 80, 0.6, 2, cost=0.10), base)
    dear = ev(pol(max_cost_ratio=1.1), make(3, 80, 0.6, 2, cost=0.20), base)
    assert row(cheap, "max_cost_ratio")["result"] == gate.PASS
    assert row(dear, "max_cost_ratio")["result"] == gate.FAIL


def test_pass_hat_k():
    g = ev(pol(min_pass_hat_k=0.5), make(3, 80, 0.98, 1))
    assert row(g, "min_pass_hat_k")["result"] == gate.PASS
    g = ev(pol(min_pass_hat_k=0.9), make(3, 80, 0.5, 1))
    assert row(g, "min_pass_hat_k")["result"] == gate.FAIL


def test_too_few_runs_flagged():
    g = ev(pol(min_pass_rate=0.1), make(1, 40, 0.9, 1))
    assert row(g, "runs")["result"] == gate.INCONCLUSIVE


def test_baseline_required():
    with pytest.raises(ValueError, match="need --baseline"):
        ev(pol(no_regression=0.02), make(3, 10, 0.5, 1))


def test_smoke_tier_says_what_it_cannot_see():
    p = policy.parse({"smoke": {"runs": 1, "sample": 20, "require": {"min_pass_rate": 0.1}}})
    g = gate.evaluate(p.tiers["smoke"], p, make(1, 20, 0.6, 1))
    assert any("catches breakage, not regressions" in n for n in g["notes"])


def test_policy_rejects_unknown_requirement():
    with pytest.raises(ValueError, match="unknown requirement"):
        policy.parse({"t": {"require": {"max_vibes": 1}}})


def test_sample_is_stratified_deterministic_and_keeps_critical():
    cases = {f"c{i}": [f"feature:{['a', 'b', 'c'][i % 3]}"] + (["critical"] if i in (1, 2) else []) for i in range(90)}
    cases["rare"] = ["feature:d"]
    s1 = sample.stratified(cases, 12, "feature", ["critical"], seed=3)
    s2 = sample.stratified(cases, 12, "feature", ["critical"], seed=3)
    assert s1 == s2 and len(s1["cases"]) == 12
    assert {"c1", "c2", "rare"} <= set(s1["cases"])
    assert s1["groups"]["feature:d"] == 1
    assert sample.stratified(cases, 12, "feature", ["critical"], seed=4)["cases"] != s1["cases"]


def test_cli_gate_exit_code_and_step_summary(tmp_path, monkeypatch, capsys):
    (tmp_path / "p.toml").write_text('[nightly]\nruns = 3\nrequire.no_regression = 0.02\n')
    write_jsonl(make(3, 100, 0.3, 2), tmp_path / "cand.jsonl")
    write_jsonl(make(3, 100, 0.8, 1), tmp_path / "base.jsonl")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    code = cli.main(["gate", "--policy", str(tmp_path / "p.toml"), "--tier", "nightly",
                     "--candidate", str(tmp_path / "cand.jsonl"), "--baseline", str(tmp_path / "base.jsonl")])
    assert code == 1
    assert "FAIL" in summary.read_text()


def test_cli_sample_and_init_ci(tmp_path, capsys):
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps({"case": f"c{i}", "tags": [f"feature:f{i % 2}"]}) + "\n"
                                                  for i in range(10)))
    assert cli.main(["sample", str(tmp_path / "cases.jsonl"), "--n", "4", "--stratify-by", "feature"]) == 0
    assert len(capsys.readouterr().out.split()) == 4
    assert cli.main(["init-ci", "--dir", str(tmp_path)]) == 0
    assert (tmp_path / ".github/workflows/agent-evals-nightly.yml").exists()
    p = policy.load(tmp_path / "agentstats.toml")
    assert set(p.tiers) == {"smoke", "nightly", "release"}
    with pytest.raises(SystemExit):
        cli.main(["init-ci", "--dir", str(tmp_path)])


def test_run_wide_shock_with_two_runs_is_inconclusive_rerun():
    """Two candidate runs, one hit by a shock that flips many cases at once (seen for real on a triage replay):
    too few runs to fall back to run-level statistics, so the gate must ask for a rerun, not decide."""
    base = make(4, 120, 0.85, 1)
    r = random.Random(9)
    cand = []
    for run, shift in ((1, 0.0), (2, -0.45)):
        for i in range(120):
            cand.append(Attempt(case=f"c{i}", passed=r.random() < 0.85 + shift, run=run, metrics={"cost_usd": 0.1}))
    p = policy.parse({"t": {"runs": 2, "require": {"no_regression": 0.05}}})
    g = gate.evaluate(p.tiers["t"], p, cand, base)
    r0 = row(g, "no_regression")
    assert r0["result"] == gate.INCONCLUSIVE and "rerun" in r0["detail"]


def test_no_significant_drop_passes_unchanged_and_fails_a_real_drop():
    p = policy.parse({"t": {"runs": 2, "require": {"no_significant_drop": 0.0}}})
    same = gate.evaluate(p.tiers["t"], p, make(2, 120, 0.8, 2), make(4, 120, 0.8, 1))
    assert row(same, "no_significant_drop")["result"] == gate.PASS
    assert "catches drops" in row(same, "no_significant_drop")["detail"]
    worse = gate.evaluate(p.tiers["t"], p, make(2, 120, 0.5, 2), make(4, 120, 0.85, 1))
    assert row(worse, "no_significant_drop")["result"] == gate.FAIL and worse["exit_code"] == 1


def test_all_infra_failures_is_unhealthy_not_a_crash():
    cand = [Attempt(case=f"c{i}", passed=None, run=1, infra_error="BadRequestError") for i in range(20)]
    p = policy.parse({"t": {"runs": 1, "require": {"no_significant_drop": 0.0, "max_infra": 0.1}}})
    g = gate.evaluate(p.tiers["t"], p, cand, make(4, 20, 0.9, 1))
    assert g["overall"] == gate.UNHEALTHY and g["exit_code"] == 3
    assert row(g, "graded attempts")["result"] == gate.UNHEALTHY
