"""Rule gates: behaviour checks as a regression signal that needs no labels."""
from __future__ import annotations

import pytest

from agentstats import gate
from agentstats.policy import parse
from agentstats.records import from_dict

CASES = [f"c{i}" for i in range(60)]


def _rows(arm, runs, called):
    """Every attempt passes its label; `called(case, run)` says whether the required tool was called."""
    return [from_dict({"arm": arm, "run": r, "case": c, "passed": True,
                       "checks": {"called_count_tool": called(c, r), "called_dup_check": True}})
            for c in CASES for r in range(1, runs + 1)]


def _gate(require, cand, base):
    pol = parse({"pr": {"runs": 2, "require": require}})
    return gate.evaluate(pol.tiers["pr"], pol, cand, base)


def test_skipping_a_tool_fails_the_rule_gates_while_accuracy_passes():
    """The triage bug: labels rewarded the skip, so accuracy can't see it. The rules can."""
    base = _rows("main", 4, lambda c, r: True)
    cand = _rows("pr", 2, lambda c, r: int(c[1:]) % 2 == 0)          # skips the tool on half the cases
    g = _gate({"no_significant_drop": 0.0, "no_rule_regression": 0.0, "min_rule_rate": 0.95}, cand, base)
    rows = {r["requirement"]: r for r in g["rows"]}
    assert rows["no_significant_drop"]["result"] == gate.PASS
    regression = rows["no_rule_regression"]
    assert regression["result"] == gate.FAIL and "called_count_tool" in regression["detail"]
    assert rows["min_rule_rate"]["result"] == gate.FAIL
    assert g["overall"] == gate.FAIL
    assert "Trajectory rules" in gate.render(g)


def test_unchanged_behaviour_passes():
    base = _rows("main", 4, lambda c, r: True)
    cand = _rows("pr", 2, lambda c, r: True)
    g = _gate({"no_rule_regression": 0.0, "min_rule_rate": 0.95}, cand, base)
    assert {r["requirement"]: r["result"] for r in g["rows"]} == {"no_rule_regression": gate.PASS,
                                                                  "min_rule_rate": gate.PASS}
    assert g["rules"][0]["conf"] == pytest.approx(0.975)               # Bonferroni over 2 rules


def test_one_miss_is_not_proven_and_a_rate_near_the_floor_is_inconclusive():
    base = _rows("main", 4, lambda c, r: True)
    cand = _rows("pr", 2, lambda c, r: (c, r) != ("c0", 1))
    assert {r["requirement"]: r["result"] for r in _gate({"no_rule_regression": 0.0}, cand, base)["rows"]} == {
        "no_rule_regression": gate.PASS}
    near = _rows("pr", 2, lambda c, r: int(c[1:]) % 25 != 0 or r == 2)   # 117 of 120 = 97.5%, interval crosses 99%
    row = next(r for r in _gate({"min_rule_rate": 0.99}, near, base)["rows"] if r["requirement"] == "min_rule_rate")
    assert row["result"] == gate.INCONCLUSIVE and "can't tell yet" in row["detail"]


def test_a_baseline_without_rule_checks_is_inconclusive_not_a_pass():
    base = [from_dict({"arm": "main", "run": r, "case": c, "passed": True}) for c in CASES for r in (1, 2)]
    cand = _rows("pr", 2, lambda c, r: True)
    row = next(r for r in _gate({"no_rule_regression": 0.0}, cand, base)["rows"])
    assert row["result"] == gate.INCONCLUSIVE and "checked in both" in row["detail"]
