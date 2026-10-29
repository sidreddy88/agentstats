"""The switch rule (switch.py), the decision record (switch_record.py) and the CLI, on synthetic attempt rows."""
from __future__ import annotations

import json

import pytest

from agentstats import cli
from agentstats import switch as decide
from agentstats.records import from_dict


def rows(arm, outcomes, cost=0.10, tag=None):
    """outcomes: {case: [True/False/None per run]} (None = infrastructure failure)."""
    out = []
    for case, results in outcomes.items():
        for run, ok in enumerate(results, start=1):
            r = {"arm": arm, "run": run, "case": case, "passed": ok, "cost_usd": cost,
                 "infra_error": "RateLimit" if ok is None else None, "tags": [f"repo:{tag(case)}"] if tag else []}
            out.append(r)
    return out


def att(rs):
    return [from_dict(r) for r in rs]


CASES = [f"c{i}" for i in range(60)]
ALL_PASS = {c: [True] * 4 for c in CASES}
RULE = decide.Rule(margin=0.03)


def test_same_results_cheaper_is_switch():
    d = decide.decide(att(rows("cur", ALL_PASS, 0.10)), att(rows("cand", ALL_PASS, 0.02)), None, RULE,
                      provenance="independent", preregistered=True)
    assert d.decision == decide.SWITCH and d.steps[0].delta["low"] >= -0.03
    assert d.warnings == []
    assert d.arms["candidate"]["cost_per_accepted_usd"] == pytest.approx(0.02)


def test_clearly_worse_is_keep():
    worse = {c: [i >= 18] * 4 for i, c in enumerate(CASES)}          # 30% of cases always fail
    d = decide.decide(att(rows("cur", ALL_PASS)), att(rows("cand", worse, 0.02)), None, RULE)
    assert d.decision == decide.KEEP and "optimistic end" in d.reason


def test_more_expensive_is_keep():
    d = decide.decide(att(rows("cur", ALL_PASS, 0.02)), att(rows("cand", ALL_PASS, 0.10)), None, RULE)
    assert d.decision == decide.KEEP and "not cheaper" in d.reason


def test_too_noisy_to_tell_is_not_yet():
    few = [f"c{i}" for i in range(10)]
    cur = {c: [True, False] for c in few}
    cand = {c: ([False, False] if c == "c0" else [True, False]) for c in few}
    d = decide.decide(att(rows("cur", cur)), att(rows("cand", cand, 0.02)), None, RULE)
    assert d.decision == decide.NOT_YET
    assert d.steps[0].delta["low"] < -0.03 < d.steps[0].delta["high"]


def test_infrastructure_failures_are_unhealthy():
    broken = {c: [None] * 4 for c in CASES}
    d = decide.decide(att(rows("cur", ALL_PASS)), att(rows("cand", broken, 0.02)), None, RULE)
    assert d.decision == decide.UNHEALTHY


def test_adapted_step_can_rescue_and_the_better_step_decides():
    worse = {c: [i >= 18] * 4 for i, c in enumerate(CASES)}
    d = decide.decide(att(rows("cur", ALL_PASS)), att(rows("cand", worse, 0.02)),
                      att(rows("adapted", ALL_PASS, 0.03)), RULE)
    assert [s.decision for s in d.steps] == [decide.KEEP, decide.SWITCH]
    assert d.decision == decide.SWITCH and d.reason.startswith("adapted:")


def test_passing_as_is_still_reports_a_better_adapted_harness():
    cur = {c: [i >= 12] * 4 for i, c in enumerate(CASES)}            # current 80%
    cand = {c: [i >= 6] * 4 for i, c in enumerate(CASES)}            # as-is 90%
    d = decide.decide(att(rows("cur", cur)), att(rows("cand", cand, 0.02)),
                      att(rows("adapted", ALL_PASS, 0.02)), RULE)
    assert d.decision == decide.SWITCH
    assert any("consider shipping it too" in w for w in d.warnings)


def test_slices_reported_by_default_and_enforced_when_strict():
    def tag(c):
        return "a" if int(c[1:]) < 30 else "b"

    cand = {c: ([i % 2 == 0] * 4 if tag(c) == "b" else [True] * 4) for i, c in enumerate(CASES)}
    cur, can = att(rows("cur", ALL_PASS, tag=tag)), att(rows("cand", cand, 0.02, tag=tag))
    loose = decide.decide(cur, can, None, decide.Rule(margin=0.35), slice_by="repo")
    assert loose.decision == decide.SWITCH and set(loose.steps[0].slices) == {"a", "b"}
    assert loose.steps[0].slices["b"].decision != decide.SWITCH
    assert any("reported, not enforced" in w for w in loose.warnings)
    strict = decide.decide(cur, can, None, decide.Rule(margin=0.35), slice_by="repo", strict_slices=True)
    assert strict.decision != decide.SWITCH and "strict slices" in strict.reason


def test_warnings_for_provenance_margin_and_missing_cost():
    cur = [r | {"cost_usd": None} for r in rows("cur", ALL_PASS)]
    d = decide.decide(att(cur), att(rows("cand", ALL_PASS)), None, RULE, provenance="current-model")
    text = " ".join(d.warnings)
    assert "favours it" in text and "not declared before" in text and "cost not checked" in text
    with pytest.raises(ValueError):
        decide.decide(att(cur), att(rows("cand", ALL_PASS)), None, RULE, provenance="vibes")


def _write(path, rs):
    path.write_text("".join(json.dumps(r) + "\n" for r in rs))
    return path


def test_cli_writes_the_record_and_exits_with_the_decision(tmp_path, capsys):
    cur = _write(tmp_path / "cur.jsonl", rows("cur", ALL_PASS, 0.10))
    cand = _write(tmp_path / "cand.jsonl", rows("cand", ALL_PASS, 0.02))
    code = cli.main(["switch", "--current", str(cur), "--candidate", str(cand), "--margin", "0.03",
                     "--provenance", "independent", "--preregistered", "--name", "demo", "--out", str(tmp_path / "d")])
    assert code == 0 and "**SWITCH**" in capsys.readouterr().out
    rec = json.loads((tmp_path / "d" / "decision.json").read_text())
    assert rec["decision"] == "SWITCH" and rec["margin_declared_before_results"] is True
    assert set(rec["inputs"]) == {"current", "candidate"} and len(rec["inputs"]["current"]["sha256"]) == 16
    assert rec["rule"]["margin"] == 0.03 and rec["steps"][0]["name"] == "as-is"
    assert (tmp_path / "d" / "decision.md").read_text().startswith("# agentstats switch decision: demo")
    worse = _write(tmp_path / "worse.jsonl", rows("cand", {c: [i >= 18] * 4 for i, c in enumerate(CASES)}, 0.02))
    assert cli.main(["switch", "--current", str(cur), "--candidate", str(worse), "--margin", "0.03",
                     "--name", "demo2", "--out", str(tmp_path / "d2")]) == 1
