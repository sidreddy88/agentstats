"""Row validation and the format contract, the no-answer outcome, and per-slice gate comparisons."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentstats import analysis, gate
from agentstats.policy import parse
from agentstats.records import RowError, from_dict, load_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


# ---- validation and the contract with the tools that write rows -------------------------------------------------

@pytest.mark.parametrize("producer", ["rows_from_harness_evolve.jsonl", "rows_from_agent_platform_triage_gate.jsonl"])
def test_rows_written_by_other_tools_load(producer):
    """harness-evolve export and agent-platform's triage gate both write agentstats rows: a format change on either
    side must fail here, not as a traceback in someone's CI (the first replay crashed on `steps` being a list)."""
    atts = load_jsonl(FIXTURES / producer)
    assert atts and all(a.case and a.passed in (True, False) for a in atts)
    assert all("cost_usd" in a.metrics for a in atts)


@pytest.mark.parametrize("row, message", [
    ({"case": "c", "passed": True, "steps": [{}, {}]}, "'steps' must be a number, got list"),
    ({"case": "c", "passed": "yes"}, "'passed' must be true, false or null"),
    ({"passed": True}, "'case' is missing"),
    ({"case": "c", "passed": True, "run": "first"}, "'run' must be an integer"),
    ({"case": "c", "passed": True, "tags": "slow"}, "'tags' must be a list of strings"),
    ({"case": "c", "passed": True, "checks": {"r": "ok"}}, "'checks' must map rule names to true/false"),
    ({"case": "c", "passed": True, "schema_version": 2}, "schema_version 2"),
    ({"case": "c", "passed": True, "no_answer": "step_limit"}, "an attempt with no answer can't pass"),
])
def test_bad_rows_get_a_clear_error(row, message):
    with pytest.raises(RowError, match=message):
        from_dict(row)


def test_errors_name_the_file_and_line(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"case": "a", "passed": true}\n{"case": "b", "passed": true, "steps": [1]}\n')
    with pytest.raises(ValueError, match=r"rows.jsonl:2: 'steps' must be a number"):
        load_jsonl(p)


# ---- no answer ----------------------------------------------------------------------------------------------------

def _rows(arm, results, tag=None):
    """results: {case: [True/False/"step_limit" per run]}."""
    out = []
    for case, rs in results.items():
        for run, r in enumerate(rs, 1):
            row = {"arm": arm, "run": run, "case": case, "tags": [f"t:{tag(case)}"] if tag else []}
            row.update({"no_answer": r, "passed": False} if isinstance(r, str) else {"passed": r})
            out.append(from_dict(row))
    return out


def test_no_answer_counts_as_a_failure_and_is_reported_apart():
    atts = [from_dict({"case": "a", "passed": True}), from_dict({"case": "b", "no_answer": "step_limit"}),
            from_dict({"case": "c", "passed": False})]
    assert atts[1].passed is False and atts[1].valid
    s = analysis.summarize(atts)
    assert s["pass_rate"]["estimate"] == pytest.approx(1 / 3)
    assert s["no_answer"] == {"count": 1, "rate": pytest.approx(1 / 3), "kinds": {"step_limit": 1}}
    assert any("no usable answer" in n for n in s["notes"])


def test_gate_fails_on_too_many_no_answers_and_says_why():
    cases = {f"c{i}": [True, True] for i in range(50)}
    broken = {c: ["step_limit", True] if i < 6 else [True, True] for i, c in enumerate(cases)}
    pol = parse({"pr": {"runs": 2, "require": {"max_no_answer": 0.02}}})
    g = gate.evaluate(pol.tiers["pr"], pol, _rows("b", broken), _rows("a", cases))
    row = next(r for r in g["rows"] if r["requirement"] == "max_no_answer")
    assert row["result"] == gate.FAIL and "step_limit" in row["detail"] and "baseline 0.0%" in row["detail"]
    assert g["overall"] == gate.FAIL


def test_without_the_requirement_a_jump_in_no_answers_is_still_noted():
    cases = {f"c{i}": [True, True] for i in range(50)}
    broken = {c: ["parse_error", True] if i < 3 else [True, True] for i, c in enumerate(cases)}
    pol = parse({"pr": {"runs": 2, "require": {"no_significant_drop": 0.0}}})
    g = gate.evaluate(pol.tiers["pr"], pol, _rows("b", broken), _rows("a", cases))
    assert any("no usable answer" in n and "parse_error" in n for n in g["notes"])


# ---- per-slice comparisons -------------------------------------------------------------------------------------

def _slice_data(drop_in_b: bool, n=40):
    tag = lambda c: "a" if int(c[1:]) < n else "b"  # noqa: E731
    base = {f"c{i}": [True] * 4 for i in range(2 * n)}
    cand = {c: ([False] * 2 if drop_in_b and tag(c) == "b" else [True] * 2) for c in base}
    return _rows("a", base, tag), _rows("b", cand, tag)


def test_a_proven_slice_drop_fails_even_when_the_rest_is_fine():
    base, cand = _slice_data(drop_in_b=True)
    pol = parse({"pr": {"runs": 2, "slice_by": "t", "require": {"max_slice_drop": 0.10}}})
    g = gate.evaluate(pol.tiers["pr"], pol, cand, base)
    row = next(r for r in g["rows"] if r["requirement"] == "max_slice_drop")
    assert row["result"] == gate.FAIL and "proven worse: b" in row["detail"]
    assert {x["slice"] for x in g["slices"]} == {"a", "b"}
    assert g["slices"][0]["conf"] == pytest.approx(0.975)           # 95% overall, Bonferroni over 2 slices
    assert "Per slice (t)" in gate.render(g)


def test_slices_are_reported_without_the_requirement_and_intervals_stay_in_range():
    base, cand = _slice_data(drop_in_b=True, n=2)
    pol = parse({"pr": {"runs": 2, "slice_by": "t", "require": {"no_significant_drop": 0.0}}})
    g = gate.evaluate(pol.tiers["pr"], pol, cand, base)
    assert not any(r["requirement"] == "max_slice_drop" for r in g["rows"])
    assert all(-1.0 <= x["low"] <= x["high"] <= 1.0 for x in g["slices"])


def test_a_large_but_unproven_slice_drop_is_named_as_a_lead():
    tag = lambda c: "small" if c in ("c0", "c1") else "big"  # noqa: E731
    base = {f"c{i}": [True, True, False, True] for i in range(40)}
    cand = {c: ([False, True] if tag(c) == "small" else [True, True]) for c in base}
    pol = parse({"pr": {"runs": 2, "slice_by": "t", "require": {"max_slice_drop": 0.10}}})
    g = gate.evaluate(pol.tiers["pr"], pol, _rows("b", cand, tag), _rows("a", base, tag))
    row = next(r for r in g["rows"] if r["requirement"] == "max_slice_drop")
    assert row["result"] == gate.PASS and "not proven" in row["detail"] and "small" in row["detail"]


def test_max_slice_drop_needs_slice_by():
    with pytest.raises(ValueError, match="needs slice_by"):
        parse({"pr": {"require": {"max_slice_drop": 0.1}}})
