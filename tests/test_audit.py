import json
import random

from agentstats import audit, cli
from agentstats.records import Attempt, write_jsonl


def triage_like(cases=80, trials=3, skip=0.75, seed=1):
    """High accuracy, but most passing attempts skipped the lookup: the triage story."""
    r = random.Random(seed)
    return [Attempt(case=f"c{i}", passed=r.random() < 0.93, run=t,
                    checks={"lookup_before_count": r.random() > skip, "dup_check": True})
            for t in range(1, trials + 1) for i in range(cases)]


def by_name(res):
    return {ln["name"]: ln for ln in res["lines"]}


def test_blind_spots_flag_passing_attempts_that_broke_a_rule():
    ln = by_name(audit.run(triage_like(), sims=50))["blind spots"]
    assert ln["status"] == "bad"
    assert 0.6 < ln["rate"]["estimate"] < 0.9
    assert ln["per_rule"]["dup_check"] == 0 and "lookup_before_count" in ln["fix"]


def test_blind_spots_ok_when_rules_hold_and_skipped_without_checks():
    clean = [Attempt(case=f"c{i}", passed=True, run=t, checks={"r": True}) for t in (1, 2) for i in range(20)]
    assert by_name(audit.run(clean, sims=20))["blind spots"]["status"] == "ok"
    bare = [Attempt(case=f"c{i}", passed=True, run=t) for t in (1, 2) for i in range(20)]
    assert by_name(audit.run(bare, sims=20))["blind spots"]["status"] == "skipped"


def test_label_bias_warns_when_baseline_matches_labels_and_other_is_marked_down():
    r = random.Random(2)
    base = [Attempt(case=f"c{i}", passed=True, run=t) for t in (1, 2) for i in range(60)]
    other = [Attempt(case=f"c{i}", passed=(i >= 9) or r.random() < 0.2, run=t, arm="other")
             for t in (1, 2) for i in range(60)]
    ln = by_name(audit.run(base, other=other, sims=20))["label bias"]
    assert ln["status"] == "warn" and "baseline's own answers" in ln["text"]
    same = by_name(audit.run(base, other=[Attempt(case=a.case, passed=True, run=a.run) for a in base], sims=20))
    assert same["label bias"]["status"] == "ok"
    assert same["informative"]["status"] == "warn"           # every case always passes for both


def test_skipped_lines_name_the_input_they_need():
    res = by_name(audit.run([Attempt(case="a", passed=True), Attempt(case="b", passed=False)], sims=10))
    assert res["power"]["status"] == "skipped" and "2+ trials" in res["power"]["text"]
    assert "--other" in res["label bias"]["text"] and "--cases" in res["label source"]["text"]
    assert "--production" in res["coverage"]["text"]


def test_label_source_and_coverage():
    atts = triage_like(cases=20, trials=2)
    cases = {f"c{i}": ["label:model:haiku-4.5" if i < 15 else "label:human", "error_type:A"] for i in range(20)}
    prod = [{"tags": ["error_type:A"]}] * 30 + [{"tags": ["error_type:B"]}] * 20
    res = by_name(audit.run(atts, cases=cases, production=prod, prefixes=["error_type"], sims=20))
    assert res["label source"]["status"] == "bad" and "model:haiku-4.5 15" in res["label source"]["text"]
    assert res["coverage"]["status"] == "bad" and res["coverage"]["missing"] == ["B"]


def test_top_fix_prefers_bad_lines():
    res = audit.run(triage_like(), sims=50)
    bad = [ln["fix"] for ln in res["lines"] if ln["status"] == "bad" and ln["fix"]]
    assert res["top_fix"] == bad[0]


def test_cli_prints_scorecard_and_json(tmp_path, capsys):
    p = tmp_path / "r.jsonl"
    write_jsonl(triage_like(cases=30, trials=2), p)
    assert cli.main(["audit", str(p), "--sims", "20"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Eval health: 30 cases, 2 run(s)") and "blind spots" in out and "Top fix:" in out
    assert cli.main(["audit", str(p), "--sims", "20", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["lines"]) == 7
