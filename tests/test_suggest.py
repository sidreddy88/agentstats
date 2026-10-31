"""agentstats suggest: rules mined from trajectories with outcome evidence, advice, and apply writing only accepted
rules. Synthetic runs with known answers."""
from __future__ import annotations

import json

import pytest

from agentstats import cli, suggest


def runs(n=200):
    """search always first; 'submit' skipped in 20 runs, all failing (a gate); 'read' skipped in 20 runs that all
    still pass (breaking it doesn't hurt: monitor only); 'verify' in half the runs (too rare to be a rule)."""
    out = []
    for i in range(n):
        tools = ["search", "grep"]
        if i % 10 != 3:
            tools.append("read")
        if i % 2:
            tools.append("verify")
        submitted = i % 10 != 0
        if submitted:
            tools.append("submit")
        passed = submitted and i % 5 != 1          # 80% of submitting runs pass, none of the others
        out.append({"agent": "DiagAgent", "case": f"c{i}", "passed": passed, "tools": tools,
                    "end": "I was unable to find an answer within the allowed number of steps." if not submitted
                    else f"Diagnosis {i}: root cause in file_{i}.py"})
    return out


def by_name(items):
    return {it["rule"]["name"]: it for it in items if it["kind"] == "rule"}


def test_a_rule_whose_breaking_hurts_is_a_gate_and_one_without_effect_a_monitor():
    rules = by_name(suggest.propose(runs())["items"])
    gate = rules["called_submit"]
    assert gate["use"] == "gate" and "0% when broken" in gate["why"]
    assert gate["evidence"]["graded"]["effect"]["high"] < 0
    assert rules["called_read"]["use"] == "monitor"            # skipped sometimes, outcomes no different
    assert "called_verify" not in rules                        # held in 50%: not a rule


def test_orderings_implied_by_an_always_first_tool_are_not_proposed():
    rules = by_name(suggest.propose(runs())["items"])
    assert not any(name.startswith("search_before_") for name in rules)


def test_a_repeated_template_ending_becomes_no_answer_advice_but_varied_answers_dont():
    advice = [it["text"] for it in suggest.propose(runs())["items"] if it["kind"] == "advice"]
    assert any("allowed number of steps" in a for a in advice)
    json_answers = [{"agent": "T", "case": str(i), "passed": None, "tools": ["a"],
                     "end": json.dumps({"decision": "real", "severity": "P2", "n": i})} for i in range(50)]
    assert not any("ended with" in it.get("text", "") for it in suggest.propose(json_answers)["items"])


def test_configured_rules_are_skipped_and_listed(tmp_path):
    (tmp_path / "diag.json").write_text(json.dumps({"agent": "DiagAgent",
                                                     "rules": [{"name": "my_submit_rule", "called": "submit"}]}))
    items = suggest.propose(runs(), tmp_path)["items"]
    assert "called_submit" not in by_name(items)
    assert any(it["kind"] == "info" and "my_submit_rule" in it["text"] for it in items)


def test_few_runs_and_no_grades_get_advice():
    few = suggest.propose(runs(5))["items"]
    assert few[0]["kind"] == "advice" and "too few" in few[0]["text"]
    ungraded = [dict(r, passed=None) for r in runs()]
    items = suggest.propose(ungraded)["items"]
    assert any("no graded runs" in it.get("text", "") for it in items)
    assert all(it["use"] == "monitor" for it in items if it["kind"] == "rule")


def test_apply_writes_only_accepted_rules_and_keeps_existing_config(tmp_path):
    (tmp_path / "diag.json").write_text(json.dumps({"agent": "DiagAgent", "adapter": "custom", "rules": []}))
    p = suggest.propose(runs())
    gate = by_name(p["items"])["called_submit"]
    written = suggest.apply(p, tmp_path, [gate["id"]])
    cfg = json.loads((tmp_path / "diag.json").read_text())
    assert written == [str(tmp_path / "diag.json")] and cfg["adapter"] == "custom"
    assert [r["name"] for r in cfg["rules"]] == ["called_submit"] and cfg["rules"][0]["use"] == "gate"
    suggest.apply(p, tmp_path, [gate["id"]])                       # applying twice doesn't duplicate the rule
    assert len(json.loads((tmp_path / "diag.json").read_text())["rules"]) == 1


def test_apply_creates_a_config_named_after_the_agent(tmp_path):
    p = suggest.propose(runs())
    suggest.apply(p, tmp_path / "agents", [by_name(p["items"])["called_submit"]["id"]])
    assert json.loads((tmp_path / "agents" / "diag.json").read_text())["agent"] == "DiagAgent"


def test_input_is_validated(tmp_path):
    bad = tmp_path / "t.jsonl"
    bad.write_text(json.dumps({"agent": "A", "tools": "grep"}) + "\n")
    with pytest.raises(ValueError, match="'tools' list"):
        suggest.load([bad])


def test_cli_suggest_then_apply(tmp_path, capsys):
    traj = tmp_path / "t.jsonl"
    traj.write_text("".join(json.dumps(r) + "\n" for r in runs()))
    assert cli.main(["suggest", str(traj), "--out", str(tmp_path / "p.json")]) == 0
    assert "GATE" in capsys.readouterr().out
    assert cli.main(["suggest-apply", str(tmp_path / "p.json"), "--config-dir", str(tmp_path / "cfg"), "--all"]) == 0
    out = capsys.readouterr().out
    assert "wrote" in out and "advice" in out
    assert (tmp_path / "cfg" / "diag.json").exists()
