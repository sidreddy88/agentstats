"""Failures become candidate eval cases (promote.py): what counts, deduplication, provenance."""
from __future__ import annotations

import json

import pytest

from agentstats import cli, promote
from agentstats.records import from_dict, write_jsonl


def row(case, passed=None, checks=None, run=1, tags=None, infra=None, no_answer=None):
    return from_dict({"case": case, "passed": passed, "checks": checks or {}, "run": run, "tags": tags or [],
                      "infra_error": infra, "no_answer": no_answer})


def production():
    """Rules-only production rows: refunds without a lookup (many, two intents), one escalation miss, noise."""
    rows = []
    for i in range(30):                                   # the same failure, thirty times over 6 conversations
        rows.append(row(f"chat-{i % 6}", checks={"lookup_before_refund": False, "escalated_when_asked": True},
                        run=i // 6 + 1, tags=[f"intent:{'refund' if i % 6 < 4 else 'billing'}"]))
    rows.append(row("chat-90", checks={"lookup_before_refund": True, "escalated_when_asked": False},
                    tags=["intent:account"]))
    rows += [row(f"ok-{i}", checks={"lookup_before_refund": True, "escalated_when_asked": True}) for i in range(20)]
    rows.append(row("broken", checks={"lookup_before_refund": False}, infra="503"))     # infra: not a failure
    return rows


def test_failures_are_failed_verdicts_or_broken_rules_not_infra():
    assert promote.is_failure(row("a", passed=False))
    assert promote.is_failure(row("a", checks={"r": False}))                    # rules-only production row
    assert promote.is_failure(row("a", passed=True, checks={"r": False}))       # right answer, wrong behaviour
    assert not promote.is_failure(row("a", passed=True, checks={"r": True}))
    assert not promote.is_failure(row("a", passed=False, infra="503"))


def test_signatures_deduplicate_and_keep_the_most_frequent_cases():
    cands, report = promote.find(production(), source="production", found="2026-11-09")
    sigs = {r["signature"]: r for r in report}
    assert set(sigs) == {"rules:lookup_before_refund", "rules:escalated_when_asked"}
    big = sigs["rules:lookup_before_refund"]
    assert big["failures"] == 30 and big["cases"] == 6 and big["proposed"] == 3
    assert report[0]["signature"] == "rules:lookup_before_refund"                # most frequent first
    assert len(cands) == 4 and all(c["approved"] is False for c in cands)
    c = next(c for c in cands if c["case"] == "chat-90")
    assert c["failed_checks"] == ["escalated_when_asked"]
    assert {"source:production", "found:2026-11-09", "label:none", "intent:account"} <= set(c["tags"])


def test_group_by_splits_signatures_and_existing_cases_are_skipped():
    cands, report = promote.find(production(), source="production", group_by="intent",
                                 existing={"chat-0", "chat-1"}, per_signature=5)
    sigs = {r["signature"]: r for r in report}
    refund = sigs["rules:lookup_before_refund | intent:refund"]
    assert refund["cases"] == 4 and refund["already_in_eval_set"] == 2 and refund["proposed"] == 2
    assert "rules:lookup_before_refund | intent:billing" in sigs
    assert not {"chat-0", "chat-1"} & {c["case"] for c in cands}


def test_graded_failures_and_no_answers():
    rows = [row("t1", passed=False), row("t1", passed=False, run=2), row("t2", passed=False, no_answer="step_limit")]
    _, report = promote.find(rows, source="nightly")
    assert {r["signature"] for r in report} == {"failed", "failed | no_answer:step_limit"}


def test_label_from_is_checked():
    with pytest.raises(ValueError, match="--label-from"):
        promote.find(production(), source="production", label_from="gpt")


def test_accept_appends_only_approved_and_warns_when_one_model_labelled_most(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps({"case": f"old-{i}", "tags": ["source:curated", "label:human"]}) + "\n"
                             for i in range(2)))
    cands, _ = promote.find(production(), source="production", label_from="model:haiku-4.5", found="2026-11-09")
    for c in cands[:3]:
        c["approved"] = True
    cpath = tmp_path / "candidates.jsonl"
    cpath.write_text("".join(json.dumps(c) + "\n" for c in cands))
    added, prov = promote.accept(cpath, cases)
    assert added == 3 and prov["cases"] == 5
    assert prov["labels"] == {"label:human": 2, "label:model:haiku-4.5": 3}
    assert prov["warnings"] and "haiku-4.5" in prov["warnings"][0]
    assert promote.accept(cpath, cases)[0] == 0                                  # accepting twice adds nothing


def test_cli_find_then_accept(tmp_path, capsys):
    rows = tmp_path / "prod.jsonl"
    write_jsonl(production(), rows)
    out = tmp_path / "cands.jsonl"
    assert cli.main(["promote", "find", str(rows), "--source", "production", "--found", "2026-11-09",
                     "--out", str(out)]) == 0
    assert "rules:lookup_before_refund" in capsys.readouterr().out
    cands = [json.loads(x) for x in out.read_text().splitlines()]
    cands[0]["approved"] = True
    out.write_text("".join(json.dumps(c) + "\n" for c in cands))
    assert cli.main(["promote", "accept", str(out), "--into", str(tmp_path / "cases.jsonl")]) == 0
    assert "added 1 cases" in capsys.readouterr().out
    assert cli.main(["sample", str(tmp_path / "cases.jsonl"), "--n", "1"]) == 0   # the cases file works with sample
