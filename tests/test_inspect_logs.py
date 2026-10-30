"""Inspect AI eval logs as attempt rows (inspect_logs.py).

The fixtures were made with Inspect's mock model (no API calls)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from agentstats import cli, inspect_logs, records

FIX = Path(__file__).parent / "fixtures"
TOOLS = FIX / "inspect_tools.json"      # 4 samples: lookup_order then an answer; t3's tool raised (sample error)
EPOCHS = FIX / "inspect_epochs.json"    # 6 samples x 2 epochs, scorers includes (all C) and match (all I)
RULES = [{"name": "looked_up", "called": "lookup_order"},
         {"name": "no_refund_tool", "not_called": "issue_refund"},
         {"name": "looked_up_a0", "args": {"tool": "lookup_order", "match": {"order_id": "A0"}}}]


def _eval_readable() -> bool:
    try:
        with zipfile.ZipFile(FIX / "inspect_tools.eval") as z:
            z.read("header.json")
        return True
    except NotImplementedError:
        try:
            import inspect_ai  # noqa: F401
            return True
        except ImportError:
            return False


def test_one_row_per_sample_and_epoch_with_the_headline_scorer():
    rows = inspect_logs.load(EPOCHS)
    assert len(rows) == 12
    assert {r.case for r in rows} == {f"case-{i}" for i in range(6)} and {r.run for r in rows} == {1, 2}
    assert all(r.passed for r in rows)                          # headline scorer: includes
    assert {r.arm for r in rows} == {"mockllm/model"}           # arm defaults to the model
    r = rows[0]
    assert "task:demo" in r.tags and "model:mockllm/model" in r.tags
    assert any(t.startswith("intent:") for t in r.tags)        # sample metadata -> tags, so slices work
    assert r.metrics["input_tokens"] > 0 and r.metrics["steps"] == 1 and "latency_s" in r.metrics


def test_choose_another_scorer():
    assert not any(r.passed for r in inspect_logs.load(EPOCHS, scorer="match"))
    with pytest.raises(ValueError, match="no scorer 'nope'"):
        inspect_logs.load(EPOCHS, scorer="nope")


def test_sample_error_is_an_infra_failure_by_default_and_a_failure_on_request():
    rows = {r.case: r for r in inspect_logs.load(TOOLS)}
    assert rows["t0"].passed is True and rows["t1"].passed is False and rows["t2"].passed is True
    assert rows["t3"].infra_error and "upstream 503" in rows["t3"].infra_error and not rows["t3"].valid
    failed = {r.case: r for r in inspect_logs.load(TOOLS, errors="fail")}["t3"]
    assert failed.passed is False and failed.infra_error is None


def test_tool_calls_fill_rule_checks_in_order():
    rows = {r.case: r for r in inspect_logs.load(TOOLS, rules=RULES)}
    assert all(r.checks["looked_up"] and r.checks["no_refund_tool"] for r in rows.values())
    assert rows["t0"].checks["looked_up_a0"] and not rows["t1"].checks["looked_up_a0"]
    assert rows["t0"].metrics["tool_calls"] == 1 and rows["t0"].metrics["steps"] == 2


@pytest.mark.parametrize("value,expected", [
    ("C", (True, None)), ("I", (False, None)), ("P", (False, None)), ("N", (False, "noanswer")),
    (True, (True, None)), (False, (False, None)), (1, (True, None)), (0.5, (False, None)), (1.0, (True, None)),
])
def test_score_values(value, expected):
    assert inspect_logs.verdict(value) == expected


def test_numeric_threshold_and_unreadable_values():
    assert inspect_logs.verdict(0.7, pass_threshold=0.5) == (True, None)
    with pytest.raises(ValueError, match="dict-valued"):
        inspect_logs.verdict({"a": 1})


def test_several_scorers_without_a_headline_need_a_choice():
    log = json.loads(EPOCHS.read_text())
    log["results"].pop("headline")
    with pytest.raises(ValueError, match="choose one with --scorer"):
        inspect_logs.to_attempts(log)


def test_no_answer_and_limits():
    log = json.loads(TOOLS.read_text())
    s = log["samples"][1]                                       # t1: scored I
    s["limit"] = {"type": "message", "limit": 2}
    log["samples"][0]["scores"]["includes"]["value"] = "N"
    rows = {r.case: r for r in inspect_logs.to_attempts(log)}
    assert rows["t0"].passed is False and rows["t0"].no_answer == "noanswer"
    assert rows["t1"].no_answer == "limit_message"


def test_a_log_where_every_sample_errored_reads_as_infrastructure_failures():
    log = json.loads(TOOLS.read_text())
    log["status"] = "error"
    log.pop("results", None)
    for s in log["samples"]:
        s.pop("scores", None)
        s["error"] = {"message": "BadRequestError: Unable to access non-serverless model"}
    rows = inspect_logs.to_attempts(log)
    assert len(rows) == 4 and all(r.infra_error and "non-serverless" in r.infra_error for r in rows)
    assert not any(r.valid for r in rows)


def test_records_load_reads_inspect_logs_directly(tmp_path):
    assert inspect_logs.is_inspect_log(EPOCHS) and not inspect_logs.is_inspect_log(FIX / "langfuse_traces.json")
    assert len(records.load(EPOCHS)) == 12
    assert cli.main(["summary", str(EPOCHS)]) == 0


def test_cli_convert_inspect(tmp_path):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(RULES))
    out = tmp_path / "rows.jsonl"
    assert cli.main(["convert", "--format", "inspect", str(TOOLS), "--rules", str(rules), "--arm", "mock",
                     "--out", str(out)]) == 0
    rows = records.load_jsonl(out)
    assert len(rows) == 4 and {r.arm for r in rows} == {"mock"} and rows[0].checks["looked_up"]


@pytest.mark.skipif(not _eval_readable(), reason="needs inspect-ai or Python 3.14+ (zstd) to read .eval")
def test_eval_archive_matches_the_json_log():
    a = {(r.case, r.run): (r.passed, r.infra_error is None) for r in inspect_logs.load(FIX / "inspect_tools.eval")}
    b = {(r.case, r.run): (r.passed, r.infra_error is None) for r in inspect_logs.load(TOOLS)}
    assert a == b
