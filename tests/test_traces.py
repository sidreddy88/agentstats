"""OpenTelemetry and Langfuse traces as attempt rows (traces.py), and rules-only gating on them."""
from __future__ import annotations

import json
from pathlib import Path

from agentstats import cli, gate, traces
from agentstats.policy import parse

FIX = Path(__file__).parent / "fixtures"
RULES = [{"name": "called_count", "called": "get_occurrence_count"},
         {"name": "count_for_a_day", "args": {"tool": "get_occurrence_count", "match": {"hours": 24}}},
         {"name": "called_dup", "called": "check_duplicate_pr"}]


def test_otlp_traces_become_rows_with_tool_calls_tokens_and_latency():
    rows = traces.convert([FIX / "otlp_traces.json"], "otlp", rules=RULES)
    by_id = {r.metrics["latency_s"]: r for r in rows}
    t1, t2 = by_id[3.0], by_id[1.0]
    assert t1.case == t2.case == "case-oom"                      # from the agentstats.case span attribute
    assert t1.checks == {"called_count": True, "count_for_a_day": True, "called_dup": True}
    assert t2.checks == {"called_count": False, "count_for_a_day": False, "called_dup": True}
    assert t1.metrics["input_tokens"] == 1200 and t1.metrics["tool_calls"] == 2
    assert all(r.passed is None for r in rows)                  # no verdicts: rules-only


def test_langfuse_traces_read_tool_observations_and_python_repr_inputs():
    rows = {r.case: r for r in traces.convert([FIX / "langfuse_traces.json"], "langfuse", rules=RULES)}
    assert rows["lf1"].checks == {"called_count": True, "count_for_a_day": True, "called_dup": True}
    assert rows["lf2"].checks["called_count"] is False
    assert rows["lf1"].metrics["cost_usd"] == 0.004 and rows["lf1"].metrics["input_tokens"] == 900
    assert "model:model-x" in rows["lf1"].tags


def test_verdicts_join_by_trace_id(tmp_path):
    v = tmp_path / "v.jsonl"
    v.write_text('{"trace_id": "lf1", "passed": true, "case": "oom", "run": 2}\n'
                 '{"trace_id": "lf2", "passed": false, "case": "oom", "run": 1}\n')
    rows = traces.convert([FIX / "langfuse_traces.json"], "langfuse", verdicts=traces.load_verdicts(v))
    assert sorted((r.case, r.run, r.passed) for r in rows) == [("oom", 1, False), ("oom", 2, True)]


def test_rules_only_rows_gate_on_behaviour_and_are_not_unhealthy():
    rows = traces.convert([FIX / "langfuse_traces.json"], "langfuse", rules=RULES)
    pol = parse({"prod": {"require": {"min_rule_rate": 0.95, "no_significant_drop": 0.0}}})
    g = gate.evaluate(pol.tiers["prod"], pol, rows)
    results = {r["requirement"]: r["result"] for r in g["rows"]}
    assert results["min_rule_rate"] in (gate.FAIL, gate.INCONCLUSIVE)   # count called in 1 of 2 traces
    assert results["no_significant_drop"] == gate.INCONCLUSIVE           # can't judge accuracy without verdicts
    assert gate.UNHEALTHY not in results.values() and g["rules_only"] is True
    assert "rules-only" in gate.render(g)


def test_cli_convert_traces(tmp_path, capsys):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(RULES))
    out = tmp_path / "rows.jsonl"
    assert cli.main(["convert-traces", str(FIX / "otlp_traces.json"), "--format", "otlp", "--rules", str(rules),
                     "--out", str(out)]) == 0
    assert "2 traces, 0 with a verdict (rules-only)" in capsys.readouterr().out
    assert len(out.read_text().splitlines()) == 2
