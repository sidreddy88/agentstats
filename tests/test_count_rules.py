"""max_calls and max_repeats: over-retrieval and tool flailing as deterministic rules."""
from __future__ import annotations

import json

import pytest

from agentstats import checks, monitor


def call(name, **args):
    return {"name": name, "args": args}


RUN = [call("grep", pattern="foo"), call("read", path="a.js"), call("grep", pattern="bar"),
       call("grep", pattern=" FOO "), call("read", path="a.js"), call("submit")]


def test_max_calls_counts_one_tool_or_all():
    assert checks.check({"name": "x", "max_calls": {"tool": "grep", "max": 3}}, RUN)
    assert not checks.check({"name": "x", "max_calls": {"tool": "grep", "max": 2}}, RUN)
    assert checks.check({"name": "x", "max_calls": {"tool": "*", "max": 6}}, RUN)
    assert not checks.check({"name": "x", "max_calls": {"tool": "*", "max": 5}}, RUN)
    assert checks.check({"name": "x", "max_calls": {"tool": "never_used", "max": 0}}, RUN)


def test_max_repeats_counts_identical_calls_after_normalising_case_and_whitespace():
    # grep "foo" again as " FOO " is a repeat; read a.js twice is a repeat; grep "bar" is not
    assert not checks.check({"name": "x", "max_repeats": {"tool": "*", "max": 1}}, RUN)
    assert checks.check({"name": "x", "max_repeats": {"tool": "*", "max": 2}}, RUN)
    assert not checks.check({"name": "x", "max_repeats": {"tool": "grep", "max": 0}}, RUN)
    assert checks.check({"name": "x", "max_repeats": {"tool": "submit", "max": 0}}, RUN)
    assert checks.check({"name": "x", "max_repeats": {"tool": "*", "max": 0}}, [call("a", k=1), call("a", k=2)])


@pytest.mark.parametrize("spec", [{"tool": "grep"}, {"tool": "grep", "max": -1}, {"tool": "grep", "max": True},
                                  {"max": 3}, "grep"])
def test_bad_specs_are_refused(tmp_path, spec):
    f = tmp_path / "rules.json"
    f.write_text(json.dumps([{"name": "x", "max_calls": spec}]))
    with pytest.raises(ValueError, match="max_calls needs"):
        checks.load_rules(f)
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"agent": "A", "rules": [{"name": "x", "max_calls": spec}]}))
    with pytest.raises(ValueError, match="max_calls needs"):
        monitor.load_configs(d, {"default": monitor.Adapter()})


def test_monitor_applies_count_rules_to_traced_runs(tmp_path):
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"agent": "A", "rules": [
        {"name": "grep_budget", "max_calls": {"tool": "grep", "max": 2}},
        {"name": "no_flailing", "max_repeats": {"tool": "*", "max": 0}}]}))
    cfg = monitor.load_configs(d, {"default": monitor.Adapter()})[0]
    trace = {"trace_id": "t1", "model": "m", "cost_usd": None, "agent": {"output": "{}"},
             "calls": [{"name": c["name"], "input": c["args"], "output": ""} for c in RUN]}
    row = monitor.to_row(trace, cfg, monitor.Adapter())
    assert row["checks"] == {"grep_budget": False, "no_flailing": False}
