"""verified: every write to the named tools was read back by whatever executed it."""
from __future__ import annotations

import json

import pytest

from agentstats import adapters, checks, inspect_logs, traces

RULE = {"name": "writes_verified", "verified": {"tool": ["suspend_user", "create_user"]}}


def call(name, result=None):
    return {"name": name, "args": {}, "result": result}


@pytest.mark.parametrize("calls, ok", [
    ([call("get_user", "dave"), call("suspend_user", {"verified": True})], True),
    ([call("suspend_user", {"verified": False})], False),                     # read back: not there
    ([call("suspend_user", {"ok": True})], False),                            # never read back
    ([call("suspend_user", None)], False),                                    # no result recorded at all
    ([call("suspend_user", '{"verified": true}')], True),                     # results recorded as JSON text
    ([call("suspend_user", "done")], False),                                  # text that isn't JSON
    ([call("suspend_user", {"verified": None})], True),                       # read-back couldn't run: skipped
    ([call("suspend_user", {"verified": True}), call("create_user", {"verified": False})], False),
    ([call("get_user", "dave")], True),                                       # no writes: nothing to verify
])
def test_each_write_must_have_been_read_back(calls, ok):
    assert checks.check(RULE, calls) is ok


def test_a_star_covers_every_tool_and_one_name_is_enough():
    assert not checks.check({"name": "x", "verified": {"tool": "*"}}, [call("get_user", {"ok": True})])
    assert checks.check({"name": "x", "verified": {"tool": "suspend_user"}}, [call("suspend_user", {"verified": True})])


@pytest.mark.parametrize("spec", [{}, {"tool": []}, {"tool": 3}, {"tool": ["a", 3]}, "suspend_user"])
def test_bad_specs_are_refused(tmp_path, spec):
    f = tmp_path / "rules.json"
    f.write_text(json.dumps([{"name": "x", "verified": spec}]))
    with pytest.raises(ValueError):
        checks.load_rules(f)


def test_inspect_results_are_matched_to_their_calls():
    sample = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": "suspend_user", "arguments": {"user": "dave"}},
                                             {"id": "c2", "function": "create_user", "arguments": {"user": "erin"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": '{"verified": false}'},
        {"role": "tool", "tool_call_id": "c1", "content": '{"verified": true}'}]}
    calls = inspect_logs.tool_calls(sample)
    assert [c["result"] for c in calls] == ['{"verified": true}', '{"verified": false}']
    assert not checks.check(RULE, calls)


def test_atif_observations_are_matched_to_their_calls():
    traj = {"steps": [{"source": "agent",
                       "tool_calls": [{"tool_call_id": "t1", "function_name": "suspend_user", "arguments": {}}],
                       "observation": {"results": [{"source_call_id": "t1", "content": '{"verified": true}'}]}}]}
    calls = adapters.tool_calls(traj)
    assert calls[0]["result"] == '{"verified": true}' and checks.check(RULE, calls)


def test_langfuse_tool_output_is_the_result(tmp_path):
    f = tmp_path / "traces.json"
    f.write_text(json.dumps([{"id": "tr1", "observations": [
        {"type": "TOOL", "name": "suspend_user", "input": {"user": "dave"}, "output": {"verified": False},
         "startTime": "2026-11-15T00:00:00Z"}]}]))
    calls = traces.langfuse_traces(f)["tr1"]["calls"]
    assert calls[0]["result"] == {"verified": False} and not checks.check(RULE, calls)


def test_monitor_reads_verified_from_langfuse_tool_output():
    """Weekly monitoring from production traces: the gateway's read-back result arrives as the tool's JSON output."""
    from datetime import UTC, datetime

    from agentstats import monitor
    agents = [{"traceId": t, "name": "AccessAgent", "environment": "production", "projectId": "p1",
               "input": "{}", "output": "{}", "latency": 1.0, "level": "DEFAULT"} for t in ("ok", "bad")]
    tools = [{"traceId": "ok", "name": "suspend_user", "startTime": "2026-11-15T00:00:01Z", "input": "{}",
              "output": json.dumps({"verified": True})},
             {"traceId": "bad", "name": "suspend_user", "startTime": "2026-11-15T00:00:01Z", "input": "{}",
              "output": json.dumps({"verified": False})}]

    def get(params):
        return {"data": {"AGENT": agents, "TOOL": tools, "GENERATION": []}[params["type"]], "meta": {}}
    cfg = {"agent": "AccessAgent", "key": "AccessAgent", "rules": [RULE]}
    src = monitor.LangfuseSource(datetime(2026, 11, 14, tzinfo=UTC), datetime(2026, 11, 16, tzinfo=UTC), get=get)
    rows = {r["case"]: r for r in (monitor.to_row(t, cfg, monitor.Adapter()) for t in src.traces(cfg))}
    assert rows["ok"]["checks"]["writes_verified"] is True
    assert rows["bad"]["checks"]["writes_verified"] is False
