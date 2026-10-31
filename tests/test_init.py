"""agentstats init: traces -> agents found -> suggested rules accepted or not -> configs and workflows written.
Synthetic Langfuse rows and an OTLP export; no network."""
from __future__ import annotations

import datetime as dt
import json

from agentstats import cli, initialize, monitor

START = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)
END = START + dt.timedelta(days=14)


def lf_get(n=40):
    """SupportAgent: always get_order then reply; RouterAgent: 5 runs (too few for rules)."""
    agents, tools = [], []
    for i in range(n):
        agents.append({"traceId": f"s{i}", "name": "SupportAgent", "environment": "production",
                       "output": json.dumps({"action": "reply"})})
        tools += [{"traceId": f"s{i}", "name": "get_order", "startTime": "1", "input": "{}", "output": "ok"},
                  {"traceId": f"s{i}", "name": "reply", "startTime": "2", "input": "{}", "output": "sent"}]
    agents += [{"traceId": f"r{i}", "name": "RouterAgent", "environment": "production", "output": "{}"}
               for i in range(5)]

    def get(params):
        assert "basic" in params["fields"] or "model" in params["fields"]   # "core" alone has no observation name
        rows = {"AGENT": agents, "TOOL": tools, "GENERATION": []}.get(params["type"], [])
        names = [f["value"] for f in json.loads(params["filter"]) if f["column"] == "name"]
        return {"data": [r for r in rows if not names or r["name"] == names[0]], "meta": {}}
    return get


def test_finds_agents_asks_about_each_rule_and_writes_configs_and_workflows(tmp_path):
    answers = iter(["y", "n", "n", "n", "n", "n"])
    asked = []

    def fake_input(prompt):
        asked.append(prompt)
        return next(answers)
    res = initialize.run(tmp_path, monitor.LangfuseSource(START, END, get=lf_get()), input_fn=fake_input,
                         log=lambda *_: None)
    assert res["agents"] == {"SupportAgent": 40, "RouterAgent": 5}
    assert res["accepted"] == 1 and len(asked) == res["proposed"] >= 2
    support = json.loads((tmp_path / "agentstats/agents/support.json").read_text())
    assert support["agent"] == "SupportAgent" and len(support["rules"]) == 1
    assert support["rules"][0]["use"] == "monitor"                         # production runs are ungraded
    router = json.loads((tmp_path / "agentstats/agents/router.json").read_text())
    assert router == {"agent": "RouterAgent", "rules": []}                  # monitored anyway: counts, errors
    wf = (tmp_path / ".github/workflows/agentstats-monitor.yml").read_text()
    assert "${{ secrets.LANGFUSE_PUBLIC_KEY }}" in wf and "agentstats monitor --config-dir agentstats/agents" in wf
    assert "agentstats==" in wf and "${{ github.token }}" in wf
    for f in ("agentstats.toml", ".github/workflows/agent-evals-smoke.yml"):
        assert (tmp_path / f).exists()
    assert any("too few" in a for a in res["advice"])
    # the configs it wrote are ones monitor reads
    assert len(monitor.load_configs(tmp_path / "agentstats/agents", {"default": monitor.Adapter()})) == 2


def test_existing_files_are_left_alone_and_rerunning_keeps_accepted_rules(tmp_path):
    (tmp_path / "agentstats.toml").write_text("# mine\n")
    src = monitor.LangfuseSource(START, END, get=lf_get())
    first = initialize.run(tmp_path, src, accept_all=True, log=lambda *_: None)
    assert "agentstats.toml" in first["skipped"] and (tmp_path / "agentstats.toml").read_text() == "# mine\n"
    n_rules = len(json.loads((tmp_path / "agentstats/agents/support.json").read_text())["rules"])
    second = initialize.run(tmp_path, src, accept_all=True, log=lambda *_: None)
    assert second["proposed"] == 0                                          # already configured: not re-proposed
    assert len(json.loads((tmp_path / "agentstats/agents/support.json").read_text())["rules"]) == n_rules
    assert ".github/workflows/agentstats-monitor.yml" in second["skipped"]


def test_no_agents_is_a_clear_error(tmp_path):
    def empty(params):
        return {"data": [], "meta": {}}
    try:
        initialize.run(tmp_path, monitor.LangfuseSource(START, END, get=empty), log=lambda *_: None)
    except ValueError as e:
        assert "observation-type" in str(e)
    else:
        raise AssertionError("expected an error")


def test_cli_init_from_an_otlp_export(tmp_path, capsys):
    def attr(k, v):
        return {"key": k, "value": {"stringValue": v}}
    base = int(START.timestamp() * 1e9)
    spans = []
    for i in range(25):
        t = base + i * 10**10
        spans += [{"traceId": f"o{i}", "spanId": "a", "name": "invoke_agent Helper", "startTimeUnixNano": str(t),
                   "endTimeUnixNano": str(t + 10), "attributes": [attr("gen_ai.operation.name", "invoke_agent"),
                                                                  attr("gen_ai.agent.name", "Helper")]},
                  {"traceId": f"o{i}", "spanId": "b", "name": "execute_tool search", "startTimeUnixNano": str(t + 1),
                   "endTimeUnixNano": str(t + 2), "attributes": [attr("gen_ai.operation.name", "execute_tool"),
                                                                 attr("gen_ai.tool.name", "search"),
                                                                 attr("gen_ai.agent.name", "Helper")]}]
    (tmp_path / "t.json").write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}))
    code = cli.main(["init", "--dir", str(tmp_path / "repo"), "--otlp", str(tmp_path / "t.json"), "--all-spans",
                     "--yes"])
    out = capsys.readouterr().out
    assert code == 0 and "Helper (25 runs)" in out and "Next steps" in out
    cfg = json.loads((tmp_path / "repo/agentstats/agents/helper.json").read_text())
    assert [r["called"] for r in cfg["rules"]] == ["search"]
    assert "REPLACE" in (tmp_path / "repo/.github/workflows/agentstats-monitor.yml").read_text()
    assert cli.main(["monitor", "--config-dir", str(tmp_path / "repo/agentstats/agents"), "--otlp",
                     str(tmp_path / "t.json"), "--all-spans", "--out", str(tmp_path / "mon")]) == 0
    assert "25 runs" in (tmp_path / "mon/report.md").read_text()



def test_help_lists_every_command_once_in_a_group(capsys):
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    grouped = [n for _, names in cli.GROUPS for n in names]
    assert len(grouped) == len(set(grouped)) and "Gate and compare:" in out
    assert "Other:" not in out                       # a new command must be put in a group
    assert all(f"  {n} " in out for n in grouped)
