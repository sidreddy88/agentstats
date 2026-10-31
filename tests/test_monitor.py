"""agentstats monitor: configs + traces (Langfuse API pages or an OTLP export) -> rows, rule summaries, judge, review
sheet, candidates, suggestions, report. Synthetic traces with known answers; no network."""
from __future__ import annotations

import datetime as dt
import json
import threading

import pytest

from agentstats import cli, judge_run, monitor

START = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)
END = START + dt.timedelta(days=7)


def write_configs(d, **extra):
    d.mkdir(parents=True, exist_ok=True)
    (d / "support.json").write_text(json.dumps({
        "agent": "SupportAgent", "group_by": "intent",
        "rules": [{"name": "looked_up_order", "called": "get_order", "use": "gate", "why": "from suggest"},
                  {"name": "order_before_refund", "before": ["get_order", "issue_refund"]},
                  {"name": "valid_action", "output": "action", "in": ["refund", "reply", "escalate"]}],
        **extra}))
    return d


def lf_runs(n=30):
    """Langfuse v2 observation rows: n SupportAgent runs; every 5th skips get_order; run 7 errors; run 9 answers
    with an action outside the allowed set."""
    agents, tools, gens = [], [], []
    for i in range(n):
        tid = f"t{i:02d}"
        out = {"action": "dance" if i == 9 else "refund", "intent": "refund"}
        agents.append({"traceId": tid, "name": "SupportAgent", "environment": "production", "projectId": "p1",
                       "input": json.dumps({"message": f"where is order {i}", "intent": "refund"}),
                       "output": json.dumps(out), "latency": 2.5, "level": "ERROR" if i == 7 else "DEFAULT",
                       "statusMessage": "rate limited" if i == 7 else None})
        if i % 5:
            tools.append({"traceId": tid, "name": "get_order", "startTime": "2026-10-02T00:00:01Z",
                          "input": json.dumps({"id": i}), "output": "shipped"})
        tools.append({"traceId": tid, "name": "issue_refund", "startTime": "2026-10-02T00:00:02Z",
                      "input": "{}", "output": "ok"})
        gens.append({"traceId": tid, "model": "small-model", "totalCost": 0.002})
    return agents, tools, gens


def fake_get(agents, tools, gens):
    def get(params):
        rows = {"AGENT": agents, "TOOL": tools, "GENERATION": gens}[params["type"]]
        if params["type"] == "AGENT":
            name = next(f["value"] for f in json.loads(params["filter"]) if f["column"] == "name")
            rows = [r for r in rows if r["name"] == name]
        return {"data": rows, "meta": {}}
    return get


def test_langfuse_runs_become_rows_with_rules_infra_and_no_answers(tmp_path):
    cfg = monitor.load_configs(write_configs(tmp_path / "cfg"), {"default": monitor.Adapter()})[0]
    src = monitor.LangfuseSource(START, END, get=fake_get(*lf_runs()))
    traces = src.traces(cfg)
    rows = {r["case"]: r for r in (monitor.to_row(t, cfg, monitor.Adapter()) for t in traces)}
    assert len(rows) == 30
    assert rows["t07"]["infra_error"] == "rate limited" and rows["t07"]["checks"] == {}
    assert rows["t05"]["checks"]["looked_up_order"] is False
    assert rows["t05"]["checks"]["order_before_refund"] is False
    assert rows["t01"]["checks"] == {"looked_up_order": True, "order_before_refund": True, "valid_action": True}
    assert rows["t09"]["checks"]["valid_action"] is False
    assert "source:production" in rows["t01"]["tags"] and "model:small-model" in rows["t01"]["tags"]
    assert "error_type:" not in " ".join(rows["t01"]["tags"])
    assert rows["t01"]["cost_usd"] == 0.002 and rows["t01"]["latency_s"] == 2.5
    assert src.link(traces).endswith("/project/p1/traces/{case}")


def test_fetch_follows_the_cursor():
    pages = [{"data": [{"id": 1}], "meta": {"cursor": "c2"}}, {"data": [{"id": 2}], "meta": {}}]
    seen = []

    def get(params):
        seen.append(params.get("cursor"))
        return pages[len(seen) - 1]
    assert [r["id"] for r in monitor.fetch_observations("TOOL", START, END, get=get)] == [1, 2]
    assert seen == [None, "c2"]


def otlp_export(path):
    def attr(k, v):
        key = "intValue" if isinstance(v, int) else "stringValue"
        return {"key": k, "value": {key: str(v) if key == "intValue" else v}}

    def span(tid, sid, name, t0, t1, attrs, status=None):
        return {"traceId": tid, "spanId": sid, "name": name, "startTimeUnixNano": str(t0), "endTimeUnixNano": str(t1),
                "attributes": [attr(k, v) for k, v in attrs.items()], **({"status": status} if status else {})}
    base = int(START.timestamp() * 1e9) + 10**12
    spans = []
    for i in range(4):
        tid, t = f"o{i}", base + i * 10**10
        spans.append(span(tid, "a", "invoke_agent SupportAgent", t, t + 3 * 10**9,
                          {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "SupportAgent",
                           "output.value": json.dumps({"action": "reply"})},
                          {"code": 2, "message": "boom"} if i == 3 else None))
        if i != 2:
            spans.append(span(tid, "b", "execute_tool get_order", t + 1, t + 2,
                              {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_order",
                               "gen_ai.tool.call.arguments": json.dumps({"id": i}), "gen_ai.tool.call.result": "ok"}))
        spans.append(span(tid, "c", "chat", t + 3, t + 4, {"gen_ai.operation.name": "chat",
                                                         "gen_ai.response.model": "m-1"}))
    doc = {"resourceSpans": [{"resource": {"attributes": [attr("deployment.environment.name", "production")]},
                              "scopeSpans": [{"spans": spans}]}]}
    path.write_text(json.dumps(doc))
    return path


def test_otlp_export_gives_the_same_rows(tmp_path):
    cfg = monitor.load_configs(write_configs(tmp_path / "cfg"), {"default": monitor.Adapter()})[0]
    traces = monitor.OtlpSource([otlp_export(tmp_path / "t.json")], START, END).traces(cfg)
    rows = {r["case"]: r for r in (monitor.to_row(t, cfg, monitor.Adapter()) for t in traces)}
    assert set(rows) == {"o0", "o1", "o2", "o3"}
    assert rows["o0"]["checks"]["looked_up_order"] is True and rows["o2"]["checks"]["looked_up_order"] is False
    assert rows["o3"]["infra_error"] == "boom"
    assert "source:production" in rows["o0"]["tags"] and "model:m-1" in rows["o0"]["tags"]
    assert rows["o0"]["latency_s"] == pytest.approx(3.0)
    assert monitor.OtlpSource([tmp_path / "t.json"], END, END + dt.timedelta(days=1)).traces(cfg) == []


def test_a_bad_config_or_unknown_adapter_is_refused(tmp_path):
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"agent": "A", "rules": [{"name": "x", "called": "t", "before": ["a", "b"]}]}))
    with pytest.raises(ValueError, match="exactly one"):
        monitor.load_configs(d, {"default": monitor.Adapter()})
    (d / "a.json").write_text(json.dumps({"agent": "A", "adapter": "nope"}))
    with pytest.raises(ValueError, match="unknown adapter"):
        monitor.load_configs(d, {"default": monitor.Adapter()})
    with pytest.raises(ValueError, match="no agent configs"):
        monitor.load_configs(tmp_path / "empty", {})


def test_user_adapters_are_loaded_from_a_file(tmp_path):
    f = tmp_path / "my_adapters.py"
    f.write_text("from agentstats.monitor import Adapter\n"
                 "class Strict(Adapter):\n"
                 "    conditions = {'is_refund': lambda t: 'refund' in str(t['agent'].get('input'))}\n"
                 "    def checks(self, trace):\n"
                 "        return {'answered_fast': (trace['agent'].get('latency') or 0) < 2}\n"
                 "    def source(self, trace):\n"
                 "        return 'eval'\n"
                 "ADAPTERS = {'strict': Strict()}\n")
    adapters = monitor.load_adapters(f)
    cfg = monitor.load_configs(write_configs(tmp_path / "cfg", adapter="strict"), adapters)[0]
    cfg["rules"][0]["when"] = "is_refund"
    t = monitor.LangfuseSource(START, END, get=fake_get(*lf_runs(3))).traces(cfg)[1]
    row = monitor.to_row(t, cfg, adapters["strict"])
    assert row["checks"]["answered_fast"] is False and row["checks"]["looked_up_order"] is True
    assert "source:eval" in row["tags"]
    bad = tmp_path / "bad.py"
    bad.write_text("X = 1\n")
    with pytest.raises(ValueError, match="ADAPTERS"):
        monitor.load_adapters(bad)


def test_full_run_writes_a_report_with_flags_judge_review_candidates_and_suggestions(tmp_path, monkeypatch):
    rubric = tmp_path / "rubric.json"
    rubric.write_text(json.dumps([{"name": "action_fits_request", "question": "Does the action fit?", "pass": "yes"}]))
    cfg_dir = write_configs(tmp_path / "cfg", rubric=str(rubric), review_n=5, judge_sample=10)
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setattr(judge_run, "http_chat", lambda *a, **k: (lambda messages: '{"action_fits_request": "yes"}'))
    report = monitor.run(cfg_dir, tmp_path / "out", source=monitor.LangfuseSource(START, END, get=fake_get(*lf_runs())),
                         start=START, end=END, judge_settings=monitor.JudgeSettings("m", key_env="FAKE_KEY"),
                         labels_dir="app/evals", log=lambda *_: None)
    text = report.read_text()
    assert "## SupportAgent" in text and "30 runs: production 30." in text
    assert "`looked_up_order` holds only" in text                  # 24 of 29: proven below 95%
    assert "rate limited (1)" in text
    assert "10 of 10 sampled runs judged" in text and "flip rate over 2 repeats: 0.0%" in text
    assert "## Suggestions" in text and "app/evals/<agent>_cases.jsonl" in text
    assert "where is order" not in text                             # raw input stays out of the report
    out = tmp_path / "out" / "support"
    assert (out / "review" / "review.csv").exists() and "where is order" in (out / "review" / "REVIEW.md").read_text()
    assert "open trace" in (out / "review" / "REVIEW.md").read_text()
    assert (out / "candidates.jsonl").exists() and (tmp_path / "out" / "proposal.json").exists()


def test_an_unstable_judge_is_reported_as_unreliable(tmp_path, monkeypatch):
    rubric = tmp_path / "rubric.json"
    rubric.write_text(json.dumps([{"name": "ok", "question": "OK?", "pass": "yes"}]))
    cfg_dir = write_configs(tmp_path / "cfg", rubric=str(rubric), review_n=5)
    seen: dict[str, int] = {}
    lock = threading.Lock()

    def flip(messages):                      # per item: yes the first time, no the second (grade runs in threads)
        with lock:
            n = seen[messages[-1]["content"]] = seen.get(messages[-1]["content"], 0) + 1
        return '{"ok": "yes"}' if n % 2 else '{"ok": "no"}'
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setattr(judge_run, "http_chat", lambda *a, **k: flip)
    report = monitor.run(cfg_dir, tmp_path / "out", source=monitor.LangfuseSource(START, END, get=fake_get(*lf_runs())),
                         start=START, end=END, judge_settings=monitor.JudgeSettings("m", key_env="FAKE_KEY"),
                         log=lambda *_: None)
    assert "**The judge is unreliable**" in report.read_text()


def test_no_runs_is_a_short_section(tmp_path):
    report = monitor.run(write_configs(tmp_path / "cfg"), tmp_path / "out",
                         source=monitor.LangfuseSource(START, END, get=fake_get([], [], [])), start=START, end=END,
                         log=lambda *_: None)
    text = report.read_text()
    assert "No runs in this window." in text and "No trajectories in this window." in text


def test_cli_monitor_on_an_otlp_export(tmp_path, capsys):
    otlp_export(tmp_path / "t.json")
    cfg = str(write_configs(tmp_path / "cfg"))
    code = cli.main(["monitor", "--config-dir", cfg, "--otlp", str(tmp_path / "t.json"), "--all-spans",
                     "--out", str(tmp_path / "out")])
    assert code == 0 and "wrote" in capsys.readouterr().out
    assert "4 runs: production 4." in (tmp_path / "out" / "report.md").read_text()
    assert cli.main(["monitor", "--config-dir", str(tmp_path / "cfg"), "--otlp", str(tmp_path / "t.json"),
                     "--out", str(tmp_path / "o2"), "--judge"]) == 2


# ---- run volume and cost per usable answer ----------------------------------------------------------------------

def test_volume_check_flags_real_drops_and_silence_not_noise():
    assert monitor.volume_check(0, 20)["flag"]                         # ran before, nothing now: tracing?
    assert monitor.volume_check(5, 30)["flag"]                         # big, significant drop
    small = monitor.volume_check(8, 14)                                # under half? no; and not significant
    assert not small["flag"]
    assert not monitor.volume_check(2, 6)["flag"]                      # too few runs before to tell
    assert not monitor.volume_check(40, 30)["flag"]
    assert monitor.volume_check(10, None) == {"current": 10, "previous": None, "flag": False, "p": None,
                                              "note": "no previous window"}


def test_report_opens_with_run_volume_and_flags_an_agent_that_went_quiet(tmp_path):
    agents, tools, gens = lf_runs(30)

    def get(params):                                                   # 30 runs in the window before, 2 now
        flt = json.loads(params["filter"])
        since = next(f["value"] for f in flt if f["operator"] == ">=")
        rows = {"AGENT": agents, "TOOL": tools, "GENERATION": gens}[params["type"]]
        if params["type"] == "AGENT":
            rows = rows if since < START.isoformat() else rows[:2]
        return {"data": rows, "meta": {}}
    report = monitor.run(write_configs(tmp_path / "cfg"), tmp_path / "out",
                         source=monitor.LangfuseSource(START, END, get=get), start=START, end=END,
                         log=lambda *_: None)
    text = report.read_text()
    assert text.index("## Run volume") < text.index("## SupportAgent")
    assert "**SupportAgent / production: runs fell 93%" in text and "| SupportAgent / production | 2 | 30 | ⚠️ |" in text
    assert json.loads((tmp_path / "out" / "volume.json").read_text())["SupportAgent / production"]["flag"] is True


def test_eval_traffic_volume_is_shown_but_never_flagged(tmp_path):
    agents, tools, gens = lf_runs(30)
    for a in agents:
        a["environment"] = "eval"

    def get(params):
        since = next(f["value"] for f in json.loads(params["filter"]) if f["operator"] == ">=")
        rows = {"AGENT": agents, "TOOL": tools, "GENERATION": gens}[params["type"]]
        if params["type"] == "AGENT" and since >= START.isoformat():
            rows = []
        return {"data": rows, "meta": {}}
    text = monitor.run(write_configs(tmp_path / "cfg"), tmp_path / "out",
                       source=monitor.LangfuseSource(START, END, get=get), start=START, end=END,
                       log=lambda *_: None).read_text()
    assert "| SupportAgent / eval | 0 | 30 |  |" in text and "No agent's run count fell" in text


def test_cost_per_usable_answer_charges_failed_runs_to_the_useful_ones():
    rows = [{"cost_usd": 1.0}, {"cost_usd": 1.0, "no_answer": "step_limit"}, {"cost_usd": 1.0, "infra_error": "x"},
            {"cost_usd": 1.0}]
    assert monitor._cost_per_useful(rows) == 2.0                       # $4 spent, 2 usable answers
    graded = [{"cost_usd": 1.0, "passed": True}, {"cost_usd": 3.0, "passed": False}]
    assert monitor._cost_per_useful(graded) == 4.0                     # graded: only passes count as useful
    assert monitor._cost_per_useful([{"cost_usd": 1.0, "no_answer": "x"}]) == float("inf")
    assert monitor._cost_per_useful([{}]) is None


def test_report_shows_cost_per_usable_answer(tmp_path):
    report = monitor.run(write_configs(tmp_path / "cfg"), tmp_path / "out",
                         source=monitor.LangfuseSource(START, END, get=fake_get(*lf_runs())), start=START, end=END,
                         log=lambda *_: None)
    # 30 runs at $0.002 (one errored); every non-errored run answered: $0.06 / 29
    assert "| cost per usable answer |" in report.read_text() and "$0.0021 |" in report.read_text()


def test_otlp_previous_window_counts(tmp_path):
    otlp_export(tmp_path / "t.json")                                   # 4 runs, all inside START..END
    cfg, ad = {"agent": "SupportAgent"}, monitor.Adapter()
    assert monitor.OtlpSource([tmp_path / "t.json"], START, END).previous_count(cfg, ad) == {}
    later = monitor.OtlpSource([tmp_path / "t.json"], END, END + (END - START))
    assert later.previous_count(cfg, ad) == {"production": 4} and later.traces(cfg) == []
    assert monitor.OtlpSource([tmp_path / "t.json"]).previous_count(cfg, ad) is None
