"""agentstats monitor: configs + traces (Langfuse API pages or an OTLP export) -> rows, rule summaries, judge, review
sheet, candidates, suggestions, report. Synthetic traces with known answers; no network."""
from __future__ import annotations

import datetime as dt
import json
import re
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
    assert "**SupportAgent / production: runs fell 93%" in text
    assert re.search(r"\| SupportAgent / production \| 2 \| 30 \| [\d.]+ \| [\d.]+ \| ⚠️ \|", text)
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
    assert re.search(r"\| SupportAgent / eval \| 0 \| 30 \| n/a \| [\d.]+ \|  \|", text)
    assert "No agent's run count fell" in text


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


def test_review_instructions_from_the_config_head_the_review_sheet(tmp_path, monkeypatch):
    rubric = tmp_path / "rubric.json"
    rubric.write_text(json.dumps([{"name": "ok", "question": "OK?", "pass": "yes"}]))
    cfg_dir = write_configs(tmp_path / "cfg", rubric=str(rubric), review_n=4,
                            review_instructions="Check every stated count against the tool results.")
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setattr(judge_run, "http_chat", lambda *a, **k: (lambda messages: '{"ok": "yes"}'))
    monitor.run(cfg_dir, tmp_path / "out", source=monitor.LangfuseSource(START, END, get=fake_get(*lf_runs())),
                start=START, end=END, judge_settings=monitor.JudgeSettings("m", key_env="FAKE_KEY"),
                log=lambda *_: None)
    md = (tmp_path / "out" / "support" / "review" / "REVIEW.md").read_text()
    assert md.index("**What to check:** Check every stated count") < md.index("## 1.")


# ---- models and spend --------------------------------------------------------------------------------------------

def test_generation_tokens_use_totals_or_sum_the_buckets():
    assert monitor._generation_tokens({"inputUsage": 1883, "outputUsage": 559}) == (1883, 559)
    assert monitor._generation_tokens({"usageDetails": {"input": 4, "input_cached_tokens": 1655, "output": 282,
                                                        "output_reasoning_tokens": 277}}) == (1659, 559)
    assert monitor._generation_tokens({}) == (0, 0)


def priced_runs():
    """SupportAgent on big-model: 30 runs, each 10,000 input + 1,000 output tokens at $0.05 recorded."""
    agents, tools, _ = lf_runs(30)
    gens = [{"traceId": a["traceId"], "model": "big-model", "totalCost": 0.05, "inputUsage": 10_000,
             "outputUsage": 1_000} for a in agents]
    return agents, tools, gens


MODELS = {"prices_per_million": {"big-model": {"input": 3.0, "output": 15.0},
                                 "small-model": {"input": 0.3, "output": 1.5},
                                 "pricier-model": {"input": 10.0, "output": 50.0}}}


def test_inventory_and_candidates_reprice_actual_tokens(tmp_path):
    (tmp_path / "evals.jsonl").write_text("{}\n")
    cfg_dir = write_configs(tmp_path / "cfg", eval_set=str(tmp_path / "evals.jsonl"))
    src = monitor.LangfuseSource(START, END, get=fake_get(*priced_runs()))
    report = monitor.run(cfg_dir, tmp_path / "out", source=src, start=START, end=END, models=MODELS,
                         log=lambda *_: None)
    text = report.read_text()
    assert text.index("## Models and spend") < text.index("## SupportAgent")
    assert "| SupportAgent | big-model | production | 30 | $1.50 | $0.0500 |" in text and "| 100% |" in text
    # small-model on the same tokens: 30 x (10,000 x 0.3 + 1,000 x 1.5) / 1e6 = $0.135; pricier-model isn't listed
    cands = json.loads((tmp_path / "out" / "candidates_models.json").read_text())
    assert [c["candidate"] for c in cands] == ["small-model"]
    assert cands[0]["estimate"] == pytest.approx(0.135) and cands[0]["saving"] == pytest.approx(1.365)
    assert "test it: `agentstats preregister`" in text


def test_a_candidate_without_an_eval_set_says_build_one_first(tmp_path):
    report = monitor.run(write_configs(tmp_path / "cfg"), tmp_path / "out",
                         source=monitor.LangfuseSource(START, END, get=fake_get(*priced_runs())), start=START, end=END,
                         models=MODELS, log=lambda *_: None)
    assert "no eval set: build one first" in report.read_text()


def test_without_a_models_file_the_inventory_still_shows(tmp_path):
    report = monitor.run(write_configs(tmp_path / "cfg"), tmp_path / "out",
                         source=monitor.LangfuseSource(START, END, get=fake_get(*priced_runs())), start=START, end=END,
                         log=lambda *_: None)
    text = report.read_text()
    assert "## Models and spend" in text and "Pass `--models models.json`" in text


def test_unrecorded_cost_is_priced_from_tokens_when_the_model_is_listed():
    rows = [{"tags": ["model:small-model"], "input_tokens": 1_000_000, "output_tokens": 0}]
    inv = monitor.model_inventory({"A": rows}, MODELS)
    assert inv[0]["spend"] == pytest.approx(0.3) and inv[0]["estimated"]


def test_models_file_is_validated(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"prices_per_million": {"x": {"input": "cheap"}}}))
    with pytest.raises(ValueError, match="numeric input and output"):
        monitor.load_models(bad)
    bad.write_text("{}")
    with pytest.raises(ValueError, match="prices_per_million"):
        monitor.load_models(bad)


def test_eval_traffic_is_inventoried_but_never_a_switch_candidate():
    rows = [{"tags": ["model:big-model", "source:eval"], "cost_usd": 1.0, "input_tokens": 10_000,
             "output_tokens": 1_000}]
    inv = monitor.model_inventory({"A": rows}, MODELS)
    assert inv[0]["source"] == "eval" and inv[0]["spend"] == 1.0
    assert monitor.downgrade_candidates(inv, MODELS, {"A": None}) == []



# ---- tool calls per run vs the window before ------------------------------------------------------------------------

def test_tool_call_check_flags_a_real_rise_not_noise_or_small_samples():
    before = [2, 2, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2]
    rose = monitor.tool_call_check([6, 7, 6, 5, 7, 6, 6, 7, 5, 6, 6, 7], before)
    assert rose["flag"] and "rose from 2.2 to 6.2" in rose["note"]
    assert not monitor.tool_call_check([2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 3], before)["flag"]   # same distribution
    assert not monitor.tool_call_check([9, 9, 9], before)["flag"]                              # too few runs now
    assert not monitor.tool_call_check([6] * 12, None)["flag"]                                 # no window before
    fell = monitor.tool_call_check([1] * 12, before)
    assert not fell["flag"] and fell["now"] == 1.0                                             # fewer calls: fine


def test_report_flags_tool_calls_per_run_rising(tmp_path):
    agents, tools, gens = lf_runs(30)
    extra = [{"traceId": a["traceId"], "name": "get_order", "startTime": "3", "input": "{}", "output": "x"}
             for a in agents for _ in range(4)]                  # this window: 4 more calls per run

    def get(params):
        since = next(f["value"] for f in json.loads(params["filter"]) if f["operator"] == ">=")
        current = since >= START.isoformat()
        rows = {"AGENT": agents, "TOOL": tools + (extra if current else []), "GENERATION": gens}[params["type"]]
        return {"data": rows, "meta": {}}
    text = monitor.run(write_configs(tmp_path / "cfg"), tmp_path / "out",
                       source=monitor.LangfuseSource(START, END, get=get), start=START, end=END,
                       log=lambda *_: None).read_text()
    assert "**SupportAgent / production: tool calls per run rose from 1.8 to 5.8" in text



# ---- model retirements ------------------------------------------------------------------------------------------

def test_retirement_warnings_for_models_still_in_use():
    inv = [{"agent": "A", "model": "old-model", "source": "production", "runs": 12},
           {"agent": "A", "model": "soon-model", "source": "production", "runs": 5},
           {"agent": "B", "model": "later-model", "source": "production", "runs": 9},
           {"agent": "B", "model": "fine-model", "source": "eval", "runs": 3}]
    models = {"prices_per_million": {}, "retirements": {"old-model": "2026-09-30", "soon-model": "2026-11-20",
                                                        "later-model": "2027-06-01", "fine-model": "2026-10-01"}}
    w = monitor.retirement_warnings(inv, models, dt.date(2026, 11, 1))
    assert len(w) == 2                                    # fine-model is eval traffic, later-model is far off
    assert "old-model retired on 2026-09-30" in w[0] and "A (production, 12 runs)" in w[0]
    assert "soon-model retires in 19 days" in w[1]
    assert monitor.retirement_warnings(inv, None, dt.date(2026, 11, 1)) == []


def test_retirement_dates_are_validated(tmp_path):
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"prices_per_million": {"x": {"input": 1, "output": 2}}, "retirements": {"x": "soon"}}))
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        monitor.load_models(f)


def test_retirement_warning_heads_the_models_section(tmp_path):
    models = {**MODELS, "retirements": {"big-model": (START + dt.timedelta(days=20)).date().isoformat()}}
    text = monitor.run(write_configs(tmp_path / "cfg"), tmp_path / "out",
                       source=monitor.LangfuseSource(START, END, get=fake_get(*priced_runs())), start=START, end=END,
                       models=models, log=lambda *_: None).read_text()
    models_section = text[text.index("## Models and spend"):]
    assert models_section.index("big-model retires in 13 days") < models_section.index("| agent | model |")
