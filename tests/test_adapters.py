import json

from agentstats import adapters, cli, records


def _trial(job, name, task, reward=None, exc=None, calls=()):
    t = job / name
    (t / "agent").mkdir(parents=True)
    (t / "config.json").write_text("{}")
    result = {"task_name": task, "exception_info": {"exception_type": exc} if exc else None,
              "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
              "agent_result": {"n_input_tokens": 1000, "n_cache_tokens": 800, "n_output_tokens": 50, "cost_usd": 0.02},
              "agent_execution": {"started_at": "2026-10-10T23:31:10Z", "finished_at": "2026-10-10T23:32:10Z"},
              "agent_info": {"name": "mini-swe-agent"}}
    (t / "result.json").write_text(json.dumps(result))
    steps = [{"source": "user"}, {"source": "agent", "tool_calls": [
        {"function_name": n, "arguments": json.dumps(a)} for n, a in calls]}]
    (t / "agent" / "trajectory.json").write_text(json.dumps({"steps": steps}))


def make_job(tmp_path):
    job = tmp_path / "jobs-x" / "2026-10-10__16-30-57"
    job.mkdir(parents=True)
    (job / "result.json").write_text("{}")
    _trial(job, "t1__abc", "task-1", reward=1.0, calls=[("bash", {"cmd": "ls"}), ("submit", {})])
    _trial(job, "t2__def", "task-2", reward=0.0, calls=[("bash", {"cmd": "rm"})])
    _trial(job, "t3__ghi", "task-3", exc="AgentTimeoutError")
    return job


def test_harbor_job(tmp_path):
    job = make_job(tmp_path)
    assert adapters.is_harbor_job(job)
    atts = {a.case: a for a in records.load(job)}
    assert atts["task-1"].passed is True and atts["task-2"].passed is False
    assert atts["task-3"].infra_error == "AgentTimeoutError" and atts["task-3"].passed is None
    assert atts["task-1"].metrics["cost_usd"] == 0.02 and atts["task-1"].metrics["latency_s"] == 60.0
    assert atts["task-1"].metrics["tool_calls"] == 2


def test_check_command(tmp_path):
    job = make_job(tmp_path)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps([{"name": "submitted", "called": "submit"},
                                 {"name": "no_rm", "args": {"tool": "bash", "match": {"cmd": "rm"}}}]))
    out = tmp_path / "out.jsonl"
    assert cli.main(["check", str(job), "--rules", str(rules), "--out", str(out)]) == 0
    rows = {r["case"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert rows["task-1"]["checks"] == {"submitted": True, "no_rm": False}
    assert rows["task-2"]["checks"] == {"submitted": False, "no_rm": True}
