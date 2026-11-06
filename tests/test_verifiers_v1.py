"""verifiers v1 traces.jsonl -> agentstats rows. Fixtures follow verifiers/v1/trace.py's serialized Trace."""
import json

from agentstats import cli, verifiers_v1


def trace(key="t1", attempt=1, score=1.0, ok=True, errors=(), dropped=False, timeout=False, tools=("bash",)):
    nodes = [{"message": {"role": "system", "content": "s"}, "parent": None, "sampled": False},
             {"message": {"role": "user", "content": "do it"}, "parent": 0, "sampled": False},
             {"message": {"role": "assistant", "content": "",
                          "tool_calls": [{"namespace": None, "name": t, "arguments": json.dumps({"cmd": "ls"})}
                                         for t in tools]}, "parent": 1, "sampled": True},
             {"message": {"role": "tool", "name": "bash", "content": "ok"}, "parent": 2, "sampled": False}]
    return {"version": 1, "id": f"{key}-{attempt}", "attempt": attempt,
            "task": {"type": "HarborTask", "key": key, "hash": "h" + key, "data": {}},
            "agent": {"name": "codex", "config": {}, "runtime": "prime", "trainable": False},
            "nodes": nodes, "calls": [{"node": 2, "model": "glm-4.5-air", "usage": None, "error": None}],
            "rewards": {"reward": {"score": score, "weight": 1.0}} if score is not None else {},
            "metrics": {"tests_passed": 3.0, "unused": None}, "ok": ok, "errors": list(errors),
            "dropped": dropped, "is_timeout": timeout, "num_input_tokens": 1200, "num_output_tokens": 300}


def test_a_passing_and_a_failing_rollout():
    a = verifiers_v1.attempt(trace(score=1.0))
    assert (a.case, a.run, a.passed, a.infra_error, a.arm) == ("t1", 1, True, None, "codex")
    assert a.metrics["input_tokens"] == 1200 and a.metrics["tool_calls"] == 1 and a.metrics["metric_tests_passed"] == 3
    assert "metric_unused" not in a.metrics and "model:glm-4.5-air" in a.tags and "harness:codex" in a.tags
    assert verifiers_v1.attempt(trace(score=0.0)).passed is False
    assert verifiers_v1.attempt(trace(score=0.6), pass_threshold=0.5).passed is True


def test_weighted_rewards_sum_like_trace_reward():
    t = trace()
    t["rewards"] = {"a": {"score": 1.0, "weight": 0.5}, "b": {"score": 1.0, "weight": 0.5}, "c": None}
    assert verifiers_v1.attempt(t).metrics["reward"] == 1.0


def test_errors_drops_and_timeouts_are_infrastructure_not_failures():
    err = verifiers_v1.attempt(trace(ok=False, errors=[{"type": "SandboxError", "message": "boot"}]))
    assert err.infra_error == "SandboxError" and err.passed is None and not err.valid
    assert verifiers_v1.attempt(trace(dropped=True)).infra_error == "Dropped"
    assert verifiers_v1.attempt(trace(timeout=True)).infra_error == "Timeout"
    assert verifiers_v1.attempt(trace(score=None)).infra_error == "NoReward"


def test_rules_check_the_tool_calls():
    rules = [{"name": "ran_tests", "called": "run_tests"}]
    assert verifiers_v1.attempt(trace(tools=("bash", "run_tests")), rules=rules).checks == {"ran_tests": True}
    assert verifiers_v1.attempt(trace(tools=("bash",)), rules=rules).checks == {"ran_tests": False}


def test_cli_converts_a_run_directory(tmp_path):
    run = tmp_path / "tb2--glm--codex--ab12"
    run.mkdir()
    (run / "traces.jsonl").write_text("".join(json.dumps(trace(key=f"t{k}", attempt=a, score=float(k % 2)))
                                             + "\n" for k in range(4) for a in (1, 2)))
    out = tmp_path / "rows.jsonl"
    assert cli.main(["convert", "--format", "verifiers", str(run), "--out", str(out)]) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 8 and {r["run"] for r in rows} == {1, 2} and sum(r["passed"] for r in rows) == 4


def test_a_file_that_isnt_verifiers_says_where(tmp_path):
    import pytest
    f = tmp_path / "traces.jsonl"
    f.write_text(json.dumps({"unrelated": 1}) + "\n")
    with pytest.raises(ValueError, match=r"traces.jsonl:1: not a verifiers v1 trace"):
        verifiers_v1.load(f)
