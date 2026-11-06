"""
Read verifiers v1 eval output (Prime Intellect's verifiers, the `verifiers.v1` namespace) as agentstats rows.

    agentstats convert --format verifiers outputs/<run-dir>/traces.jsonl --out rows.jsonl [--rules rules.json]

`uv run eval @ config.toml` writes one run directory with `traces.jsonl`: one serialized `Trace` per rollout, appended
as rollouts finish. Fields read (verifiers/v1/trace.py):
  task.key / task.hash      the case (a task's identity across rollouts)
  attempt                   the run number (from 1)
  rewards                   {name: {"score", "weight"} | null}: the reward is sum(score * weight), as Trace.reward
  metrics                   {name: float | null}: kept as metric_<name>
  ok, errors, dropped, is_timeout
                            an infrastructure failure, kept apart from pass rates (a dropped trace is one verifiers
                            itself leaves out of reward means)
  num_input_tokens, num_output_tokens, calls[].model, agent.name
  nodes[].message.tool_calls  tool calls ({namespace, name, arguments}), in graph order, for --rules checks
A rollout passes when its reward is at or above --pass-threshold (default 1.0, as for Harbor tasks, whose verifier
reward is recorded onto the trace).

Source: https://www.primeintellect.ai/blog/verifiers-v1 and github.com/PrimeIntellect-ai/verifiers (verifiers/v1/
trace.py, cli/output.py). Checked against the schema in the source, not yet against a real run's file: if your
traces.jsonl differs, the converter says which field it couldn't find.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentstats.checks import run_checks
from agentstats.records import Attempt


def _traces_file(path: str | Path) -> Path:
    p = Path(path)
    return p / "traces.jsonl" if p.is_dir() else p


def _reward(trace: dict) -> float | None:
    rewards = [r for r in (trace.get("rewards") or {}).values() if r]
    if not rewards:
        return None
    return sum(float(r.get("score", 0.0)) * float(r.get("weight", 1.0)) for r in rewards)


def tool_calls(trace: dict) -> list[dict]:
    """Tool calls in graph order as {"name", "args"} (the shape agentstats' rule checks expect). The name is
    namespace.name when a namespace is set."""
    calls = []
    for node in trace.get("nodes") or []:
        msg = (node or {}).get("message") or {}
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            name = c.get("name") or fn.get("name")
            if not name:
                continue
            ns = c.get("namespace")
            args: Any = c.get("arguments", fn.get("arguments"))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            calls.append({"name": f"{ns}.{name}" if ns else name, "args": args or {}})
    return calls


def _infra(trace: dict) -> str | None:
    if trace.get("dropped"):
        return "Dropped"
    errors = trace.get("errors") or []
    if errors and not trace.get("ok", False):
        return str(errors[0].get("type") or "Error")
    if trace.get("is_timeout"):
        return "Timeout"
    return None


def attempt(trace: dict, *, pass_threshold: float = 1.0, rules: list[dict] | None = None,
            arm: str | None = None) -> Attempt:
    task = trace.get("task") or {}
    case = str(task.get("key") or task.get("hash") or trace.get("id") or "")
    if not case:
        raise ValueError("verifiers trace has no task.key, task.hash or id")
    reward = _reward(trace)
    infra = _infra(trace)
    if infra is None and reward is None:
        infra = "NoReward"
    calls_ = trace.get("calls") or []
    model = next((c.get("model") for c in calls_ if c.get("model")), None)
    agent = (trace.get("agent") or {}).get("name")
    metrics: dict[str, float] = {}
    for src, dst in (("num_input_tokens", "input_tokens"), ("num_output_tokens", "output_tokens")):
        if trace.get(src) is not None:
            metrics[dst] = float(trace[src])
    tools = tool_calls(trace)
    metrics["tool_calls"] = float(len(tools))
    metrics["model_calls"] = float(len(calls_))
    if reward is not None:
        metrics["reward"] = reward
    for name, v in (trace.get("metrics") or {}).items():
        if v is not None:
            metrics[f"metric_{name}"] = float(v)
    tags = [t for t in (f"model:{model}" if model else None, f"harness:{agent}" if agent else None,
                        f"task_type:{task['type']}" if task.get("type") else None) if t]
    passed = None if infra else reward >= pass_threshold
    return Attempt(case=case, passed=passed, arm=arm or agent or model or "default",
                   run=int(trace.get("attempt") or 1), infra_error=infra, tags=tags,
                   checks=run_checks(rules, tools) if rules else {}, metrics=metrics)


def load(path: str | Path, *, pass_threshold: float = 1.0, rules: list[dict] | None = None,
         arm: str | None = None) -> list[Attempt]:
    f = _traces_file(path)
    out = []
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append(attempt(json.loads(line), pass_threshold=pass_threshold, rules=rules, arm=arm))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"{f}:{i}: not a verifiers v1 trace ({exc})") from exc
    return out
