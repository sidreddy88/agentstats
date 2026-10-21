"""
Build attempt records from agent run output: Harbor job directories and ATIF trajectories.

Harbor (harborframework.com) writes one directory per trial with `result.json` (verifier reward, exception,
tokens, cost, timings) and `agent/trajectory.json` in ATIF, the Agent Trajectory Interchange Format. A trial that
raised an exception is an infrastructure failure, not an agent failure, and is kept apart from pass rates.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agentstats.records import Attempt


def is_harbor_job(path: Path) -> bool:
    return path.is_dir() and any((t / "result.json").exists() and (t / "config.json").exists()
                                 for t in path.iterdir() if t.is_dir())


def _seconds(span: dict | None) -> float | None:
    if not span or not span.get("started_at") or not span.get("finished_at"):
        return None
    start = datetime.fromisoformat(span["started_at"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(span["finished_at"].replace("Z", "+00:00"))
    return (end - start).total_seconds()


def tool_calls(trajectory: dict | str | Path) -> list[dict]:
    """The agent's tool calls in order, from an ATIF trajectory: [{"name", "args", "step"}]."""
    d = trajectory if isinstance(trajectory, dict) else json.loads(Path(trajectory).read_text())
    calls = []
    for i, st in enumerate(d.get("steps") or []):
        if st.get("source") != "agent":
            continue
        for call in st.get("tool_calls") or []:
            args = call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({"name": call.get("function_name"), "args": args, "step": i})
    return calls


def trajectory_metrics(trajectory: dict | str | Path) -> dict[str, float]:
    d = trajectory if isinstance(trajectory, dict) else json.loads(Path(trajectory).read_text())
    agent_steps = [s for s in d.get("steps") or [] if s.get("source") == "agent"]
    out: dict[str, float] = {"steps": float(len(agent_steps)),
                             "tool_calls": float(sum(len(s.get("tool_calls") or []) for s in agent_steps))}
    fm = d.get("final_metrics") or {}
    for src, dst in (("total_prompt_tokens", "input_tokens"), ("total_completion_tokens", "output_tokens"),
                     ("total_cached_tokens", "cached_tokens"), ("total_cost_usd", "cost_usd")):
        if fm.get(src) is not None:
            out[dst] = float(fm[src])
    return out


def harbor_trial(trial_dir: str | Path, *, arm: str | None = None, run: int = 1,
                 pass_threshold: float = 1.0) -> Attempt:
    trial_dir = Path(trial_dir)
    result: dict[str, Any] = json.loads((trial_dir / "result.json").read_text())
    case = result.get("task_name") or trial_dir.name.rsplit("__", 1)[0]
    exc = result.get("exception_info")
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    infra = (exc or {}).get("exception_type") or ("Exception" if exc else None)
    passed = None if infra or reward is None else reward >= pass_threshold
    if not infra and reward is None:
        infra = "NoReward"

    metrics: dict[str, float] = {}
    traj = trial_dir / "agent" / "trajectory.json"
    if traj.exists():
        try:
            metrics.update(trajectory_metrics(traj))
        except (json.JSONDecodeError, OSError):
            pass
    ar = result.get("agent_result") or {}
    for src, dst in (("n_input_tokens", "input_tokens"), ("n_cache_tokens", "cached_tokens"),
                     ("n_output_tokens", "output_tokens"), ("cost_usd", "cost_usd")):
        if ar.get(src) is not None:
            metrics[dst] = float(ar[src])  # result.json is the billing source of truth
    if (secs := _seconds(result.get("agent_execution"))) is not None:
        metrics["latency_s"] = secs
    agent = (result.get("agent_info") or {}).get("name") or "agent"
    return Attempt(case=case, passed=passed, arm=arm or agent, run=run, infra_error=infra, metrics=metrics)


def harbor_job(job_dir: str | Path, *, arm: str | None = None, run: int = 1,
               pass_threshold: float = 1.0) -> list[Attempt]:
    """Every trial in a Harbor job directory. Repeated trials of one task (Harbor's n_attempts) become
    separate runs, numbered from `run`."""
    job_dir = Path(job_dir)
    trials = sorted(t for t in job_dir.iterdir() if t.is_dir() and (t / "result.json").exists())
    out, seen = [], {}
    for t in trials:
        a = harbor_trial(t, arm=arm or job_dir.parent.name, run=run, pass_threshold=pass_threshold)
        seen[a.case] = seen.get(a.case, 0) + 1
        a.run = run + seen[a.case] - 1
        out.append(a)
    return out


def atif_dir(directory: str | Path, verdicts: dict[str, bool | None], *, arm: str = "default",
             run: int = 1) -> list[Attempt]:
    """A folder of ATIF trajectories named `<case>.json`, with verdicts from your own grader."""
    out = []
    for p in sorted(Path(directory).glob("*.json")):
        out.append(Attempt(case=p.stem, passed=verdicts.get(p.stem), arm=arm, run=run,
                           infra_error=None if p.stem in verdicts else "NoVerdict",
                           metrics=trajectory_metrics(p)))
    return out
