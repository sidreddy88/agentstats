"""
Inspect AI eval logs as attempt rows. One row per sample and epoch: the case is the sample id, the run is the epoch,
the arm is the model (or --arm).

    agentstats convert --format inspect logs/2026-11-02_my-task.eval --rules rules.json --out rows.jsonl
    agentstats summary logs/2026-11-02_my-task.eval          # .eval files are also read directly

Pass/fail comes from one scorer: the log's headline scorer by default, or --scorer. Score values map as Inspect
defines them: C (correct) passes; I (incorrect) and P (partial) fail; N (no answer) fails and is marked no_answer;
true/false as is; a number passes at or above --pass-threshold (default 1.0).

A sample that errored has no score. By default its error becomes an infrastructure failure (kept out of pass rates,
counted, and a gate reports UNHEALTHY if there are too many), because Inspect sample errors are usually runtime
failures (a sandbox, a provider). If your errors are the agent's own fault, use --errors fail. A sample that hit a
limit (tokens, messages, time) and didn't pass is marked no_answer as "limit_<type>".

Tool calls are read from the assistant messages, in order, and fill the row's checks from --rules (the same rules as
`agentstats check` and `convert-traces`). Tokens, time, steps (assistant turns) and tool calls become metrics; task,
model and scalar sample metadata become tags (e.g. intent:refund), so slices work.

Reading: with inspect_ai installed (anyone who has .eval logs usually does) its own reader is used, so every log
version works. Without it, .json logs are read directly, and .eval logs (zip archives, zstd-compressed) are read with
the standard library on Python 3.14+; otherwise install inspect-ai or run `inspect log convert --to json`.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from agentstats.checks import run_checks
from agentstats.records import Attempt, from_dict

ERRORS = ("infra", "fail")


def is_inspect_log(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix == ".eval":
        return True
    if path.suffix != ".json" or not path.is_file():
        return False
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and "eval" in d and "samples" in d


def read_log(path: str | Path) -> dict:
    """The log as a plain dict (the same shape as Inspect's JSON log format)."""
    path = Path(path)
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        read_eval_log = None
    if read_eval_log is not None:
        return read_eval_log(str(path)).model_dump(mode="json", exclude_none=True)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    try:
        with zipfile.ZipFile(path) as z:
            header = json.loads(z.read("header.json"))
            header["samples"] = [json.loads(z.read(n)) for n in sorted(z.namelist())
                                 if n.startswith("samples/") and n.endswith(".json")]
            return header
    except NotImplementedError:
        raise ValueError(f"{path}: .eval logs are zstd-compressed and this Python can't read them; "
                         "pip install inspect-ai, or run `inspect log convert --to json` first") from None


def scorer_names(log: dict) -> list[str]:
    names = [s["name"] for s in (log.get("results") or {}).get("scores", [])]
    if not names:
        names = sorted({k for s in log.get("samples", []) for k in (s.get("scores") or {})})
    return names


def pick_scorer(log: dict, scorer: str | None) -> str | None:
    """The scorer that decides pass/fail; None when no sample was scored (e.g. every sample errored)."""
    names = scorer_names(log)
    if scorer:
        if scorer not in names:
            raise ValueError(f"no scorer {scorer!r} in this log; it has {names}")
        return scorer
    headline = ((log.get("results") or {}).get("headline") or {}).get("scorer")
    if headline:
        return headline
    if len(names) == 1:
        return names[0]
    if not names:
        return None
    raise ValueError(f"the log has several scorers {names}: choose one with --scorer")


def verdict(value, pass_threshold: float = 1.0) -> tuple[bool | None, str | None]:
    """(passed, no_answer) from an Inspect score value."""
    if isinstance(value, bool):
        return value, None
    if isinstance(value, (int, float)):
        return float(value) >= pass_threshold, None
    if isinstance(value, str):
        v = value.strip().upper()
        if v in ("C", "CORRECT"):
            return True, None
        if v in ("I", "INCORRECT", "P", "PARTIAL"):
            return False, None
        if v in ("N", "NOANSWER"):
            return False, "noanswer"
    raise ValueError(f"can't read score value {value!r} as pass/fail (dict-valued scores aren't supported)")


def tool_calls(sample: dict) -> list[dict]:
    """Tool calls in order, as {"name", "args"}: the shape agentstats' rule checks expect."""
    calls = []
    for m in sample.get("messages", []):
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                calls.append({"name": c.get("function"), "args": c.get("arguments") or {}})
    return calls


def _metrics(sample: dict, calls: list[dict]) -> dict[str, float]:
    usage = (sample.get("model_usage") or {}).values()
    tokens_in = sum(u.get("input_tokens", 0) for u in usage)
    tokens_out = sum(u.get("output_tokens", 0) for u in usage)
    out = {"steps": float(sum(1 for m in sample.get("messages", []) if m.get("role") == "assistant")),
           "tool_calls": float(len(calls))}
    if tokens_in or tokens_out:
        out.update(input_tokens=float(tokens_in), output_tokens=float(tokens_out))
    cost = sum(u.get("total_cost") or 0 for u in usage)
    if cost:
        out["cost_usd"] = float(cost)
    if sample.get("total_time") is not None:
        out["latency_s"] = float(sample["total_time"])
    return out


def _tags(log: dict, sample: dict) -> list[str]:
    ev = log.get("eval", {})
    tags = [f"task:{ev['task']}"] if ev.get("task") else []
    if ev.get("model"):
        tags.append(f"model:{ev['model']}")
    for k, v in (sample.get("metadata") or {}).items():
        if isinstance(v, (str, int, float, bool)):
            tags.append(f"{k}:{v}")
    return tags


def to_attempts(log: dict, *, scorer: str | None = None, pass_threshold: float = 1.0, errors: str = "infra",
                rules: list[dict] | None = None, arm: str | None = None) -> list[Attempt]:
    if errors not in ERRORS:
        raise ValueError(f"errors must be one of {ERRORS}")
    samples = log.get("samples") or []
    if not samples:
        raise ValueError("the log has no samples (it may have been written with samples left out)")
    name = pick_scorer(log, scorer)
    arm = arm or log.get("eval", {}).get("model") or "default"
    out = []
    for s in samples:
        calls = tool_calls(s)
        row = {"arm": arm, "case": str(s["id"]), "run": int(s.get("epoch") or 1), "passed": None, "infra_error": None,
               "tags": _tags(log, s), "checks": run_checks(rules, calls) if rules else {}, **_metrics(s, calls)}
        error = s.get("error")
        score = (s.get("scores") or {}).get(name) if name else None
        if error:
            message = str(error.get("message") or error) if isinstance(error, dict) else str(error)
            if errors == "infra":
                row["infra_error"] = message[:200]
            else:
                row["passed"] = False
        elif score is not None:
            row["passed"], no_answer = verdict(score.get("value"), pass_threshold)
            if no_answer:
                row["no_answer"] = no_answer
        limit = s.get("limit")
        if limit and row["passed"] is False and not row.get("no_answer"):
            row["no_answer"] = f"limit_{limit.get('type', 'unknown') if isinstance(limit, dict) else limit}"
        out.append(from_dict(row))
    return out


def load(path: str | Path, **kwargs) -> list[Attempt]:
    return to_attempts(read_log(path), **kwargs)
