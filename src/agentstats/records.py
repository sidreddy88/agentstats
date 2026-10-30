"""
Attempt records: one row per (arm, run, case) attempt, the unit everything else works on.

    {"arm": "baseline", "run": 1, "case": "django-11099", "passed": true, "infra_error": null,
     "cost_usd": 0.12, "input_tokens": 41000, "output_tokens": 2100, "steps": 14, "tool_calls": 9,
     "latency_s": 33.0, "tags": ["search"], "checks": {"no_refund_before_lookup": true}}

A *run* is one full pass over the eval set. Run-level statistics (the right ones for a hand-picked eval set)
need several runs; per-case statistics need several cases. `infra_error` marks attempts that failed for reasons
outside the agent (timeout, quota, billing, sandbox crash): they are excluded from pass rates and reported apart.
`no_answer` marks attempts where the agent produced no usable answer at all (it hit its step limit, or its output
couldn't be parsed): they count as failures, like any wrong answer, but are reported apart, because a jump in them
usually means the harness or the loop broke, not that the agent's judgement got worse.

Rows are validated on load, with the field and the value in the error. `schema_version` is optional (default 1);
a row written for a newer format is refused rather than half-read.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

NUMERIC_FIELDS = ("cost_usd", "input_tokens", "cached_tokens", "output_tokens", "steps", "tool_calls", "latency_s")
SCHEMA_VERSION = 1


class RowError(ValueError):
    """A row that doesn't match the attempt format; the message names the field and what was wrong."""


@dataclass
class Attempt:
    case: str
    passed: bool | None
    arm: str = "default"
    run: int = 1
    infra_error: str | None = None
    no_answer: str | None = None      # why there was no usable answer, e.g. "step_limit", "parse_error"
    tags: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """Counts toward pass rates: graded, and not an infrastructure failure."""
        return self.infra_error is None and self.passed is not None

    def to_json(self) -> dict:
        row = {"arm": self.arm, "run": self.run, "case": self.case, "passed": self.passed,
               "infra_error": self.infra_error, "tags": self.tags, "checks": self.checks}
        if self.no_answer:
            row["no_answer"] = self.no_answer
        row.update(self.metrics)
        return row


def _number(row: dict, k: str) -> float:
    v = row[k]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        hint = " (a list of steps? write its length)" if isinstance(v, list) else ""
        raise RowError(f"'{k}' must be a number, got {type(v).__name__} {str(v)[:60]!r}{hint}")
    return float(v)


def from_dict(row: dict, *, arm: str | None = None) -> Attempt:
    if not isinstance(row, dict):
        raise RowError(f"a row must be a JSON object, got {type(row).__name__}")
    version = row.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise RowError(f"schema_version {version!r}: this agentstats reads version {SCHEMA_VERSION}; upgrade it")
    if row.get("case") in (None, ""):
        raise RowError("'case' is missing")
    if not isinstance(row["case"], (str, int)):
        raise RowError(f"'case' must be a string, got {type(row['case']).__name__}")
    passed = row.get("passed")
    if isinstance(passed, (int, float)) and not isinstance(passed, bool) and passed in (0, 1):
        passed = bool(passed)
    if passed is not None and not isinstance(passed, bool):
        raise RowError(f"'passed' must be true, false or null, got {passed!r}")
    run = row.get("run", 1)
    if isinstance(run, bool) or not isinstance(run, int) and not (isinstance(run, str) and run.isdigit()):
        raise RowError(f"'run' must be an integer (the trial or run number), got {run!r}")
    for k in ("infra_error", "no_answer"):
        if row.get(k) is not None and not isinstance(row[k], str):
            raise RowError(f"'{k}' must be a string (the reason) or null, got {row[k]!r}")
    no_answer = row.get("no_answer") or None
    if no_answer and passed:
        raise RowError(f"'no_answer' is {no_answer!r} but 'passed' is true: an attempt with no answer can't pass")
    if no_answer and passed is None and not row.get("infra_error"):
        passed = False                    # no usable answer is a failed attempt
    tags = row.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise RowError(f"'tags' must be a list of strings, got {tags!r}"[:160])
    checks = row.get("checks") or {}
    if not isinstance(checks, dict) or not all(isinstance(v, bool) for v in checks.values()):
        raise RowError(f"'checks' must map rule names to true/false, got {checks!r}"[:160])
    metrics = {k: _number(row, k) for k in NUMERIC_FIELDS if row.get(k) is not None}
    return Attempt(case=str(row["case"]), passed=passed, arm=arm or str(row.get("arm", "default")),
                   run=int(run), infra_error=row.get("infra_error") or None, no_answer=no_answer,
                   tags=list(tags), checks=dict(checks), metrics=metrics)


def load_jsonl(path: str | Path, *, arm: str | None = None) -> list[Attempt]:
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append(from_dict(json.loads(line), arm=arm))
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"{path}:{i}: {e}") from e
    return out


def write_jsonl(attempts: list[Attempt], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(a.to_json()) + "\n" for a in attempts))


def load(path: str | Path, *, arm: str | None = None, run: int | None = None) -> list[Attempt]:
    """A JSONL file, an Inspect AI log (.eval, or .json), a Harbor job directory, or a directory of Harbor jobs
    (one per run)."""
    from agentstats import adapters, inspect_logs

    path = Path(path)
    if path.is_file() and inspect_logs.is_inspect_log(path):
        return inspect_logs.load(path, arm=arm)
    if path.is_file():
        return load_jsonl(path, arm=arm)
    if adapters.is_harbor_job(path):
        return adapters.harbor_job(path, arm=arm, run=run or 1)
    jobs = sorted(p for p in path.iterdir() if p.is_dir() and adapters.is_harbor_job(p))
    if jobs:
        return [a for i, job in enumerate(jobs, 1) for a in adapters.harbor_job(job, arm=arm, run=i)]
    raise ValueError(f"{path}: not a JSONL file, a Harbor job directory, or a directory of Harbor jobs")


def by_case(attempts: list[Attempt]) -> dict[str, list[Attempt]]:
    out: dict[str, list[Attempt]] = defaultdict(list)
    for a in attempts:
        out[a.case].append(a)
    return dict(out)


def by_run(attempts: list[Attempt]) -> dict[int, list[Attempt]]:
    out: dict[int, list[Attempt]] = defaultdict(list)
    for a in attempts:
        out[a.run].append(a)
    return dict(sorted(out.items()))


def arms(attempts: list[Attempt]) -> list[str]:
    return sorted({a.arm for a in attempts})
