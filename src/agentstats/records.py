"""
Attempt records: one row per (arm, run, case) attempt, the unit everything else works on.

    {"arm": "baseline", "run": 1, "case": "django-11099", "passed": true, "infra_error": null,
     "cost_usd": 0.12, "input_tokens": 41000, "output_tokens": 2100, "steps": 14, "tool_calls": 9,
     "latency_s": 33.0, "tags": ["search"], "checks": {"no_refund_before_lookup": true}}

A *run* is one full pass over the eval set. Run-level statistics (the right ones for a hand-picked eval set)
need several runs; per-case statistics need several cases. `infra_error` marks attempts that failed for reasons
outside the agent (timeout, quota, billing, sandbox crash): they are excluded from pass rates and reported apart.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

NUMERIC_FIELDS = ("cost_usd", "input_tokens", "cached_tokens", "output_tokens", "steps", "tool_calls", "latency_s")


@dataclass
class Attempt:
    case: str
    passed: bool | None
    arm: str = "default"
    run: int = 1
    infra_error: str | None = None
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
        row.update(self.metrics)
        return row


def from_dict(row: dict, *, arm: str | None = None) -> Attempt:
    if "case" not in row:
        raise ValueError(f"record has no 'case': {row}")
    passed = row.get("passed")
    if isinstance(passed, (int, float)) and not isinstance(passed, bool):
        passed = bool(passed)
    metrics = {k: float(row[k]) for k in NUMERIC_FIELDS if row.get(k) is not None}
    return Attempt(case=str(row["case"]), passed=passed, arm=arm or str(row.get("arm", "default")),
                   run=int(row.get("run", 1)), infra_error=row.get("infra_error") or None,
                   tags=list(row.get("tags") or []), checks=dict(row.get("checks") or {}), metrics=metrics)


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
    """A JSONL file, a Harbor job directory, or a directory of Harbor jobs (one per run)."""
    from agentstats import adapters

    path = Path(path)
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
