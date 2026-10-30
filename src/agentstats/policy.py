"""
Release-process tiers (evals in CI: smoke, nightly, release), from a TOML policy file.

    design = "curated"              # how the eval set was built (see analysis.py)

    [smoke]                         # every change, minutes: a stratified sample + rule checks
    sample = 20
    stratify_by = "feature"         # tag prefix: feature:search, feature:checkout, ...
    always_include = ["critical"]   # tags always in the sample
    runs = 1
    require.rules = 1.0             # every trajectory rule check passes
    require.min_pass_rate = 0.5     # a crash threshold, not a regression test

    [pr]                            # every pull request: fail only on a proven drop
    runs = 2
    require.no_significant_drop = 0.0

    [nightly]                       # the whole eval set, k runs per case
    runs = 3
    require.no_regression = 0.03    # vs baseline: CI lower bound of the change >= -3 pts
    require.max_infra = 0.05

    [release]                       # the whole set, against the production baseline, with cost and latency
    runs = 5
    require.no_regression = 0.02
    require.max_cost_ratio = 1.10
    require.max_p95_latency_ratio = 1.20
    require.min_pass_hat_k = 0.70
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from agentstats.analysis import DESIGNS

REQUIREMENTS = {
    "min_pass_rate": "pass rate (point estimate) at least this; a crash threshold",
    "rules": "share of trajectory rule checks passing, at least this",
    "min_rule_rate": "every trajectory rule holds at least this often: PASS when the estimate clears it, FAIL when "
                     "the whole interval is below it, INCONCLUSIVE in between",
    "no_rule_regression": "vs baseline, per trajectory rule: FAIL if any rule is proven to hold less often than in "
                          "the baseline by more than this (fixed-cases test, Bonferroni across rules). Needs no labels",
    "no_regression": "vs baseline: lower CI bound of (candidate - baseline) at least -margin (non-inferiority: "
                     "proves it isn't worse; needs many runs)",
    "no_significant_drop": "vs baseline: FAIL only if the whole CI is below -margin (proves it got worse; "
                           "otherwise PASS, with the smallest drop this run could catch)",
    "max_infra": "infrastructure failure rate at most this; above it the run is unhealthy",
    "max_cost_ratio": "cost per attempt, candidate / baseline, at most this (CI-based)",
    "max_p95_latency_ratio": "p95 latency, candidate / baseline, at most this (CI-based)",
    "min_pass_hat_k": "pass^k (k = the tier's runs) at least this (CI-based)",
    "max_no_answer": "share of attempts that ended with no usable answer (step limit, unparseable output) at most "
                     "this; a jump usually means the harness or loop broke",
    "max_slice_drop": "vs baseline, per slice (the tier's slice_by tag prefix): FAIL if any slice is proven worse by "
                      "more than this (Bonferroni-adjusted interval across slices)",
}
NEEDS_BASELINE = ("no_regression", "no_significant_drop", "max_cost_ratio", "max_p95_latency_ratio", "max_slice_drop",
                  "no_rule_regression")


@dataclass
class Tier:
    name: str
    runs: int = 1
    sample: int | None = None
    stratify_by: str | None = None
    always_include: list[str] = field(default_factory=list)
    seed: int = 0
    slice_by: str | None = None      # tag prefix to compare per slice against the baseline, e.g. "template"
    require: dict[str, float] = field(default_factory=dict)


@dataclass
class Policy:
    design: str
    cluster_by: str | None
    tiers: dict[str, Tier]


def parse(data: dict) -> Policy:
    design = data.get("design", "curated")
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, not {design!r}")
    tiers = {}
    for name, body in data.items():
        if not isinstance(body, dict):
            continue
        req = dict(body.get("require") or {})
        unknown = set(req) - set(REQUIREMENTS)
        if unknown:
            raise ValueError(f"[{name}] unknown requirement(s) {sorted(unknown)}; known: {sorted(REQUIREMENTS)}")
        tiers[name] = Tier(name=name, runs=int(body.get("runs", 1)), sample=body.get("sample"),
                           stratify_by=body.get("stratify_by"), always_include=list(body.get("always_include") or []),
                           seed=int(body.get("seed", 0)), slice_by=body.get("slice_by"),
                           require={k: float(v) for k, v in req.items()})
        if "max_slice_drop" in req and not tiers[name].slice_by:
            raise ValueError(f"[{name}] max_slice_drop needs slice_by (a tag prefix, e.g. slice_by = \"template\")")
    if not tiers:
        raise ValueError("policy defines no tiers")
    return Policy(design=design, cluster_by=data.get("cluster_by"), tiers=tiers)


def load(path: str | Path) -> Policy:
    return parse(tomllib.loads(Path(path).read_text()))
