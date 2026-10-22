"""
A deterministic, stratified sample of cases for a smoke tier.

Every group (tag `<stratify_by>:<value>`) gets a share of the sample proportional to its size (largest
remainder), and at least one case when the sample is big enough to cover every group. Cases carrying an
`always_include` tag are always in. Same seed, same cases: a smoke result can be reproduced.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def load_cases(path: str | Path) -> dict[str, list[str]]:
    """case -> tags, from a JSONL of cases ({"case": ..., "tags": [...]}) or of attempt rows."""
    out: dict[str, set[str]] = defaultdict(set)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["case"])].update(row.get("tags") or [])
    return {c: sorted(t) for c, t in out.items()}


def group_of(tags: list[str], stratify_by: str | None) -> str:
    if not stratify_by:
        return "all"
    vals = sorted(t for t in tags if t.startswith(stratify_by + ":"))
    return vals[0] if vals else f"{stratify_by}:(none)"


def stratified(cases: dict[str, list[str]], n: int, stratify_by: str | None = None,
               always_include: list[str] | None = None, seed: int = 0) -> dict:
    rng = random.Random(seed)
    always = sorted(c for c, tags in cases.items() if set(tags) & set(always_include or []))
    notes = []
    if len(always) >= n:
        if len(always) > n:
            notes.append(f"{len(always)} always-include cases exceed the sample size {n}; all are included")
        return {"cases": always, "groups": {}, "notes": notes}

    pool: dict[str, list[str]] = defaultdict(list)
    for c in sorted(cases):
        if c not in always:
            pool[group_of(cases[c], stratify_by)].append(c)
    left = n - len(always)
    total = sum(len(v) for v in pool.values())
    if left >= total:
        return {"cases": sorted(cases), "groups": {g: len(v) for g, v in pool.items()}, "notes": notes}

    quotas = {g: left * len(v) / total for g, v in pool.items()}
    alloc = {g: int(q) for g, q in quotas.items()}
    if left >= len(pool):
        for g in pool:
            alloc[g] = max(alloc[g], 1)
    while sum(alloc.values()) > left:  # the minimum-one rule overshot: take from the largest
        g = max(alloc, key=lambda k: (alloc[k] - quotas[k], alloc[k]))
        alloc[g] -= 1
    for g in sorted(pool, key=lambda k: quotas[k] - int(quotas[k]), reverse=True):
        if sum(alloc.values()) >= left:
            break
        if alloc[g] < len(pool[g]):
            alloc[g] += 1
    picked = list(always)
    for g, members in sorted(pool.items()):
        picked += rng.sample(members, min(alloc[g], len(members)))
    return {"cases": sorted(picked), "groups": {g: alloc[g] for g in sorted(pool)}, "notes": notes}
