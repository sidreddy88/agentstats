"""
Failures become new eval cases, with provenance.

    agentstats promote find nightly/ prod_rows.jsonl --source production --label-from none \\
        --existing evals/cases.jsonl --group-by intent --out candidates.jsonl
    # a person sets "approved": true on the candidates worth keeping (and writes their expected answers in the runner)
    agentstats promote accept candidates.jsonl --into evals/cases.jsonl

What counts as a failure: a graded attempt that failed, or any attempt (graded or rules-only, e.g. production traces
with no labels) that broke a behaviour rule. Infrastructure failures don't count.

Deduplication. Failures are grouped by signature: the rules that broke (or "failed" when none did), the no-answer
reason if any, and the --group-by tag (e.g. intent:refund) if given. The same failure seen thirty times is one
signature; up to --per-signature cases are proposed for it, the ones that failed most often first (the most
reproducible). Cases already in the eval set (--existing) are skipped.

Provenance. Every candidate gets tags: source:<where it was found>, found:<date>, label:<who will label it>
(human, none, or model:<name> when the expected answer comes from a model). When accepting, the eval set's provenance
mix is printed, with a warning when most labels came from one model: comparisons against that model are then tilted
in its favour (its own answers are the answer key).

agentstats only keeps case ids and tags. The case content (input, expected answer) lives in your runner's dataset:
accepting a candidate is the reminder to add it there.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from agentstats.records import Attempt

MODEL_LABEL_WARN = 0.5


def is_failure(a: Attempt) -> bool:
    if a.infra_error:
        return False
    return a.passed is False or any(v is False for v in a.checks.values())


def signature(a: Attempt, group_by: str | None = None) -> str:
    broken = sorted(k for k, v in a.checks.items() if v is False)
    parts = ["rules:" + "+".join(broken)] if broken else ["failed"]
    if a.no_answer:
        parts.append(f"no_answer:{a.no_answer}")
    if group_by:
        vals = sorted(t for t in a.tags if t.startswith(group_by + ":"))
        parts.append(vals[0] if vals else f"{group_by}:(none)")
    return " | ".join(parts)


def find(attempts: list[Attempt], *, source: str, label_from: str = "none", existing: set[str] | None = None,
         group_by: str | None = None, per_signature: int = 3,
         found: str | None = None) -> tuple[list[dict], list[dict]]:
    """(candidates, report rows per signature)."""
    if label_from not in ("none", "human") and not label_from.startswith("model:"):
        raise ValueError("--label-from must be none, human or model:<name>")
    existing = existing or set()
    found = found or datetime.now(UTC).date().isoformat()
    groups: dict[str, list[Attempt]] = defaultdict(list)
    for a in attempts:
        if is_failure(a):
            groups[signature(a, group_by)].append(a)
    candidates, report = [], []
    for sig in sorted(groups, key=lambda s: (-len(groups[s]), s)):
        atts = groups[sig]
        fails = Counter(a.case for a in atts)
        tags: dict[str, set[str]] = defaultdict(set)
        for a in atts:
            tags[a.case].update(t for t in a.tags if not t.startswith(("source:", "found:", "label:")))
        new = [c for c in sorted(fails, key=lambda c: (-fails[c], c)) if c not in existing]
        chosen = new[:per_signature]
        report.append({"signature": sig, "failures": len(atts), "cases": len(fails),
                       "already_in_eval_set": len(fails) - len(new), "proposed": len(chosen)})
        for c in chosen:
            example = next(a for a in atts if a.case == c)
            candidates.append({
                "case": c, "signature": sig, "failures": fails[c], "arm": example.arm,
                "failed_checks": sorted(k for k, v in example.checks.items() if v is False),
                "no_answer": example.no_answer,
                "tags": sorted(tags[c]) + [f"source:{source}", f"found:{found}", f"label:{label_from}"],
                "approved": False})
    return candidates, report


def provenance(cases: dict[str, list[str]]) -> dict:
    """Share of the eval set by label origin and by source, with a warning when one model labelled most of it."""
    labels = Counter(next((t for t in tags if t.startswith("label:")), "label:(unknown)") for tags in cases.values())
    sources = Counter(next((t for t in tags if t.startswith("source:")), "source:(unknown)") for tags in cases.values())
    n = len(cases)
    warnings = [f"{100 * k / n:.0f}% of the eval set's labels came from {lab.removeprefix('label:')}: comparisons "
                "against that model are tilted in its favour (its own answers are the answer key)"
                for lab, k in labels.items() if lab.startswith("label:model:") and n and k / n > MODEL_LABEL_WARN]
    return {"cases": n, "labels": dict(labels), "sources": dict(sources), "warnings": warnings}


def accept(candidates_path: str | Path, into: str | Path) -> tuple[int, dict]:
    """Append approved candidates to the eval-set cases file. Returns (added, provenance of the result)."""
    from agentstats.sample import load_cases

    rows = [json.loads(x) for x in Path(candidates_path).read_text().splitlines() if x.strip()]
    into = Path(into)
    current = load_cases(into) if into.exists() else {}
    added = []
    for r in rows:
        if r.get("approved") is True and r["case"] not in current:
            added.append({"case": r["case"], "tags": r["tags"]})
            current[r["case"]] = r["tags"]
    with into.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in added)
    return len(added), provenance(current)


def render_report(report: list[dict]) -> str:
    lines = ["| signature | failures | cases | already in eval set | proposed |", "|---|---|---|---|---|"]
    lines += [f"| {r['signature']} | {r['failures']} | {r['cases']} | {r['already_in_eval_set']} | {r['proposed']} |"
              for r in report]
    return "\n".join(lines)
