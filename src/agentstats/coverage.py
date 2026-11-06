"""
Is the eval set like production? Compares how often each tag value (error type, severity, language, feature...)
appears in an eval set and in production runs, so an eval set that over-represents a few kinds of input, or never
covers one that production sees, shows up before its pass rate is trusted.

Inputs are agentstats rows (JSONL with "case" and "tags", e.g. "error_type:TypeError"). Each eval case counts once,
however many runs it had; each production row is one run. `rename` maps an eval-set tag prefix onto the production
one when they differ (e.g. template -> error_type).

For each value: its share in each, the difference with a Newcombe interval, and a flag:
  missing      seen in production, no eval case at all (the most actionable gap)
  under / over the difference's interval excludes zero (proven, not just noise), with at least MIN_RUNS production
               runs: a share from a handful of runs is too rough to call over- or under-represented
Values only in the eval set are listed but not flagged: an eval set can rightly cover rare or past failures.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from agentstats.intervals import newcombe

MIN_RUNS = 10


def _values(rows: list[dict], prefix: str, *, by_case: bool, rename: dict[str, str]) -> Counter:
    seen: dict[str, set] = {}
    counts: Counter = Counter()
    for i, r in enumerate(rows):
        key = str(r.get("case")) if by_case else str(i)
        vals = set()
        for t in r.get("tags") or []:
            p, _, v = str(t).partition(":")
            if rename.get(p, p) == prefix and v:
                vals.add(v)
        if not vals:
            vals = {"(none)"}
        if by_case:
            if key in seen:
                continue
            seen[key] = vals
        for v in vals:
            counts[v] += 1
    return counts


def compare(eval_rows: list[dict], production_rows: list[dict], prefixes: list[str],
            rename: dict[str, str] | None = None, conf: float = 0.95) -> dict:
    rename = rename or {}
    n_eval = len({str(r.get("case")) for r in eval_rows})
    n_prod = len(production_rows)
    out: dict = {"eval_cases": n_eval, "production_runs": n_prod, "prefixes": {}}
    for prefix in prefixes:
        ev = _values(eval_rows, prefix, by_case=True, rename=rename)
        pr = _values(production_rows, prefix, by_case=False, rename={})
        if set(ev) == {"(none)"}:
            out.setdefault("untagged", []).append(prefix)       # the eval set doesn't record this tag at all
            continue
        table = []
        for v in sorted(set(ev) | set(pr), key=lambda x: (-pr.get(x, 0), -ev.get(x, 0), x)):
            e, p = ev.get(v, 0), pr.get(v, 0)
            row = {"value": v, "eval": e, "production": p,
                   "eval_share": e / n_eval if n_eval else 0.0, "production_share": p / n_prod if n_prod else 0.0}
            if n_eval and n_prod:
                iv = newcombe(p, n_prod, e, n_eval, conf)            # eval share minus production share (b - a)
                row["difference"] = iv.to_json()
                if p and not e:
                    row["flag"] = "missing"
                elif n_prod >= MIN_RUNS and iv.high < 0:
                    row["flag"] = "under"
                elif n_prod >= MIN_RUNS and iv.low > 0 and p:
                    row["flag"] = "over"
            table.append(row)
        out["prefixes"][prefix] = table
    return out


def render(res: dict, top: int = 12) -> str:
    lines = [f"Eval set: {res['eval_cases']} cases. Production: {res['production_runs']} runs."]
    if res["production_runs"] < 30:
        lines.append(f"Only {res['production_runs']} production runs: shares are rough"
                     + (f" (over/under needs {MIN_RUNS}+)" if res["production_runs"] < MIN_RUNS else "")
                     + "; 'missing' is still worth acting on.")
    for prefix in res.get("untagged", []):
        lines.append(f"The eval set has no `{prefix}:` tags, so {prefix} can't be compared "
                     "(tag the cases to check it).")
    for prefix, table in res["prefixes"].items():
        flagged = [r for r in table if r.get("flag")]
        lines += ["", f"**{prefix}**" + (f": {len(flagged)} flagged" if flagged else ": no gap proven"), "",
                  "| value | eval set | production | |", "|---|---|---|---|"]
        shown = (flagged + [r for r in table if not r.get("flag")])[:top]
        for r in shown:
            note = {"missing": "**in production, not in the eval set**", "under": "**under-represented**",
                    "over": "over-represented"}.get(r.get("flag", ""), "")
            lines.append(f"| {r['value']} | {r['eval']} ({100 * r['eval_share']:.0f}%) | "
                         f"{r['production']} ({100 * r['production_share']:.0f}%) | {note} |")
        if len(table) > len(shown):
            lines.append(f"| ... {len(table) - len(shown)} more | | | |")
    return "\n".join(lines) + "\n"


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
