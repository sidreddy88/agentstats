"""
Sampled human review of graded attempts: is the grader (usually an LLM judge) right?

    agentstats review sample rows.jsonl --n 40 --stratify-by intent --link "https://traces/{case}?run={run}" \\
        --out review/2026-11-09
    # a person fills review/2026-11-09/review.csv (human_verdict: pass / fail / unsure)
    agentstats review import review/2026-11-09/review.csv --key review/2026-11-09/key.jsonl --rater alice \\
        --out labels.jsonl
    agentstats judge labels.jsonl          # strictness, leniency, kappa on the full label history

Which attempts get reviewed. Each graded attempt (infrastructure failures left out) falls in one stratum:
- "disagree": the verdict and the behaviour rules point opposite ways (passed with a rule violated, or failed with
  every rule satisfied): where a grader is most often wrong;
- "fail": the other failures;
- "pass": the other passes.
Half the sample goes to "disagree" (all of them if there are fewer), half of the rest to "fail", the remainder to
"pass"; unused places move on. Within a stratum the pick is spread over --stratify-by groups (e.g. intent) and is
deterministic for a --seed.

Blind by default. review.csv has the item, a link and empty columns; the grader's verdict, the stratum and the
sampling weight are only in key.jsonl, so the reviewer can't be anchored by them (or by why an item was picked).

Weights. Over-sampling disagreements makes the grader look worse than it is on the whole population. Each item's
weight is (attempts in its stratum) / (attempts sampled from it), and the import reports agreement both on the sample
and reweighted to the population, with a stratified bootstrap interval for the reweighted figures.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from agentstats.records import Attempt
from agentstats.sample import stratified

STRATA = ("disagree", "fail", "pass")
VERDICTS = {"pass": "pass", "p": "pass", "yes": "pass", "y": "pass", "1": "pass", "true": "pass", "correct": "pass",
            "fail": "fail", "f": "fail", "no": "fail", "n": "fail", "0": "fail", "false": "fail", "wrong": "fail",
            "unsure": "na", "na": "na", "n/a": "na", "skip": "na", "?": "na"}
SHEET_COLUMNS = ["item", "case", "arm", "run", "link", "human_verdict", "notes"]


def item_id(a: Attempt) -> str:
    return f"{a.arm}|{a.case}|{a.run}"


def stratum(a: Attempt) -> str:
    if a.checks:
        if a.passed and not all(a.checks.values()):
            return "disagree"
        if a.passed is False and all(a.checks.values()):
            return "disagree"
    return "pass" if a.passed else "fail"


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    alloc, left = {}, n
    for s in STRATA:
        if s == "pass":
            want = left
        elif s == "disagree":
            want = n // 2
        else:
            want = left // 2
        alloc[s] = min(sizes.get(s, 0), want)
        left -= alloc[s]
    for s in STRATA:                      # places a small stratum couldn't use move to the others, in order
        extra = min(left, sizes.get(s, 0) - alloc[s])
        alloc[s] += extra
        left -= extra
    return alloc


def sample(attempts: list[Attempt], n: int, *, stratify_by: str | None = None, seed: int = 0,
           link: str | None = None) -> tuple[list[dict], list[dict], dict]:
    """(sheet rows, key rows, summary). The sheet is blind; the key holds verdicts, strata and weights."""
    graded = [a for a in attempts if a.valid]
    if not graded:
        raise ValueError("no graded attempts to review (rows need passed true/false and no infra_error)")
    by_stratum: dict[str, dict[str, Attempt]] = defaultdict(dict)
    for a in graded:
        by_stratum[stratum(a)][item_id(a)] = a
    sizes = {s: len(v) for s, v in by_stratum.items()}
    alloc = _allocate(sizes, n)
    picked: list[tuple[str, Attempt, float]] = []
    for s in STRATA:
        k, pool = alloc.get(s, 0), by_stratum.get(s, {})
        if not k:
            continue
        ids = stratified({i: a.tags for i, a in pool.items()}, k, stratify_by, seed=seed)["cases"]
        weight = len(pool) / len(ids)
        picked += [(s, pool[i], weight) for i in ids]
    random.Random(seed).shuffle(picked)    # mixed order, so the sheet doesn't reveal the strata
    sheet, key = [], []
    for s, a, w in picked:
        iid = item_id(a)
        sheet.append({"item": iid, "case": a.case, "arm": a.arm, "run": a.run,
                      "link": link.format(case=a.case, arm=a.arm, run=a.run) if link else "",
                      "human_verdict": "", "notes": ""})
        key.append({"item": iid, "case": a.case, "arm": a.arm, "run": a.run, "stratum": s, "weight": w,
                    "judge_verdict": "pass" if a.passed else "fail",
                    "failed_checks": sorted(k for k, v in a.checks.items() if not v), "tags": a.tags})
    summary = {"graded": len(graded), "strata": sizes, "sampled": alloc, "n": len(sheet)}
    return sheet, key, summary


def write(out_dir: str | Path, sheet: list[dict], key: list[dict], summary: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path, key_path = out_dir / "review.csv", out_dir / "key.jsonl"
    if sheet_path.exists():
        raise FileExistsError(f"{sheet_path} exists: pick a new directory (a filled sheet is never overwritten)")
    with sheet_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(sheet)
    key_path.write_text("".join(json.dumps(r) + "\n" for r in key))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return sheet_path, key_path


def _norm_verdict(v: str) -> str | None:
    v = (v or "").strip().lower()
    if not v:
        return None
    if v not in VERDICTS:
        raise ValueError(f"human_verdict {v!r}: use pass, fail or unsure")
    return VERDICTS[v]


def _rates(items: list[dict]) -> dict:
    """Weighted agreement, leniency (judge pass on a human fail) and strictness (judge fail on a human pass)."""
    def ratio(sel, hit):
        den = sum(i["weight"] for i in items if sel(i))
        return sum(i["weight"] for i in items if sel(i) and hit(i)) / den if den else None
    return {"agreement": ratio(lambda i: True, lambda i: i["human"] == i["judge"]),
            "leniency": ratio(lambda i: i["human"] == "fail", lambda i: i["judge"] == "pass"),
            "strictness": ratio(lambda i: i["human"] == "pass", lambda i: i["judge"] == "fail")}


def _bootstrap(items: list[dict], reps: int = 2000, seed: int = 0) -> dict:
    """95% intervals for the weighted rates: resample items within each stratum (the design they were drawn by)."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        groups[i["stratum"]].append(i)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        resample = [rng.choice(g) for g in groups.values() for _ in g]
        for k, v in _rates(resample).items():
            if v is not None:
                draws[k].append(v)
    out = {}
    for k, vs in draws.items():
        vs.sort()
        out[k] = {"low": vs[int(0.025 * (len(vs) - 1))], "high": vs[int(0.975 * (len(vs) - 1))]}
    return out


def import_labels(sheet_path: str | Path, key_path: str | Path, rater: str = "human") -> tuple[list[dict], dict]:
    """(label rows for `agentstats judge`, report). Blank verdicts are skipped; unsure becomes na."""
    key = {r["item"]: r for r in (json.loads(x) for x in Path(key_path).read_text().splitlines() if x.strip())}
    with Path(sheet_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    unknown = [r["item"] for r in rows if r["item"] not in key]
    if unknown:
        raise ValueError(f"{len(unknown)} sheet items aren't in the key (e.g. {unknown[0]}): wrong key file?")
    labels, scored, unsure, blank = [], [], 0, 0
    for r in rows:
        human = _norm_verdict(r.get("human_verdict", ""))
        if human is None:
            blank += 1
            continue
        k = key[r["item"]]
        labels.append({"item": r["item"], "rater": f"human:{rater}", "verdict": human})
        labels.append({"item": r["item"], "rater": "judge", "verdict": k["judge_verdict"]})
        if human == "na":
            unsure += 1
            continue
        scored.append({"human": human, "judge": k["judge_verdict"], "stratum": k["stratum"], "weight": k["weight"]})
    unweighted = _rates([dict(i, weight=1.0) for i in scored]) if scored else {}
    report = {"labelled": len(scored), "unsure": unsure, "blank": blank, "on_sample": unweighted,
              "population": _rates(scored) if scored else {}, "population_ci": _bootstrap(scored) if scored else {},
              "by_stratum": {s: dict(n=sum(1 for i in scored if i["stratum"] == s),
                                     **_rates([dict(i, weight=1.0) for i in scored if i["stratum"] == s]))
                             for s in STRATA if any(i["stratum"] == s for i in scored)}}
    return labels, report


def render(report: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"
    pop, ci, smp = report["population"], report["population_ci"], report["on_sample"]
    lines = [f"labelled {report['labelled']} (unsure {report['unsure']}, blank {report['blank']})", "",
             "| | on the sample | reweighted to the population | 95% CI |", "|---|---|---|---|"]
    for k, label in (("agreement", "agreement"), ("leniency", "leniency (passes a human fails)"),
                     ("strictness", "strictness (fails a human passes)")):
        c = ci.get(k)
        interval = f"{pct(c['low'])} to {pct(c['high'])}" if c else "n/a"
        lines.append(f"| {label} | {pct(smp.get(k))} | {pct(pop.get(k))} | {interval} |")
    lines += ["", "By stratum (unweighted): " + ", ".join(
        f"{s}: n={v['n']}, agreement {pct(v['agreement'])}" for s, v in report["by_stratum"].items())]
    return "\n".join(lines)
