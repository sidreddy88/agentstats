"""
How good is this eval? One scorecard from files you already have, no model calls.

    agentstats audit results.jsonl [--other other.jsonl] [--cases cases.jsonl]
                     [--production prod_rows.jsonl --prefix error_type]

Each line is ok / warn / bad, or skipped with the input it needs:
  power        smallest drop the PR gate catches 80% of the time (simulated on these results, as `size` does)
  difficulty   regression suite, frontier, or too hard (as in `summary`)
  blind spots  passing attempts that broke a behaviour rule: the outcome check can't see these
  label source share of labels written by a model (from `promote` provenance tags in --cases)
  label bias   the baseline agrees with the labels almost perfectly while another version is marked down on the same
               cases: the labels may be the baseline's own answers. A heuristic, so it only warns.
  informative  cases that every version always passes carry no information about a change
  coverage     production types with no eval cases (as in `coverage`)
"""
from __future__ import annotations

from collections import defaultdict

from agentstats import coverage, promote, reliability, sizing
from agentstats.intervals import t_interval, wilson
from agentstats.records import Attempt

POWER_SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
POWER_TARGET = 0.80             # "catches" = the gate fails this often
POWER_OK, POWER_WARN = 0.05, 0.10
BLIND_BAD = 0.05                # bad when the interval's lower bound is above this
BIAS_BASELINE, BIAS_GAP = 0.95, 0.05
UNINFORMATIVE_WARN = 0.80
COVERAGE_BAD = 0.05             # a missing type this common in production is bad

ICON = {"ok": "✅", "warn": "⚠️ ", "bad": "❌", "skipped": "··"}


def _line(name: str, status: str, text: str, fix: str | None = None, **data) -> dict:
    return {"name": name, "status": status, "text": text, "fix": fix, **data}


def _skip(name: str, needs: str) -> dict:
    return _line(name, "skipped", f"skipped: {needs}")


def _per_case(atts: list[Attempt]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = defaultdict(list)
    for a in atts:
        if a.valid:
            out[a.case].append(bool(a.passed))
    return dict(out)


def power(atts: list[Attempt], trials: int, conf: float, sims: int, seed: int) -> dict:
    per = _per_case(atts)
    if not per or min(len(v) for v in per.values()) < 2:
        return _skip("power", "needs 2+ trials per case")
    res = sizing.simulate(atts, trials=trials, conf=conf, shifts=POWER_SHIFTS,
                          sims=sims, seed=seed)
    caught = next((r for r in res["rows"] if r["shift"] > 0 and r["FAIL"] >= POWER_TARGET), None)
    if caught is None:
        worst = res["rows"][-1]
        return _line("power", "bad", f"catches even a {100 * worst['mean_drop']:.0f}-pt drop only "
                     f"{worst['FAIL']:.0%} of the time", "add cases or trials (`agentstats size` for the table)",
                     drop=None)
    d = caught["mean_drop"]
    status = "ok" if d <= POWER_OK else "warn" if d <= POWER_WARN else "bad"
    return _line("power", status, f"catches a {100 * d:.0f}-pt drop {caught['FAIL']:.0%} of the time "
                 f"({res['cases']} cases, {res['trials']} trials per PR)",
                 None if status == "ok" else "add cases or trials to catch smaller drops", drop=d)


def difficulty(atts: list[Attempt]) -> dict:
    per = _per_case(atts)
    h = reliability.suite_health({c: (len(v), sum(v)) for c, v in per.items()})
    if not h:
        return _skip("difficulty", "needs graded attempts")
    status = {"frontier": "ok", "between": "warn", "regression": "warn", "too_hard": "bad"}[h["kind"]]
    return _line("difficulty", status, f"{h['kind'].replace('_', ' ')} ({h['pass_rate']:.0%} mean case pass rate, "
                 f"{h['always_share']:.0%} always pass)", None if status == "ok" else h["advice"], kind=h["kind"])


def blind_spots(atts: list[Attempt], conf: float) -> dict:
    passed = [a for a in atts if a.valid and a.passed and a.checks]
    if not passed:
        return _skip("blind spots", "needs passing attempts with behaviour checks (`checks` in the rows)")
    broke = [a for a in passed if not all(a.checks.values())]
    iv = wilson(len(broke), len(passed), conf)
    rules = sorted({r for a in passed for r in a.checks})
    per_rule = {r: sum(1 for a in passed if a.checks.get(r) is False) / sum(1 for a in passed if r in a.checks)
                for r in rules}
    worst = max(per_rule, key=per_rule.get)
    text = (f"{iv.estimate:.0%} of passing attempts broke a rule ({100 * iv.low:.0f}-{100 * iv.high:.0f}%); "
            f"worst: {worst} broken in {per_rule[worst]:.0%}")
    status = "bad" if iv.low > BLIND_BAD else "warn" if broke else "ok"
    fix = None if status == "ok" else f"make {worst} part of passing: the outcome check can't see it"
    return _line("blind spots", status, text if broke else f"no passing attempt broke a rule ({len(passed)} checked)",
                 fix, rate=iv.to_json(), per_rule=per_rule)


def label_source(cases: dict[str, list[str]] | None) -> dict:
    if cases is None:
        return _skip("label source", "pass --cases (with label:<who> tags, as `promote` writes)")
    prov = promote.provenance(cases)
    known = {k: v for k, v in prov["labels"].items() if k != "label:(unknown)"}
    if not known:
        return _skip("label source", "no label:<who> tags in --cases")
    n = prov["cases"]
    model = sum(v for k, v in known.items() if k.startswith("label:model:"))
    status = "bad" if prov["warnings"] else "warn" if model else "ok"
    mix = ", ".join(f"{k.removeprefix('label:')} {v}" for k, v in sorted(known.items(), key=lambda kv: -kv[1]))
    return _line("label source", status, f"{mix} (of {n} cases)",
                 None if status == "ok" else "relabel model-written cases by hand or with a different model family",
                 labels=prov["labels"])


def label_bias(atts: list[Attempt], other: list[Attempt] | None, conf: float) -> dict:
    if other is None:
        return _skip("label bias", "pass --other (another model or version on the same cases)")
    a, b = _per_case(atts), _per_case(other)
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return _skip("label bias", "--other shares fewer than 2 graded cases with the results")
    ra = [sum(a[c]) / len(a[c]) for c in common]
    rb = [sum(b[c]) / len(b[c]) for c in common]
    base, oth = sum(ra) / len(ra), sum(rb) / len(rb)
    iv = t_interval([x - y for x, y in zip(ra, rb, strict=True)], conf)
    only = sum(1 for x, y in zip(ra, rb, strict=True) if x == 1 and y == 0)
    text = (f"baseline {base:.0%} vs other {oth:.0%} on {len(common)} shared cases; {only} only the baseline "
            "ever passes")
    suspicious = base >= BIAS_BASELINE and base - oth >= BIAS_GAP and iv.low > 0
    return _line("label bias", "warn" if suspicious else "ok",
                 text + ("; labels may be the baseline's own answers" if suspicious else ""),
                 "have a person or another model family re-check the cases only the baseline passes"
                 if suspicious else None, gap=iv.to_json())


def informative(atts: list[Attempt], other: list[Attempt] | None) -> dict:
    versions = [_per_case(atts)] + ([_per_case(other)] if other is not None else [])
    common = sorted(set.intersection(*(set(v) for v in versions)))
    if not common:
        return _skip("informative", "needs graded attempts")
    flat = sum(1 for c in common if all(all(v[c]) for v in versions)) / len(common)
    who = "every version" if other is not None else "the baseline (pass --other to compare versions)"
    status = "warn" if flat >= UNINFORMATIVE_WARN else "ok"
    return _line("informative", status, f"{flat:.0%} of cases always pass for {who}",
                 "add harder cases: these can catch breakage but never show a gain" if status == "warn" else None,
                 always_share=flat)


def coverage_line(atts: list[Attempt], cases: dict[str, list[str]] | None, production: list[dict] | None,
                  prefixes: list[str], rename: dict[str, str]) -> dict:
    if production is None or not prefixes:
        return _skip("coverage", "pass --production and --prefix")
    rows = ([{"case": c, "tags": t} for c, t in cases.items()] if cases is not None
            else [{"case": a.case, "tags": a.tags} for a in atts])
    res = coverage.compare(rows, production, prefixes, rename)
    missing = [r for t in res["prefixes"].values() for r in t if r.get("flag") == "missing"]
    under = [r for t in res["prefixes"].values() for r in t if r.get("flag") == "under"]
    if res.get("untagged") and not res["prefixes"]:
        return _skip("coverage", f"the eval set has no {', '.join(res['untagged'])} tags")
    if not missing and not under:
        return _line("coverage", "ok", f"every production type has eval cases ({res['production_runs']} runs)")
    top = max(missing + under, key=lambda r: r["production_share"])
    gap = sum(r["production_share"] for r in missing)
    status = "bad" if missing and top["production_share"] >= COVERAGE_BAD else "warn"
    text = (f"{len(missing)} production types have no eval cases ({gap:.0%} of runs)" if missing
            else f"{len(under)} production types are under-represented")
    return _line("coverage", status, f"{text}; biggest: {top['value']} at {top['production_share']:.0%} of runs",
                 f"add cases for {top['value']}", missing=[r["value"] for r in missing])


def run(atts: list[Attempt], *, other: list[Attempt] | None = None, cases: dict[str, list[str]] | None = None,
        production: list[dict] | None = None, prefixes: list[str] | None = None,
        rename: dict[str, str] | None = None, trials: int = 2, conf: float = 0.95, sims: int = 300,
        seed: int = 0) -> dict:
    lines = [power(atts, trials, 0.90, sims, seed), difficulty(atts), blind_spots(atts, conf), label_source(cases),
             label_bias(atts, other, conf), informative(atts, other),
             coverage_line(atts, cases, production, prefixes or [], rename or {})]
    fixes = [ln["fix"] for s in ("bad", "warn") for ln in lines if ln["status"] == s and ln["fix"]]
    return {"cases": len(_per_case(atts)), "runs": len({a.run for a in atts}), "lines": lines,
            "top_fix": fixes[0] if fixes else None}


def render(res: dict) -> str:
    out = [f"Eval health: {res['cases']} cases, {res['runs']} run(s)", ""]
    width = max(len(ln["name"]) for ln in res["lines"])
    out += [f"{ICON[ln['status']]} {ln['name']:<{width}}  {ln['text']}" for ln in res["lines"]]
    if res["top_fix"]:
        out += ["", f"Top fix: {res['top_fix']}"]
    return "\n".join(out) + "\n"
