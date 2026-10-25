"""
Size a gate before trusting it: how often would it pass, fail or be inconclusive for a range of true changes?

Simulated from your baseline's own per-case rates, so the answer fits your eval set (many cases near 0% or 100%,
a few that flip). For each simulation, each case's true pass rate is drawn from its baseline trials (Jeffreys prior,
so a case that passed 4 of 4 still carries some uncertainty), shifted by `shift` on the log-odds scale for the
candidate, and BOTH sides are simulated from it: a fresh baseline with the baseline's trial count and a candidate with
`trials`. Comparing simulated candidates with the observed baseline instead would be biased: a case that passed 4 of 4
has a true rate below 1 on average, so even no change would look like a drop. (Method from agent-platform's diagnosis
gate v2 power analysis.)

Rules:
  drop            FAIL only when the whole interval is below -margin, else PASS   (gate requirement no_significant_drop)
  noninferiority  PASS when the lower bound >= -margin, FAIL when the upper bound < -margin, else INCONCLUSIVE
                                                                                   (gate requirement no_regression)

Pick trials and margin so that "no change" passes often enough (e.g. >= 90%) and the drops you care about fail often
enough (e.g. >= 80%).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from agentstats.comparisons import fixed_cases
from agentstats.records import Attempt

RULES = ("drop", "noninferiority")
DEFAULT_SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def _per_case(baseline: list[Attempt]) -> dict[str, tuple[int, int]]:
    out: dict[str, list[bool]] = defaultdict(list)
    for a in baseline:
        if a.valid:
            out[a.case].append(bool(a.passed))
    return {c: (len(v), sum(v)) for c, v in out.items()}


def simulate(baseline: list[Attempt], trials: int, rule: str = "drop", margin: float = 0.0, conf: float = 0.90,
             shifts=DEFAULT_SHIFTS, sims: int = 1000, seed: int = 0) -> dict:
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    counts = _per_case(baseline)
    if not counts:
        raise ValueError("baseline has no graded attempts")
    m = min(n for n, _ in counts.values())
    rng = random.Random(seed)
    rows = []
    for shift in shifts:
        out = {"PASS": 0, "INCONCLUSIVE": 0, "FAIL": 0}
        drop = 0.0
        for _ in range(sims):
            a, b, d = {}, {}, 0.0
            for case, (n, s) in counts.items():
                p = min(max(rng.betavariate(s + 0.5, n - s + 0.5), 1e-6), 1 - 1e-6)
                q = 1 / (1 + math.exp(-(math.log(p / (1 - p)) - shift)))
                d += p - q
                a[case] = (m, sum(rng.random() < p for _ in range(m)))
                b[case] = (trials, sum(rng.random() < q for _ in range(trials)))
            drop += d / len(counts)
            iv = fixed_cases(a, b, conf)
            if rule == "drop":
                out["FAIL" if iv.high < -margin else "PASS"] += 1
            else:
                out["PASS" if iv.low >= -margin else "FAIL" if iv.high < -margin else "INCONCLUSIVE"] += 1
        rows.append({"shift": shift, "mean_drop": drop / sims, **{k: v / sims for k, v in out.items()}})
    return {"cases": len(counts), "baseline_trials": m, "trials": trials, "rule": rule, "margin": margin,
            "conf": conf, "sims": sims, "rows": rows}


def render(r: dict) -> str:
    title = f"{r['rule']} at a {100 * r['margin']:.0f}-point margin, {100 * r['conf']:.0f}% interval"
    lines = [f"# agentstats size: {title}",
             "", f"{r['cases']} cases, baseline {r['baseline_trials']} trial(s), candidate {r['trials']} trial(s), "
             f"{r['sims']} simulations per row.", "",
             "| true shift (log-odds) | mean drop | PASS | INCONCLUSIVE | FAIL |", "|---|---|---|---|---|"]
    for row in r["rows"]:
        lines.append(f"| {row['shift']} | {100 * row['mean_drop']:.1f} pts | {row['PASS']:.0%} | "
                     f"{row['INCONCLUSIVE']:.0%} | {row['FAIL']:.0%} |")
    zero = next((x for x in r["rows"] if x["shift"] == 0), None)
    if zero:
        lines += ["", f"No change: {zero['PASS']:.0%} pass" + (f", {zero['FAIL']:.0%} false alarms"
                  if r["rule"] == "drop" else f", {zero['INCONCLUSIVE']:.0%} inconclusive, {zero['FAIL']:.0%} fail")]
    return "\n".join(lines) + "\n"
