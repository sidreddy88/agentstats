"""
A CI gate for one tier: check a tier's requirements on a candidate's results, against a baseline where needed.

Each requirement is PASS, FAIL or INCONCLUSIVE. Requirements that compare with a baseline are non-inferiority
tests: "no regression" PASSes only when the lower confidence bound of (candidate - baseline) clears -margin.
"The interval includes zero" is not evidence that nothing regressed; an under-powered run is INCONCLUSIVE, with
an estimate of how many runs would settle it, never PASS.

Exit codes: 0 pass, 1 fail, 2 inconclusive, 3 unhealthy (too many infrastructure failures to judge the agent).
"""
from __future__ import annotations

import math

from agentstats import analysis
from agentstats.comparisons import fixed_cases
from agentstats.intervals import Interval, bootstrap, mean, quantile, wilson
from agentstats.policy import NEEDS_BASELINE, Policy, Tier
from agentstats.records import Attempt, by_case

PASS, FAIL, INCONCLUSIVE, UNHEALTHY = "PASS", "FAIL", "INCONCLUSIVE", "UNHEALTHY"
EXIT = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2, UNHEALTHY: 3}


def _row(name, measured, threshold, result, detail=""):
    return {"requirement": name, "measured": measured, "threshold": threshold, "result": result, "detail": detail}


def _ratio_interval(baseline: list[Attempt], candidate: list[Attempt], field: str, stat,
                    conf: float) -> Interval | None:
    """Bootstrap over matched cases of stat(candidate values) / stat(baseline values)."""
    a, b = by_case([x for x in baseline if x.valid]), by_case([x for x in candidate if x.valid])
    units = []
    for c in sorted(set(a) & set(b)):
        av = [x.metrics[field] for x in a[c] if field in x.metrics]
        bv = [x.metrics[field] for x in b[c] if field in x.metrics]
        if av and bv:
            units.append((av, bv))
    if len(units) < 2:
        return None

    def ratio(us):
        den = stat([v for av, _ in us for v in av])
        return float("nan") if den == 0 else stat([v for _, bv in us for v in bv]) / den
    return bootstrap(units, ratio, conf, method="bootstrap over matched cases")


def _upper_bound_check(name, iv: Interval | None, limit: float, fmt) -> dict:
    if iv is None or math.isnan(iv.low):
        return _row(name, "n/a", f"<= {fmt(limit)}", INCONCLUSIVE, "not enough matched cases with this metric")
    measured = f"{fmt(iv.estimate)} (CI {fmt(iv.low)} to {fmt(iv.high)})"
    if iv.high <= limit:
        return _row(name, measured, f"<= {fmt(limit)}", PASS)
    if iv.low > limit:
        return _row(name, measured, f"<= {fmt(limit)}", FAIL)
    return _row(name, measured, f"<= {fmt(limit)}", INCONCLUSIVE, "the interval straddles the limit; add runs")


def _lower_bound_check(name, iv: dict | None, floor: float, fmt) -> dict:
    if not iv or iv.get("low") is None or math.isnan(iv["low"]):
        return _row(name, "n/a", f">= {fmt(floor)}", INCONCLUSIVE, "not measurable with these runs")
    measured = f"{fmt(iv['estimate'])} (CI {fmt(iv['low'])} to {fmt(iv['high'])})"
    if iv["low"] >= floor:
        return _row(name, measured, f">= {fmt(floor)}", PASS)
    if iv["high"] < floor:
        return _row(name, measured, f">= {fmt(floor)}", FAIL)
    return _row(name, measured, f">= {fmt(floor)}", INCONCLUSIVE, "the interval straddles the floor; add runs")


def _pct(x):
    return f"{100 * x:.1f}%"


def _pts(x):
    return f"{100 * x:+.1f} pts"


def _x(x):
    return f"{x:.2f}x"


def runs_needed(delta: dict, margin: float, runs: int, crit: float = 1.96) -> str:
    """Rough number of runs per arm for the lower bound to clear -margin, if the estimate holds."""
    est, se = delta["estimate"], delta.get("se")
    if est + margin <= 0:
        return "the estimate itself is below the margin; more runs are unlikely to make this pass"
    if not se or math.isnan(se):
        return "measure run-to-run noise first: run at least 3 times per arm"
    target = (est + margin) / crit
    need = math.ceil(max(runs, 1) * (se / target) ** 2)
    return f"about {max(need, runs + 1)} runs per arm would settle it if the estimate holds"


def slice_comparisons(baseline: list[Attempt], candidate: list[Attempt], prefix: str, conf: float) -> list[dict]:
    """Candidate vs baseline per slice (cases tagged prefix:<name>), fixed-cases test on the cases both arms graded.
    With many slices some will look worse by chance, so each interval is Bonferroni-adjusted across slices."""
    def counts(atts: list[Attempt]) -> dict[str, tuple[int, int]]:
        return {c: (len(v), sum(1 for a in v if a.passed)) for c, v in by_case([a for a in atts if a.valid]).items()}

    def tags_of(atts: list[Attempt]) -> dict[str, str]:
        return {a.case: t[len(prefix) + 1:] for a in atts for t in a.tags if t.startswith(prefix + ":")}

    a_counts, b_counts = counts(baseline), counts(candidate)
    names = tags_of(baseline) | tags_of(candidate)
    groups: dict[str, list[str]] = {}
    for c in sorted(set(a_counts) & set(b_counts) & set(names)):
        groups.setdefault(names[c], []).append(c)
    adj = 1 - (1 - conf) / max(len(groups), 1)
    out = []
    for name, cases in sorted(groups.items()):
        iv = fixed_cases({c: a_counts[c] for c in cases}, {c: b_counts[c] for c in cases}, adj)
        def rate(cnt: dict[str, tuple[int, int]], cases: list[str] = cases) -> float:
            return sum(cnt[c][1] for c in cases) / sum(cnt[c][0] for c in cases)
        out.append({"slice": name, "cases": len(cases), "baseline": rate(a_counts), "candidate": rate(b_counts),
                    "delta": iv.estimate, "low": max(iv.low, -1.0), "high": min(iv.high, 1.0), "conf": adj})
    return out


def _rule_counts(atts: list[Attempt]) -> dict[str, dict[str, tuple[int, int]]]:
    """rule -> case -> (attempts where the rule was checked, attempts where it held), over graded attempts."""
    out: dict[str, dict[str, list[int]]] = {}
    for a in atts:
        if not a.valid:
            continue
        for rule, ok in a.checks.items():
            n_s = out.setdefault(rule, {}).setdefault(a.case, [0, 0])
            n_s[0] += 1
            n_s[1] += bool(ok)
    return {r: {c: (n, s) for c, (n, s) in cases.items()} for r, cases in out.items()}


def rule_comparisons(baseline: list[Attempt] | None, candidate: list[Attempt], conf: float) -> list[dict]:
    """Per trajectory rule: how often it held in the candidate (Wilson interval over attempts) and, with a baseline,
    candidate vs baseline on the cases both checked (fixed-cases, Bonferroni-adjusted across rules). These need no
    labels: they measure behaviour ("called the lookup tool"), so they still work when the labels are wrong."""
    cand = _rule_counts(candidate)
    base = _rule_counts(baseline) if baseline is not None else {}
    adj = 1 - (1 - conf) / max(len(cand), 1)
    out = []
    for rule in sorted(cand):
        n = sum(k for k, _ in cand[rule].values())
        s = sum(v for _, v in cand[rule].values())
        iv = wilson(s, n, conf)
        row = {"rule": rule, "attempts": n, "rate": s / n, "low": iv.low, "high": iv.high, "conf": adj}
        shared = sorted(set(cand[rule]) & set(base.get(rule, {})))
        if shared:
            b_counts = {c: base[rule][c] for c in shared}
            d = fixed_cases(b_counts, {c: cand[rule][c] for c in shared}, adj)
            row.update(baseline_rate=sum(v for _, v in b_counts.values()) / sum(k for k, _ in b_counts.values()),
                       delta=d.estimate, delta_low=max(d.low, -1.0), delta_high=min(d.high, 1.0), cases=len(shared))
        out.append(row)
    return out


def evaluate(tier: Tier, policy: Policy, candidate: list[Attempt], baseline: list[Attempt] | None = None,
             conf: float = 0.95) -> dict:
    req = tier.require
    rows: list[dict] = []
    notes: list[str] = []
    missing = [r for r in req if r in NEEDS_BASELINE and baseline is None]
    if missing:
        raise ValueError(f"[{tier.name}] requirements {missing} need --baseline")
    summ = analysis.summarize(candidate, policy.design, policy.cluster_by, conf)
    runs = summ["runs"]
    if runs < tier.runs:
        rows.append(_row("runs", str(runs), f">= {tier.runs}", INCONCLUSIVE,
                         f"the {tier.name} tier expects {tier.runs} complete runs"))

    if "max_infra" in req:
        rate = summ["infra"]["rate"]
        rows.append(_row("max_infra", _pct(rate), f"<= {_pct(req['max_infra'])}",
                         PASS if rate <= req["max_infra"] else UNHEALTHY, str(summ["infra"]["kinds"] or "")))

    if "min_pass_rate" in req:
        est = summ["pass_rate"]["estimate"]
        rows.append(_row("min_pass_rate", f"{_pct(est)} (CI {_pct(summ['pass_rate']['low'])} to "
                         f"{_pct(summ['pass_rate']['high'])})", f">= {_pct(req['min_pass_rate'])}",
                         PASS if est >= req["min_pass_rate"] else FAIL, "point estimate: a breakage check"))

    if "rules" in req:
        outcomes = [v for a in candidate if a.valid for v in a.checks.values()]
        if not outcomes:
            rows.append(_row("rules", "n/a", f">= {_pct(req['rules'])}", INCONCLUSIVE, "no rule checks in the results"))
        else:
            share = sum(outcomes) / len(outcomes)
            failing = sorted({f"{a.case}:{r}" for a in candidate if a.valid for r, ok in a.checks.items() if not ok})
            rows.append(_row("rules", f"{_pct(share)} of {len(outcomes)} checks", f">= {_pct(req['rules'])}",
                             PASS if share >= req["rules"] else FAIL, ", ".join(failing[:8])))

    rules = None
    if "min_rule_rate" in req or "no_rule_regression" in req or any(a.checks for a in candidate):
        rules = rule_comparisons(baseline, candidate, conf) or None
    if "min_rule_rate" in req:
        floor = req["min_rule_rate"]
        if not rules:
            rows.append(_row("min_rule_rate", "n/a", f">= {_pct(floor)}", INCONCLUSIVE,
                             "no rule checks in the results"))
        else:
            below = [r for r in rules if r["high"] < floor]
            unsure = [r for r in rules if r["rate"] < floor and r not in below]
            worst = min(rules, key=lambda r: r["rate"])
            rows.append(_row("min_rule_rate", f"lowest: {worst['rule']} {_pct(worst['rate'])} "
                             f"(CI {_pct(worst['low'])} to {_pct(worst['high'])})", f">= {_pct(floor)}",
                             FAIL if below else INCONCLUSIVE if unsure else PASS,
                             (f"below: {', '.join(r['rule'] for r in below)}; " if below else "")
                             + (f"can't tell yet: {', '.join(r['rule'] for r in unsure)}; " if unsure else "")
                             + f"{len(rules)} rules"))
    if "no_rule_regression" in req:
        m = req["no_rule_regression"]
        compared = [r for r in (rules or []) if "delta" in r]
        if not compared:
            rows.append(_row("no_rule_regression", "n/a", f"no rule proven < {_pts(-m)}", INCONCLUSIVE,
                             "no rule was checked in both the baseline and the candidate"))
        else:
            dropped = [r for r in compared if r["delta_high"] < -m]
            worst = min(compared, key=lambda r: r["delta"])
            rows.append(_row("no_rule_regression", f"worst: {worst['rule']} {_pts(worst['delta'])} "
                             f"(CI {_pts(worst['delta_low'])} to {_pts(worst['delta_high'])})",
                             f"no rule proven < {_pts(-m)}", FAIL if dropped else PASS,
                             (f"proven worse: {', '.join(r['rule'] for r in dropped)}; " if dropped else "")
                             + f"{len(compared)} rules vs baseline, {100 * compared[0]['conf']:.2f}% intervals "
                             "(Bonferroni)"))

    comparison = None
    graded_c = [a for a in candidate if a.valid]
    graded_b = [a for a in baseline if a.valid] if baseline is not None else None
    if not graded_c or (graded_b is not None and not graded_b and any(r in req for r in NEEDS_BASELINE)):
        who = "candidate" if not graded_c else "baseline"
        rows.append(_row("graded attempts", "0", "> 0", UNHEALTHY,
                         f"every {who} attempt failed for infrastructure reasons: "
                         f"{summ['infra']['kinds'] if who == 'candidate' else 'see the baseline'}"))
        req = {k: v for k, v in req.items() if k not in NEEDS_BASELINE and k != "min_pass_hat_k"}
    if "no_regression" in req:
        m = req["no_regression"]
        comparison = analysis.compare(baseline, candidate, policy.design, policy.cluster_by, conf)
        d = comparison["delta"]
        measured = f"{_pts(d['estimate'])} (CI {_pts(d['low'])} to {_pts(d['high'])}, {comparison['method']})"
        shock = comparison.get("shock") or {}
        if shock.get("detected") and not shock.get("fallback"):
            arms = ", ".join("baseline" if a == "A" else "candidate" for a in shock["arms"])
            rows.append(_row("no_regression", measured, f"lower bound >= {_pts(-m)}", INCONCLUSIVE,
                             f"run-wide shock in the {arms} runs (whole runs vary more than per-case randomness "
                             "predicts) with too few runs to correct for it: rerun"))
        elif d["low"] is None or math.isnan(d["low"]):
            rows.append(_row("no_regression", measured, f"lower bound >= {_pts(-m)}", INCONCLUSIVE,
                             runs_needed(d, m, runs)))
        elif d["low"] >= -m:
            detail = "a real but small drop, inside the margin" if d["high"] < 0 else ""
            rows.append(_row("no_regression", measured, f"lower bound >= {_pts(-m)}", PASS, detail))
        elif d["high"] < -m:
            rows.append(_row("no_regression", measured, f"lower bound >= {_pts(-m)}", FAIL,
                             "worse than the baseline by more than the margin"))
        else:
            rows.append(_row("no_regression", measured, f"lower bound >= {_pts(-m)}", INCONCLUSIVE,
                             runs_needed(d, m, min(comparison["a"]["runs"], comparison["b"]["runs"]))))
        notes += [f"Comparison: {n}" for n in comparison["notes"]]

    if "no_significant_drop" in req:
        m = req["no_significant_drop"]
        comparison = comparison or analysis.compare(baseline, candidate, policy.design, policy.cluster_by, conf)
        d = comparison["delta"]
        measured = f"{_pts(d['estimate'])} (CI {_pts(d['low'])} to {_pts(d['high'])}, {comparison['method']})"
        sd = comparison.get("smallest_detectable")
        catch = f"this run reliably catches drops of about {100 * sd:.1f} points or more" if sd else ""
        shock = comparison.get("shock") or {}
        if d["high"] is None or math.isnan(d["high"]):
            rows.append(_row("no_significant_drop", measured, f"upper bound >= {_pts(-m)}", INCONCLUSIVE,
                             runs_needed(d, m, runs)))
        elif d["high"] < -m:
            rows.append(_row("no_significant_drop", measured, f"upper bound >= {_pts(-m)}", FAIL,
                             "a real drop: the whole interval is below the threshold"))
        elif shock.get("detected") and not shock.get("fallback"):
            rows.append(_row("no_significant_drop", measured, f"upper bound >= {_pts(-m)}", INCONCLUSIVE,
                             "run-wide shock with too few runs to correct for it: rerun"))
        else:
            rows.append(_row("no_significant_drop", measured, f"upper bound >= {_pts(-m)}", PASS, catch))
        if "no_regression" not in req:
            notes += [f"Comparison: {n}" for n in comparison["notes"]]

    graded_cand = [a for a in candidate if a.valid]
    na = analysis.no_answer_stats(graded_cand)
    na_base = analysis.no_answer_stats([a for a in baseline if a.valid]) if baseline is not None else None
    if "max_no_answer" in req:
        detail = (f"{na['count']} attempts: {na['kinds']}" if na["count"] else "")
        if na_base is not None:
            detail += f"{'; ' if detail else ''}baseline {_pct(na_base['rate'])}"
        rows.append(_row("max_no_answer", _pct(na["rate"]), f"<= {_pct(req['max_no_answer'])}",
                         PASS if na["rate"] <= req["max_no_answer"] else FAIL, detail))
    elif na["count"] and (na_base is None or na["rate"] > na_base["rate"]):
        notes.append(f"{na['count']} candidate attempts ended with no usable answer ({_pct(na['rate'])}: "
                     f"{na['kinds']})" + (f", baseline {_pct(na_base['rate'])}" if na_base is not None else "")
                     + ". They count as failures; a jump usually means the harness or loop broke.")

    slices = None
    if tier.slice_by and baseline is not None:
        slices = slice_comparisons(baseline, candidate, tier.slice_by, conf)
        if "max_slice_drop" in req and slices:
            m = req["max_slice_drop"]
            worst = min(slices, key=lambda x: x["delta"])
            dropped = [x for x in slices if x["high"] < -m]
            leads = [x for x in slices if x["delta"] < -m and x not in dropped]
            detail = (f"proven worse: {', '.join(x['slice'] for x in dropped)}; " if dropped else "")
            detail += (f"estimated beyond the limit but not proven (few cases): "
                       f"{', '.join(x['slice'] + ' ' + _pts(x['delta']) for x in leads)}; " if leads else "")
            detail += f"{len(slices)} slices by {tier.slice_by}, {100 * slices[0]['conf']:.2f}% intervals (Bonferroni)"
            rows.append(_row("max_slice_drop", f"worst: {worst['slice']} {_pts(worst['delta'])} "
                             f"(CI {_pts(worst['low'])} to {_pts(worst['high'])})", f"no slice proven < {_pts(-m)}",
                             FAIL if dropped else PASS, detail))

    if "max_cost_ratio" in req:
        rows.append(_upper_bound_check("max_cost_ratio", _ratio_interval(baseline, candidate, "cost_usd", mean, conf),
                                       req["max_cost_ratio"], _x))
    if "max_p95_latency_ratio" in req:
        iv = _ratio_interval(baseline, candidate, "latency_s", lambda xs: quantile(xs, 0.95), conf)
        rows.append(_upper_bound_check("max_p95_latency_ratio", iv, req["max_p95_latency_ratio"], _x))
    if "min_pass_hat_k" in req:
        rel = summ.get("reliability") or {}
        k = min(tier.runs, rel.get("k_max", 0))
        rows.append(_lower_bound_check(f"min_pass_hat_k (k={k or tier.runs})",
                                       (rel.get("pass_hat") or {}).get(k) if k >= 2 else None,
                                       req["min_pass_hat_k"], _pct))

    # What a small tier can't see: say so, so nobody reads a green smoke run as a regression check.
    if tier.sample and summ["cases"]:
        p = summ["pass_rate"]["estimate"]
        mde = 2.8 * math.sqrt(max(2 * p * (1 - p), 1e-9) / (summ["cases"] * max(runs, 1)))
        notes.append(f"With {summ['cases']} cases and {runs} run(s), this tier can only see drops larger than "
                     f"about {100 * mde:.0f} points: it catches breakage, not regressions.")

    results = {r["result"] for r in rows}
    overall = (UNHEALTHY if UNHEALTHY in results else FAIL if FAIL in results
               else INCONCLUSIVE if INCONCLUSIVE in results else PASS)
    return {"tier": tier.name, "overall": overall, "exit_code": EXIT[overall], "rows": rows, "notes": notes,
            "cases": summ["cases"], "runs": runs, "pass_rate": summ["pass_rate"], "comparison": comparison,
            "slices": slices, "slice_by": tier.slice_by, "no_answer": na, "rules": rules}


def render(g: dict) -> str:
    icon = {PASS: "✅", FAIL: "❌", INCONCLUSIVE: "⚠️", UNHEALTHY: "🚧"}
    lines = [f"## agentstats gate: {g['tier']} {icon[g['overall']]} {g['overall']}", "",
             f"{g['cases']} cases, {g['runs']} run(s).", "",
             "| requirement | measured | threshold | result | detail |", "|---|---|---|---|---|"]
    for r in g["rows"]:
        lines.append(f"| {r['requirement']} | {r['measured']} | {r['threshold']} | {icon[r['result']]} {r['result']} "
                     f"| {r['detail']} |")
    if g["notes"]:
        lines.append("")
        lines += [f"- {n}" for n in g["notes"]]
    if g.get("rules"):
        lines += ["", "Trajectory rules (how often each held; vs baseline where both checked it):", "",
                  "| rule | attempts | held | baseline | change (CI) |", "|---|---|---|---|---|"]
        for r in sorted(g["rules"], key=lambda r: r.get("delta", 0)):
            change = (f"{_pts(r['delta'])} ({_pts(r['delta_low'])} to {_pts(r['delta_high'])})" if "delta" in r
                      else "n/a")
            base = _pct(r["baseline_rate"]) if "baseline_rate" in r else "n/a"
            lines.append(f"| {r['rule']} | {r['attempts']} | {_pct(r['rate'])} | {base} | {change} |")
    if g.get("slices"):
        lines += ["", f"Per slice ({g['slice_by']}), candidate vs baseline, "
                      f"{100 * g['slices'][0]['conf']:.2f}% intervals (Bonferroni across slices):", "",
                  "| slice | cases | baseline | candidate | change (CI) |", "|---|---|---|---|---|"]
        for x in sorted(g["slices"], key=lambda x: x["delta"]):
            lines.append(f"| {x['slice']} | {x['cases']} | {_pct(x['baseline'])} | {_pct(x['candidate'])} | "
                         f"{_pts(x['delta'])} ({_pts(x['low'])} to {_pts(x['high'])}) |")
    return "\n".join(lines) + "\n"
