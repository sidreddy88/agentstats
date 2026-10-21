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
from agentstats.intervals import Interval, bootstrap, mean, quantile
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

    comparison = None
    if "no_regression" in req:
        m = req["no_regression"]
        comparison = analysis.compare(baseline, candidate, policy.design, policy.cluster_by, conf)
        d = comparison["delta"]
        measured = f"{_pts(d['estimate'])} (CI {_pts(d['low'])} to {_pts(d['high'])}, {comparison['method']})"
        if d["low"] is None or math.isnan(d["low"]):
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
            "cases": summ["cases"], "runs": runs, "pass_rate": summ["pass_rate"], "comparison": comparison}


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
    return "\n".join(lines) + "\n"
