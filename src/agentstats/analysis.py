"""
Summaries and comparisons that pick their statistics from how the eval set was built, and say which claim the
numbers support.

Eval set designs (after Yesha Shah, "Agentic Evals" Part 2, and Miller 2024, arXiv 2411.00640):
  curated    Hand-picked cases (most eval sets and benchmarks). Cases aren't a random sample of anything bigger,
             so the claim is "on this eval set": the cases are fixed and the only randomness is the agent's own,
             measured from repeated trials of each case (comparisons.fixed_cases). If whole runs vary more than
             that per-case randomness predicts (a run-wide shock, such as a provider's bad hour), it falls back
             to treating each complete run as one observation: Student's t over runs, Welch's t to compare.
  sampled    Cases drawn at random from real traffic, unrelated to each other. Each case is an observation:
             per-case t interval, paired per-case differences to compare. Claim: "on the population the cases
             were drawn from".
  clustered  Sampled, but cases come in groups that fail together (same feature). Cluster-robust standard
             errors over groups. Needs every case in exactly one group; overlapping groups can't be modelled
             credibly, so it falls back to run-level statistics.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from scipy import stats

from agentstats import comparisons, reliability
from agentstats.intervals import Interval, bootstrap, clustered, mean, quantile, t_interval, wilson
from agentstats.records import NUMERIC_FIELDS, Attempt, by_case, by_run

DESIGNS = ("curated", "sampled", "clustered")
INFRA_WARN = 0.05

CLAIM = {
    "fixed_cases": "on this eval set",
    "run_level": "on this eval set",
    "per_case": "on the population the cases were sampled from",
    "clustered": "on the population the case groups were sampled from",
    "wilson": "on this eval set, from a single run (indicative: run-to-run noise is unmeasured)",
    "welch": "on this eval set",
    "paired": "on the population the cases were sampled from",
    "paired_clustered": "on the population the case groups were sampled from",
    "mcnemar": "on this eval set, from one trial per case per arm (assumes no run-wide shocks, which one run "
               "can't check)",
}


def _clusters(cases: Sequence[str], case_tags: dict[str, list[str]],
              cluster_by: str | None) -> tuple[list[str] | None, str]:
    if not cluster_by:
        return None, "clustered design needs --cluster-by (a tag prefix such as 'feature')"
    out, overlapping, missing = [], [], []
    for c in cases:
        tags = case_tags.get(c, [])
        vals = [t for t in tags if t.startswith(cluster_by + ":")] if cluster_by != "tag" else tags
        if not vals:
            missing.append(c)
        elif len(set(vals)) > 1:
            overlapping.append(c)
        else:
            out.append(vals[0])
    if missing:
        return None, f"{len(missing)} cases have no '{cluster_by}' group (e.g. {missing[0]})"
    if overlapping:
        return None, (f"{len(overlapping)} cases belong to more than one group (e.g. {overlapping[0]}); "
                      "overlapping groups can't be modelled credibly")
    return out, ""


def _clip(iv: Interval, low: float, high: float) -> None:
    """Pass rates live in [0, 1] and their differences in [-1, 1]; a t interval from few runs can spill over."""
    if iv.low == iv.low and iv.low < low:
        iv.low = low
        iv.note = (iv.note + "; " if iv.note else "") + "clipped to the possible range"
    if iv.high == iv.high and iv.high > high:
        iv.high = high
        if "clipped" not in iv.note:
            iv.note = (iv.note + "; " if iv.note else "") + "clipped to the possible range"


def _case_tags(attempts: list[Attempt]) -> dict[str, list[str]]:
    tags: dict[str, set[str]] = {}
    for a in attempts:
        tags.setdefault(a.case, set()).update(a.tags)
    return {c: sorted(t) for c, t in tags.items()}


def _common_cases(attempts: list[Attempt], notes: list[str], label: str = "") -> list[Attempt]:
    """Run-level pass rates only mean something if every run covers the same cases: keep the cases attempted
    in every run. An infrastructure failure doesn't remove a case; it only leaves that attempt out of its run."""
    runs = by_run(attempts)
    if len(runs) < 2:
        return attempts
    shared = set.intersection(*({a.case for a in atts} for atts in runs.values()))
    allc = {a.case for a in attempts}
    if shared != allc:
        notes.append(f"{label}Runs cover different cases; using the {len(shared)} cases attempted in every run "
                     f"(dropped {len(allc - shared)}).")
    return [a for a in attempts if a.case in shared]


def _pass_rates(valid: list[Attempt]) -> tuple[dict[int, float], dict[str, tuple[int, int]]]:
    runs = {r: mean([1.0 if a.passed else 0.0 for a in atts]) for r, atts in by_run(valid).items()}
    counts = {c: (len(atts), sum(1 for a in atts if a.passed)) for c, atts in by_case(valid).items()}
    return runs, counts


def _headline(design: str, runs: dict[int, float], counts: dict[str, tuple[int, int]], case_tags, cluster_by,
              conf: float) -> tuple[Interval, list[str], dict[str, Interval]]:
    """The interval the design supports, notes, and the alternatives that were not used (for the report)."""
    notes: list[str] = []
    k = len(runs)
    p = {c: pc / n for c, (n, pc) in counts.items()}
    total_n = sum(n for n, _ in counts.values())
    total_c = sum(c for _, c in counts.values())
    alt: dict[str, Interval] = {}
    if k >= 2:
        alt["run_level"] = t_interval(list(runs.values()), conf, method="run_level")
    if len(p) >= 2:
        alt["per_case"] = t_interval(list(p.values()), conf, method="per_case")

    if design == "clustered":
        clusters, why = _clusters(list(p), case_tags, cluster_by)
        if clusters is not None:
            iv = clustered(list(p.values()), clusters, conf)
            return iv, notes, alt
        notes.append(f"Can't use a clustered interval: {why}. Falling back to the curated (run-level) design.")
        design = "curated"
    if design == "sampled":
        if k == 1:
            notes.append("One run: per-case outcomes are binary, so a Wilson interval is used.")
            return wilson(total_c, total_n, conf), notes, alt
        if len(p) < 30:
            notes.append(f"Only {len(p)} cases: a per-case interval this small may be unreliable; add cases.")
        return alt["per_case"], notes, alt
    # curated
    if k >= 2:
        n = len(counts)
        var = sum((c / t) * (1 - c / t) / (t - 1) for t, c in counts.values() if t >= 2) / n ** 2
        se = math.sqrt(var)
        zc = float(stats.norm.ppf(0.5 + conf / 2))
        est = mean(list(p.values()))
        fixed = Interval(est, est - zc * se, est + zc * se, "fixed_cases", n, se=se)
        alt["fixed_cases"] = fixed
        shock = comparisons.overdispersion(list(runs.values()), counts)
        if shock and shock["p_value"] < 0.05:
            msg = (f"Whole runs vary {shock['ratio']:.1f}x more than per-case randomness predicts "
                   f"(p = {shock['p_value']:.3f}): a run-wide shock.")
            if k >= 3:
                notes.append(msg + " Using complete runs as observations instead.")
                return alt["run_level"], notes, alt
            notes.append(msg + " With fewer than 3 runs there's nothing safer to fall back to; treat this with care.")
        return fixed, notes, alt
    notes.append("One run of a hand-picked set: run-to-run noise is unmeasured. Run the eval at least 3 times "
                 "for an interval you can rely on.")
    return wilson(total_c, total_n, conf), notes, alt


def _efficiency(valid: list[Attempt], conf: float) -> dict:
    out: dict = {}
    cases = by_case(valid)
    for f in NUMERIC_FIELDS:
        per_case = [mean([a.metrics[f] for a in atts]) for atts in cases.values()
                    if all(f in a.metrics for a in atts)]
        if len(per_case) >= 2:
            out[f] = {"mean": t_interval(per_case, conf, method="per_case").to_json(),
                      "p50": quantile([a.metrics[f] for a in valid if f in a.metrics], 0.5),
                      "p95": quantile([a.metrics[f] for a in valid if f in a.metrics], 0.95)}
    if all("cost_usd" in a.metrics for a in valid) and any(a.passed for a in valid):
        units = [(sum(a.metrics["cost_usd"] for a in atts), sum(1 for a in atts if a.passed))
                 for atts in cases.values()]

        def cost_per_resolved(us):
            solved = sum(s for _, s in us)
            return float("nan") if solved == 0 else sum(c for c, _ in us) / solved
        out["cost_per_resolved_task"] = bootstrap(units, cost_per_resolved, conf,
                                                  method="bootstrap over cases").to_json()
    return out


def no_answer_stats(valid: list[Attempt]) -> dict:
    """Attempts that ended with no usable answer (step limit, unparseable output): counted as failures, shown apart."""
    na = [a for a in valid if a.no_answer]
    return {"count": len(na), "rate": len(na) / len(valid) if valid else 0.0,
            "kinds": dict(Counter(a.no_answer for a in na))}


def summarize(attempts: list[Attempt], design: str = "curated", cluster_by: str | None = None,
              conf: float = 0.95) -> dict:
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    pre_notes: list[str] = []
    in_scope = _common_cases(attempts, pre_notes)
    valid = [a for a in in_scope if a.valid]
    infra = [a for a in in_scope if a.infra_error]
    ungraded = [a for a in in_scope if not a.infra_error and a.passed is None]
    runs, counts = _pass_rates(valid)
    case_tags = _case_tags(attempts)
    iv, notes, alt = _headline(design, runs, counts, case_tags, cluster_by, conf)
    _clip(iv, 0.0, 1.0)
    for v in alt.values():
        _clip(v, 0.0, 1.0)
    notes[:0] = pre_notes

    out: dict = {
        "design": design, "attempts_loaded": len(attempts), "attempts": len(valid), "cases": len(counts),
        "runs": len(runs),
        "pass_rate": iv.to_json(), "claim": CLAIM.get(iv.method, ""), "notes": notes,
        "not_used": {m: v.to_json() for m, v in alt.items() if m != iv.method},
        "run_pass_rates": runs,
        "infra": {"count": len(infra), "rate": len(infra) / len(in_scope) if in_scope else 0.0,
                  "kinds": dict(Counter(a.infra_error for a in infra)), "ungraded": len(ungraded)},
        "efficiency": _efficiency(valid, conf),
        "no_answer": no_answer_stats(valid),
    }
    if out["no_answer"]["count"]:
        na = out["no_answer"]
        notes.append(f"{na['count']} of {len(valid) - sum(1 for a in valid if a.passed)} failures had no usable answer "
                     f"({na['kinds']}): counted as failures, but usually a harness or loop problem, not judgement.")
    if out["infra"]["rate"] > INFRA_WARN:
        notes.append(f"{out['infra']['rate']:.0%} of attempts were infrastructure failures (excluded). "
                     "Above 5%: check the environment before trusting the numbers.")
    if len(runs) >= 2:
        out["reliability"] = reliability.summarize(counts, conf)
    tags = sorted({t for ts in case_tags.values() for t in ts})
    if tags:
        out["slices"] = {}
        for t in tags:
            atts = [a for a in valid if t in a.tags]
            if atts:
                cases_t = len({a.case for a in atts})
                out["slices"][t] = {"cases": cases_t,
                                    "pass_rate": wilson(sum(1 for a in atts if a.passed), len(atts), conf).to_json()}
        if len(counts) < 100:
            notes.append("Fewer than 100 cases: per-slice intervals are wide; treat slices as leads, not results.")
    rules = sorted({r for a in valid for r in a.checks})
    if rules:
        out["checks"] = {r: wilson(sum(1 for a in valid if a.checks.get(r)),
                                   sum(1 for a in valid if r in a.checks), conf).to_json() for r in rules}
    return out


def compare(a_attempts: list[Attempt], b_attempts: list[Attempt], design: str = "curated",
            cluster_by: str | None = None, conf: float = 0.95, min_effect: float | None = None) -> dict:
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    notes: list[str] = []
    a_valid = [a for a in _common_cases(a_attempts, notes, "A: ") if a.valid]
    b_valid = [b for b in _common_cases(b_attempts, notes, "B: ") if b.valid]
    a_cases, b_cases = set(by_case(a_valid)), set(by_case(b_valid))
    matched = sorted(a_cases & b_cases)
    dropped = {"only_a": sorted(a_cases - b_cases), "only_b": sorted(b_cases - a_cases)}
    a_m = [x for x in a_valid if x.case in matched]
    b_m = [x for x in b_valid if x.case in matched]
    a_runs, a_counts = _pass_rates(a_m)
    b_runs, b_counts = _pass_rates(b_m)
    if not matched:
        raise ValueError("no cases graded in both arms")

    pa = {c: s / n for c, (n, s) in a_counts.items()}
    pb = {c: s / n for c, (n, s) in b_counts.items()}
    diffs = [pb[c] - pa[c] for c in matched]
    single = len(a_runs) == 1 and len(b_runs) == 1
    mcn = None
    if single:
        a1 = {x.case: bool(x.passed) for x in a_m}
        b1 = {x.case: bool(x.passed) for x in b_m}
        mcn = comparisons.mcnemar([a1[c] for c in matched], [b1[c] for c in matched])

    shock = {"detected": False, "fallback": False, "arms": {}}
    method_design = design
    if design == "clustered":
        clusters, why = _clusters(matched, _case_tags(a_attempts + b_attempts), cluster_by)
        if clusters is None:
            notes.append(f"Can't use a clustered comparison: {why}. Falling back to the curated design.")
            method_design = "curated"
        else:
            iv = comparisons.paired_clustered(diffs, clusters, conf)
    if method_design == "sampled":
        iv = comparisons.paired(diffs, conf)
    elif method_design == "curated":
        iv = comparisons.fixed_cases(a_counts, b_counts, conf)
        if single:
            iv.method = "mcnemar"
        shocks = {arm: comparisons.overdispersion(list(r.values()), cnt)
                  for arm, r, cnt in (("A", a_runs, a_counts), ("B", b_runs, b_counts))}
        hit = {arm: s for arm, s in shocks.items() if s and s["p_value"] < 0.05}
        shock = {"detected": bool(hit), "fallback": False, "arms": hit}
        if hit:
            desc = "; ".join(f"{arm}: runs vary {s['ratio']:.1f}x more than per-case randomness predicts "
                             f"(p = {s['p_value']:.3f})" for arm, s in hit.items())
            if len(a_runs) >= 3 and len(b_runs) >= 3:
                iv = comparisons.welch(list(a_runs.values()), list(b_runs.values()), conf)
                shock["fallback"] = True
                notes.append(f"Run-wide shock ({desc}). Using complete runs as observations (Welch's t) instead.")
            else:
                notes.append(f"Run-wide shock ({desc}). With fewer than 3 runs per arm there's nothing safer to fall "
                             "back to; treat this with care.")

    _clip(iv, -1.0, 1.0)
    if iv.excludes_zero:
        verdict = "BETTER" if iv.estimate > 0 else "WORSE"
    else:
        verdict = "NO DETECTABLE DIFFERENCE"
    if single and mcn is not None and iv.method == "mcnemar":
        verdict = ("BETTER" if mcn["b_only"] > mcn["a_only"] else "WORSE") if mcn["p_value"] < 1 - conf \
            else "NO DETECTABLE DIFFERENCE"
    ship = None
    if min_effect is not None:
        ship = bool(iv.low > min_effect) if iv.low == iv.low else False

    infra_a = sum(1 for x in a_attempts if x.infra_error) / max(len(a_attempts), 1)
    infra_b = sum(1 for x in b_attempts if x.infra_error) / max(len(b_attempts), 1)
    if abs(infra_a - infra_b) > 0.05 or max(infra_a, infra_b) > INFRA_WARN:
        notes.append(f"Infrastructure failure rates differ or are high (A {infra_a:.0%}, B {infra_b:.0%}): "
                     "one arm may have hit an outage. Infra failures are excluded, but check before trusting this.")
    if dropped["only_a"] or dropped["only_b"]:
        notes.append(f"Compared on {len(matched)} cases graded in both arms; dropped "
                     f"{len(dropped['only_a'])} only in A and {len(dropped['only_b'])} only in B.")

    out = {
        "design": design, "method": iv.method, "claim": CLAIM.get(iv.method, ""),
        "a": {"pass_rate": mean(list(a_runs.values())), "runs": len(a_runs), "infra_rate": infra_a},
        "b": {"pass_rate": mean(list(b_runs.values())), "runs": len(b_runs), "infra_rate": infra_b},
        "cases": len(matched), "dropped": dropped,
        "delta": iv.to_json(), "verdict": verdict,
        "smallest_detectable": comparisons.smallest_detectable(iv, conf),
        "min_effect": min_effect, "ship": ship, "notes": notes, "shock": shock,
        "moved": {"b_better": sorted(c for c in matched if pb[c] > pa[c]),
                  "b_worse": sorted(c for c in matched if pb[c] < pa[c])},
    }
    if mcn:
        out["mcnemar"] = mcn
    if all("cost_usd" in x.metrics for x in a_m + b_m):
        ca = {c: mean([x.metrics["cost_usd"] for x in a_m if x.case == c]) for c in matched}
        cb = {c: mean([x.metrics["cost_usd"] for x in b_m if x.case == c]) for c in matched}
        out["cost_per_attempt_delta"] = comparisons.paired([cb[c] - ca[c] for c in matched], conf).to_json()
        out["cost_per_attempt"] = {"a": mean(list(ca.values())), "b": mean(list(cb.values()))}
    return out
