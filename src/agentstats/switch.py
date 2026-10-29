"""
The switch decision: is the cheaper setup shown to be no worse than the current one, at no higher cost?

`agentstats switch` doesn't run agents; the statistics are agentstats' own gate:
- results come in as agentstats attempt rows (your runner, or `harness-evolve export` for an adapted harness);
- agentstats does the statistics (non-inferiority on fixed cases, cost ratio, infrastructure failures);
- this module applies the switch rule, in steps, and switch_record.py writes down what it decided and why.

The rule, for one candidate vs the current setup, with a margin m chosen in advance:

    lower CI bound of (candidate - current) >= -m, and cost ratio upper bound <= max   -> SWITCH
    upper CI bound < -m (worse than the margin even at the optimistic end), or costs more -> KEEP
    anything else (the interval crosses -m; too few runs; a run-wide shock)            -> NOT YET

The steps:
    1. the cheaper model as-is (same harness as the current setup);
    2. if adapted results are given (the cheaper model with a harness adapted to it, e.g. by harness-evolve,
       measured on held-out cases), the adapted setup. The better of the two decisions is the result.
If step 1 is already SWITCH, adapted results are still compared and reported: switching and improving the new model
are separate decisions (from the diagnosis replay, where adapting after a passing switch added ~9 points).

Slices (--slice-by <tag prefix>): every comparison is repeated per slice. By default slices are reported, not
enforced, because small slices give wide intervals (agentstats warns below 100 cases). With strict slices, the
decision can't be better than its worst slice: one slice KEEP makes it KEEP, one slice NOT YET makes it NOT YET.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agentstats import gate
from agentstats.policy import parse
from agentstats.records import Attempt

SWITCH, KEEP, NOT_YET, UNHEALTHY = "SWITCH", "KEEP", "NOT YET", "UNHEALTHY"
ORDER = {SWITCH: 0, NOT_YET: 1, KEEP: 2, UNHEALTHY: 3}       # worse decisions sort later
EXIT = {SWITCH: 0, KEEP: 1, NOT_YET: 2, UNHEALTHY: 3}
PROVENANCE = ("independent", "current-model", "unknown")


@dataclass
class Rule:
    margin: float = 0.03            # how much worse is acceptable, as a fraction (0.03 = 3 points)
    max_cost_ratio: float = 1.0     # candidate cost per attempt / current, upper CI bound
    conf: float = 0.95
    max_infra: float = 0.05
    design: str = "curated"


@dataclass
class Comparison:
    name: str
    decision: str
    reason: str
    delta: dict | None = None        # agentstats' delta: estimate, low, high
    cost_ratio: str | None = None
    gate_rows: list[dict] = field(default_factory=list)
    slices: dict[str, Comparison] = field(default_factory=dict)


def _has_cost(attempts: list[Attempt]) -> bool:
    return any("cost_usd" in a.metrics for a in attempts if a.valid)


def compare(name: str, current: list[Attempt], candidate: list[Attempt], rule: Rule) -> Comparison:
    """One candidate vs the current setup, through agentstats' gate, mapped to SWITCH / KEEP / NOT YET."""
    require = {"no_regression": rule.margin, "max_infra": rule.max_infra}
    with_cost = _has_cost(current) and _has_cost(candidate)
    if with_cost:
        require["max_cost_ratio"] = rule.max_cost_ratio
    policy = parse({"design": rule.design, "switch": {"runs": 1, "require": require}})
    g = gate.evaluate(policy.tiers["switch"], policy, candidate, current, rule.conf)
    rows = {r["requirement"]: r for r in g["rows"]}
    nr, cost = rows.get("no_regression"), rows.get("max_cost_ratio")
    delta = (g.get("comparison") or {}).get("delta")
    c = Comparison(name, NOT_YET, "", delta, cost["measured"] if cost else None, g["rows"])

    if g["overall"] == gate.UNHEALTHY:
        c.decision, c.reason = UNHEALTHY, "too many infrastructure failures to judge: rerun"
    elif nr is None:
        c.decision, c.reason = NOT_YET, "no comparison was possible"
    elif nr["result"] == gate.FAIL:
        c.decision, c.reason = KEEP, f"worse than the margin even at the optimistic end: {nr['measured']}"
    elif cost and cost["result"] == gate.FAIL:
        c.decision, c.reason = KEEP, f"not cheaper: cost ratio {cost['measured']}"
    elif nr["result"] == gate.PASS and (cost is None or cost["result"] == gate.PASS):
        c.decision = SWITCH
        c.reason = f"shown no more than {100 * rule.margin:g} points worse: {nr['measured']}"
        c.reason += f"; cost {cost['measured']}" if cost else "; no cost data, cost not checked"
    else:
        why = nr["detail"] if nr["result"] != gate.PASS else f"cost: {cost['detail'] or cost['measured']}"
        c.decision, c.reason = NOT_YET, f"can't tell yet ({nr['measured']}): {why}"
    return c


def _slices(attempts: list[Attempt], prefix: str) -> dict[str, list[Attempt]]:
    out: dict[str, list[Attempt]] = {}
    for a in attempts:
        for t in a.tags:
            if t.startswith(prefix + ":"):
                out.setdefault(t[len(prefix) + 1:], []).append(a)
    return out


def compare_with_slices(name: str, current: list[Attempt], candidate: list[Attempt], rule: Rule,
                        slice_by: str | None) -> Comparison:
    c = compare(name, current, candidate, rule)
    if slice_by:
        cur, cand = _slices(current, slice_by), _slices(candidate, slice_by)
        for s in sorted(set(cur) & set(cand)):
            c.slices[s] = compare(f"{name} [{slice_by}:{s}]", cur[s], cand[s], rule)
    return c


def cost_per_accepted(attempts: list[Attempt]) -> dict:
    valid = [a for a in attempts if a.valid]
    passed = sum(1 for a in valid if a.passed)
    out = {"attempts": len(valid), "pass_rate": passed / len(valid) if valid else None}
    if _has_cost(attempts):
        cost = sum(a.metrics.get("cost_usd", 0.0) for a in valid)
        out["cost_per_attempt_usd"] = cost / len(valid) if valid else None
        out["cost_per_accepted_usd"] = cost / passed if passed else None
    return out


@dataclass
class Decision:
    decision: str
    reason: str
    steps: list[Comparison]
    arms: dict[str, dict]
    warnings: list[str]


def decide(current: list[Attempt], candidate: list[Attempt], adapted: list[Attempt] | None, rule: Rule, *,
           slice_by: str | None = None, strict_slices: bool = False, provenance: str = "unknown",
           preregistered: bool = False) -> Decision:
    if provenance not in PROVENANCE:
        raise ValueError(f"provenance must be one of {PROVENANCE}")
    steps = [compare_with_slices("as-is", current, candidate, rule, slice_by)]
    if adapted is not None:
        steps.append(compare_with_slices("adapted", current, adapted, rule, slice_by))

    def effective(c: Comparison) -> tuple[str, str]:
        """A step's decision; with strict slices, no better than its worst slice."""
        if strict_slices and c.slices:
            worst = max(c.slices.values(), key=lambda s: ORDER[s.decision])
            if ORDER[worst.decision] > ORDER[c.decision]:
                return worst.decision, f"strict slices: {worst.name}: {worst.reason}"
        return c.decision, f"{c.name}: {c.reason}"

    # The better of the two steps decides (the adapted step on a tie). Two chances to pass make a false SWITCH a
    # little more likely than the nominal rate; with a 95% interval on each, at most about twice as likely.
    results = [(effective(c), c) for c in steps]
    (decision, reason), final = min(reversed(results), key=lambda x: ORDER[x[0][0]])

    warnings = []
    if not strict_slices and final.slices:
        blocking = [s.name for s in final.slices.values() if ORDER[s.decision] > ORDER[decision]]
        if blocking:
            warnings.append(f"slices that don't support the decision (reported, not enforced): {', '.join(blocking)}")
    if steps[0].decision == SWITCH and len(steps) > 1 and steps[1].delta and steps[0].delta:
        gain = steps[1].delta["estimate"] - steps[0].delta["estimate"]
        if gain > 0:
            warnings.append(f"switching as-is is enough, but the adapted harness measured {100 * gain:+.1f} points "
                            "more than the as-is model: consider shipping it too (a separate decision)")
    if not (_has_cost(current) and _has_cost(candidate)):
        warnings.append("cost not checked: the current or candidate rows have no cost_usd")
    if provenance == "current-model":
        warnings.append("the eval set was created, filtered or labelled by the current model, so the comparison "
                        "favours it: a SWITCH still holds, a KEEP or NOT YET may overstate the gap")
    elif provenance == "unknown":
        warnings.append("eval-set provenance not stated: if the current model shaped the cases, the comparison "
                        "favours it")
    if not preregistered:
        warnings.append("the margin was not declared before the results were seen")

    arms = {"current": cost_per_accepted(current), "candidate": cost_per_accepted(candidate)}
    if adapted is not None:
        arms["adapted"] = cost_per_accepted(adapted)
    return Decision(decision, reason, steps, arms, warnings)
