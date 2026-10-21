"""Plain-text/markdown rendering of summary, compare and judge results."""
from __future__ import annotations

import math

METHOD_NAME = {
    "fixed_cases": "fixed cases, noise from repeated trials per case",
    "run_level": "Student's t over complete runs",
    "per_case": "t over per-case pass rates",
    "clustered": "cluster-robust standard error",
    "wilson": "Wilson interval",
    "welch": "Welch's t-test over complete runs",
    "paired": "paired per-case differences",
    "paired_clustered": "paired differences, cluster-robust",
    "mcnemar": "exact McNemar test (one trial per case per arm)",
}


WHEN_USED = {
    "fixed_cases": "for hand-picked sets when runs don't show run-wide shocks",
    "run_level": "for hand-picked sets with 3+ runs that show run-wide shocks",
    "per_case": "when the cases were sampled at random from real traffic",
    "clustered": "for sampled cases in groups that fail together",
}


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _pp(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f} pts"


def _ci(iv: dict, fmt=_pct) -> str:
    return f"{fmt(iv['estimate'])} (95% CI {fmt(iv['low'])} to {fmt(iv['high'])})"


def _num(x: float) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    return f"{x:,.4f}" if abs(x) < 1 else f"{x:,.1f}"


def summary(s: dict) -> str:
    pr = s["pass_rate"]
    lines = [f"# agentstats summary ({s['design']} eval set)", "",
             f"**Pass rate: {_ci(pr)}**, {s['claim']}.",
             f"Method: {METHOD_NAME.get(pr['method'], pr['method'])}. {s['cases']} cases, {s['runs']} run(s), "
             f"{s['attempts']} graded attempts used ({s['attempts_loaded']} loaded)."
             + (f" Interval {pr['note']}." if "clipped" in pr.get("note", "") else ""), ""]
    for m, iv in s.get("not_used", {}).items():
        lines.append(f"- Not used: {METHOD_NAME.get(m, m)} would give {_ci(iv)} ({WHEN_USED.get(m, '')}).")
    for n in s["notes"]:
        lines.append(f"- Note: {n}")
    if s.get("run_pass_rates") and len(s["run_pass_rates"]) > 1:
        lines.append("- Per run: " + ", ".join(f"run {r} {_pct(v)}" for r, v in s["run_pass_rates"].items()))
    inf = s["infra"]
    lines += ["", "## Infrastructure",
              f"{inf['count']} infrastructure failures ({_pct(inf['rate'])}), excluded from pass rates"
              + (f": {inf['kinds']}" if inf["kinds"] else "") + "."]
    if "reliability" in s:
        r = s["reliability"]
        lines += ["", "## Reliability", "| k | pass@k (capability) | pass^k (consistency) |", "|---|---|---|"]
        for k in r["pass_at"]:
            lines.append(f"| {k} | {_ci(r['pass_at'][k])} | {_ci(r['pass_hat'][k])} |")
        c = r["cases"]
        lines.append(f"\nAlways pass: {len(c['always'])}, never pass: {len(c['never'])}, "
                     f"flaky: {len(c['flaky'])}" + (f" ({', '.join(c['flaky'][:10])}"
                                                     + (", ..." if len(c['flaky']) > 10 else "") + ")"
                                                     if c["flaky"] else ""))
    if s["efficiency"]:
        lines += ["", "## Efficiency (per attempt, mean over cases)", "| metric | mean (95% CI) | p50 | p95 |",
                  "|---|---|---|---|"]
        for f, v in s["efficiency"].items():
            if f == "cost_per_resolved_task":
                continue
            lines.append(f"| {f} | {_ci(v['mean'], _num)} | {_num(v['p50'])} | {_num(v['p95'])} |")
        if "cost_per_resolved_task" in s["efficiency"]:
            lines.append(f"\nCost per resolved task: ${_ci(s['efficiency']['cost_per_resolved_task'], _num)}")
    if s.get("slices"):
        lines += ["", "## Slices", "| tag | cases | pass rate |", "|---|---|---|"]
        for t, v in s["slices"].items():
            lines.append(f"| {t} | {v['cases']} | {_ci(v['pass_rate'])} |")
    if s.get("checks"):
        lines += ["", "## Trajectory checks", "| rule | pass rate |", "|---|---|"]
        for r, v in s["checks"].items():
            lines.append(f"| {r} | {_ci(v)} |")
    return "\n".join(lines) + "\n"


def compare(c: dict, a_name: str = "A", b_name: str = "B") -> str:
    d = c["delta"]
    lines = [f"# agentstats compare: {b_name} vs {a_name} ({c['design']} eval set)", "",
             f"**{c['verdict']}**: {b_name} minus {a_name} = {_ci(d, _pp)}, {c['claim']}.",
             f"Method: {METHOD_NAME.get(c['method'], c['method'])}. {c['cases']} matched cases; "
             f"{a_name} {_pct(c['a']['pass_rate'])} over {c['a']['runs']} run(s), "
             f"{b_name} {_pct(c['b']['pass_rate'])} over {c['b']['runs']} run(s).", ""]
    if c.get("mcnemar"):
        m = c["mcnemar"]
        lines.append(f"- McNemar (exact): {m['b_only']} cases pass only with {b_name}, {m['a_only']} only with "
                     f"{a_name}; p = {m['p_value']:.3g}.")
    if c.get("smallest_detectable") is not None:
        lines.append(f"- Smallest difference this setup can reliably detect (80% power): "
                     f"about {_pp(c['smallest_detectable'])}.")
    if c.get("min_effect") is not None:
        lines.append(f"- Ship rule (CI lower bound > {_pp(c['min_effect'])}): {'PASS' if c['ship'] else 'FAIL'}.")
    for n in c["notes"]:
        lines.append(f"- Note: {n}")
    mv = c["moved"]
    lines.append(f"- Cases that moved: {len(mv['b_better'])} better with {b_name}, {len(mv['b_worse'])} worse.")
    if "cost_per_attempt_delta" in c:
        lines.append(f"- Cost per attempt: {a_name} ${_num(c['cost_per_attempt']['a'])}, "
                     f"{b_name} ${_num(c['cost_per_attempt']['b'])}; difference "
                     f"${_ci(c['cost_per_attempt_delta'], _num)}.")
    return "\n".join(lines) + "\n"


def judge(j: dict) -> str:
    lines = ["# agentstats judge check", "", f"{j['items']} items."]
    if "human_agreement" in j:
        h = j["human_agreement"]
        lines += ["", "## Human agreement",
                  f"Fleiss' kappa {h['fleiss_kappa']:.2f} over {h['items']} items with {h['raters_per_item']} "
                  f"raters each; {h['unanimous']} unanimous."]
    if "alignment" in j:
        a = j["alignment"]

        def r(key):
            iv = a["ci"].get(key)
            return "n/a" if iv is None else _ci(iv)
        lines += ["", f"## Judge vs human majority ({a['scope']}, n = {a['n']})",
                  f"- Accuracy {r('accuracy')}",
                  f"- Strictness, false fails among human passes: {r('strictness')}",
                  f"- Leniency, false passes among human fails: {r('leniency')}",
                  f"- Cohen's kappa {a['cohen_kappa']:.2f}",
                  f"TP {a['tp']}, FN {a['fn']}, FP {a['fp']}, TN {a['tn']}."]
        if a["disagreements"]:
            lines.append("Disagreements: " + ", ".join(f"{x['item']} (human {x['human']}, judge {x['judge']})"
                                                        for x in a["disagreements"][:10]))
    if "consistency" in j:
        c = j["consistency"]
        lines += ["", "## Judge consistency",
                  f"{c['items']} items judged {c['runs_per_item']}+ times: flip rate {100 * c['flip_rate']:.1f}%"
                  + (f", mean pairwise disagreement {c['mean_pairwise_disagreement']:.3f}"
                     if "mean_pairwise_disagreement" in c else
                     f", mean pairwise |delta| {c['mean_pairwise_delta']:.3f}") + "."]
    return "\n".join(lines) + "\n"
