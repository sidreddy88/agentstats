"""
The decision record: what was compared, under which rule, what was decided and why, written next to the inputs'
hashes so the decision can be checked later ("why are we still on the expensive model?").

    <out>/decision.json    everything, machine-readable
    <out>/decision.md      the same, for people
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import agentstats
from agentstats.switch import Comparison, Decision, Rule


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _comparison(c: Comparison) -> dict:
    return {"name": c.name, "decision": c.decision, "reason": c.reason, "delta": c.delta, "cost_ratio": c.cost_ratio,
            "gate_rows": c.gate_rows, "slices": {k: _comparison(v) for k, v in c.slices.items()}}


def build(d: Decision, rule: Rule, inputs: dict[str, Path], *, name: str, provenance: str, preregistered: bool,
          slice_by: str | None, strict_slices: bool, note: str = "", prereg: dict | None = None) -> dict:
    return {
        "name": name,
        "decision": d.decision,
        "reason": d.reason,
        "warnings": d.warnings,
        "rule": asdict(rule),
        "margin_declared_before_results": preregistered,
        "preregistration": prereg,
        "provenance": provenance,
        "slices": {"by": slice_by, "strict": strict_slices},
        "inputs": {arm: {"path": str(p), "sha256": _sha256(p)} for arm, p in inputs.items()},
        "arms": d.arms,
        "steps": [_comparison(c) for c in d.steps],
        "note": note,
        "versions": {"agentstats": agentstats.__version__},
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _pts(x):
    return "n/a" if x is None else f"{100 * x:+.1f}"


def render(r: dict) -> str:
    lines = [f"# agentstats switch decision: {r['name']}", "", f"**{r['decision']}**: {r['reason']}", ""]
    rule = r["rule"]
    rule_line = (f"Rule: switch only if shown no more than {100 * rule['margin']:g} points worse "
                 f"({100 * rule['conf']:g}% interval, {rule['design']} eval set) at a cost ratio of at most "
                 f"{rule['max_cost_ratio']:g}x. Margin declared before the results: "
                 f"{'yes' if r['margin_declared_before_results'] else 'no'}. Eval-set provenance: {r['provenance']}.")
    lines += [rule_line, ""]
    p = r.get("preregistration")
    if p and p.get("file"):
        status = (f"verified: committed {p.get('registered_at')} ({p.get('commit', '')[:7]}), before the results"
                  if p["verified"] else "NOT verified: " + "; ".join(p["reasons"]))
        lines += [f"Pre-registration `{p['file']}`: {status}.", ""]
    elif p:
        lines += ["Pre-registration: asserted with --preregistered, not checked.", ""]
    lines += ["| step | decision | change (pts) | 95% CI | cost ratio |", "|---|---|---|---|---|"]
    for s in r["steps"]:
        d = s["delta"] or {}
        lines.append(f"| {s['name']} | {s['decision']} | {_pts(d.get('estimate'))} | "
                     f"{_pts(d.get('low'))} to {_pts(d.get('high'))} | {s['cost_ratio'] or 'n/a'} |")
        for sl in s["slices"].values():
            d = sl["delta"] or {}
            lines.append(f"| &nbsp;&nbsp;{sl['name']} | {sl['decision']} | {_pts(d.get('estimate'))} | "
                         f"{_pts(d.get('low'))} to {_pts(d.get('high'))} | {sl['cost_ratio'] or 'n/a'} |")
    lines += ["", "| arm | attempts | pass rate | $ per attempt | $ per accepted result |", "|---|---|---|---|---|"]
    for arm, a in r["arms"].items():
        pr = f"{100 * a['pass_rate']:.1f}%" if a.get("pass_rate") is not None else "n/a"
        cpa = f"{a['cost_per_attempt_usd']:.4f}" if a.get("cost_per_attempt_usd") is not None else "n/a"
        cpr = f"{a['cost_per_accepted_usd']:.4f}" if a.get("cost_per_accepted_usd") is not None else "n/a"
        lines.append(f"| {arm} | {a['attempts']} | {pr} | {cpa} | {cpr} |")
    if r["warnings"]:
        lines += ["", "Warnings:"] + [f"- {w}" for w in r["warnings"]]
    if r["note"]:
        lines += ["", f"Note: {r['note']}"]
    lines += ["", "Inputs:"] + [f"- {arm}: `{i['path']}` (sha256 {i['sha256']})" for arm, i in r["inputs"].items()]
    decided = f"Decided {r['decided_at']} with agentstats {r['versions']['agentstats']}."
    lines += ["", decided]
    return "\n".join(lines) + "\n"


def write(out_dir: Path, r: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j, m = out_dir / "decision.json", out_dir / "decision.md"
    j.write_text(json.dumps(r, indent=1))
    m.write_text(render(r))
    return j, m
