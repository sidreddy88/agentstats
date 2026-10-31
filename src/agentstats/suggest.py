"""
Suggest what to evaluate for each agent, from its trajectories, with evidence; the user accepts or rejects each item.

    agentstats suggest trajectories.jsonl --config-dir config/agents --out proposal.json  # print the checklist
    agentstats suggest-apply proposal.json --config-dir config/agents                     # accept item by item
    agentstats suggest-apply proposal.json --config-dir config/agents --accept 1,3,4      # or non-interactively

Input: one JSON object per agent run:
    {"agent": "DiagnosisAgent", "case": "trace-or-case-id", "passed": true | false | null,
     "tools": ["grep_codebase", "get_file_contents", "submit_diagnosis"], "end": "the run's final text (optional)"}
`passed` is null for production runs (no labels); graded runs (an eval or a CI gate) carry true/false.

For each agent, candidate behaviour rules are mined from the tool sequences (called(tool); before(a, b) for tools
that tend to appear together and in order) and kept when they hold often. With graded runs, each rule gets the pass
rate when it held vs when it was broken and a Newcombe interval on the difference:
- gate:    holds often, and breaking it is proven to hurt (the interval for the difference excludes 0 and its
           estimate is at least `min_effect`): worth a CI gate and monitoring;
- monitor: holds often, no proven effect on outcomes (or no graded runs): a drift signal, not a gate;
Rules that hold too rarely aren't proposed. Frequent final texts of runs without a pass become no-answer labels
to consider, and agents with no rules configured, no graded runs or very few runs get advice.

`suggest-apply` writes only accepted rule items, into <config-dir>/<agent-key>.json ({"agent", "rules": [...]}: the
format agent-platform's weekly monitoring reads); existing keys and rules are kept. Advice is listed, never written.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

from agentstats.checks import RULE_KINDS
from agentstats.intervals import newcombe, wilson

MIN_RUNS = 20            # fewer runs than this for an agent: too little to suggest rules from
MIN_HOLD = 0.85          # a candidate rule must hold at least this often to be proposed
MIN_SUPPORT = 0.30       # a tool must appear in at least this share of runs to be a candidate
MIN_BROKEN = 10          # graded runs breaking a rule needed before judging its effect
MIN_EFFECT = 0.20        # pass-rate drop (held - broken) needed for "gate"
MAX_MONITOR = 6          # at most this many monitor-only rules proposed per agent (strongest first)
MAX_INVARIANTS = 2       # of those, at most this many that held in every run (they carry the least information)


def load(paths: list[str | Path]) -> list[dict]:
    rows = []
    for p in paths:
        for i, line in enumerate(Path(p).read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "agent" not in r or not isinstance(r.get("tools"), list):
                raise ValueError(f"{p}:{i}: each run needs 'agent' and a 'tools' list")
            rows.append(r)
    return rows


def _key(agent: str) -> str:
    """Config file name for an agent: DiagnosisAgent -> diagnosis, ErrorClarityAgent -> error_clarity."""
    name = re.sub(r"Agent$", "", agent) or agent
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _holds(rule: dict, tools: list[str]) -> bool:
    if "called" in rule:
        return rule["called"] in tools
    a, b = rule["before"]
    return b not in tools or (a in tools and tools.index(a) < tools.index(b))


def _evidence(rule: dict, runs: list[dict], conf: float) -> dict:
    held = [r for r in runs if _holds(rule, r["tools"])]
    out = {"runs": len(runs), "held": len(held), "hold_rate": wilson(len(held), len(runs), conf).to_json()}
    graded = [r for r in runs if isinstance(r.get("passed"), bool)]
    if graded:
        g_held = [r for r in graded if _holds(rule, r["tools"])]
        g_broken = [r for r in graded if not _holds(rule, r["tools"])]
        out["graded"] = {"held": len(g_held), "broken": len(g_broken),
                         "pass_when_held": sum(r["passed"] for r in g_held) / len(g_held) if g_held else None,
                         "pass_when_broken": sum(r["passed"] for r in g_broken) / len(g_broken) if g_broken else None}
        if len(g_broken) >= MIN_BROKEN and g_held:
            # difference = pass rate when broken - when held (negative = breaking it hurts)
            iv = newcombe(sum(r["passed"] for r in g_held), len(g_held),
                          sum(r["passed"] for r in g_broken), len(g_broken), conf)
            out["graded"]["effect"] = iv.to_json()
    return out


def _classify(ev: dict) -> tuple[str, str]:
    g = ev.get("graded") or {}
    eff = g.get("effect")
    if eff and eff["high"] < 0 and -eff["estimate"] >= MIN_EFFECT:
        return "gate", (f"breaking it is proven to hurt: pass rate {100 * g['pass_when_broken']:.0f}% when broken vs "
                        f"{100 * g['pass_when_held']:.0f}% when held ({g['broken']} broken graded runs)")
    if eff:
        return "monitor", (f"no proven effect on outcomes: {100 * g['pass_when_broken']:.0f}% when broken vs "
                           f"{100 * g['pass_when_held']:.0f}% when held; a drift signal, not a gate")
    if g:
        return "monitor", f"too few graded runs break it ({g['broken']}) to judge its effect; monitor it"
    return "monitor", "no graded runs to judge its effect; monitor it"


def mine_rules(runs: list[dict], conf: float = 0.95) -> list[dict]:
    """Candidate rules for one agent's runs, with evidence: gates first (largest effect), then monitors."""
    n = len(runs)
    present = Counter(t for r in runs for t in set(r["tools"]))
    common = sorted(t for t, c in present.items() if c / n >= MIN_SUPPORT)
    candidates = [{"name": f"called_{t}", "called": t} for t in common]
    # A tool that is (almost) always the first step makes every "it before X" rule true at once: one fact, not many.
    first = Counter(r["tools"][0] for r in runs if r["tools"])
    always_first = {t for t, c in first.items() if c / n >= MIN_HOLD}
    for a, b in permutations(common, 2):
        if a in always_first:
            continue
        both = [r for r in runs if a in r["tools"] and b in r["tools"]]
        # Only real orderings: among runs with both tools, a comes first in most (not a rule that holds vacuously).
        if len(both) / n >= MIN_SUPPORT and \
                sum(r["tools"].index(a) < r["tools"].index(b) for r in both) / len(both) >= MIN_HOLD:
            candidates.append({"name": f"{a}_before_{b}", "before": [a, b]})
    gates, monitors = [], []
    for rule in candidates:
        ev = _evidence(rule, runs, conf)
        if ev["held"] / n < MIN_HOLD:
            continue
        use, why = _classify(ev)
        (gates if use == "gate" else monitors).append({"rule": rule, "use": use, "why": why, "evidence": ev})
    gates.sort(key=lambda s: s["evidence"]["graded"]["effect"]["estimate"])          # most harmful to break first
    # Rules that are sometimes broken say more than ones that never are; keep only a couple of the latter.
    varying = sorted((m for m in monitors if m["evidence"]["held"] < n), key=lambda s: -s["evidence"]["held"])
    invariants = [m for m in monitors if m["evidence"]["held"] == n][:MAX_INVARIANTS]
    return gates + (varying + invariants)[:MAX_MONITOR]


def no_answer_labels(runs: list[dict], min_share: float = 0.05) -> list[dict]:
    """A final text repeated word for word across many runs that didn't pass is a template (a step limit, a fallback),
    not an answer: a candidate for an explicit no-answer label. Structured (JSON) answers are never templates."""
    ends = [re.sub(r"\s+", " ", str(r.get("end") or "")).strip()[:300] for r in runs
            if r.get("passed") is not True and r.get("end")]
    ends = [e for e in ends if e and not e.startswith(("{", "["))]
    with_end = sum(1 for r in runs if r.get("end"))      # share among runs that record a final text at all
    out = []
    for text, c in Counter(ends).most_common(5):
        if c >= 3 and with_end and c / with_end >= min_share:
            out.append({"text": text, "runs": c, "share": c / with_end})
    return out


def propose(runs: list[dict], config_dir: str | Path | None = None, conf: float = 0.95) -> dict:
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_agent[r["agent"]].append(r)
    configs = {}
    if config_dir and Path(config_dir).is_dir():
        for p in Path(config_dir).glob("*.json"):
            c = json.loads(p.read_text())
            if "agent" in c:
                configs[c["agent"]] = {"path": str(p), "rules": c.get("rules", [])}
    items, n = [], 0
    for agent in sorted(by_agent):
        rs = by_agent[agent]
        existing = configs.get(agent, {}).get("rules", [])
        have = {json.dumps({k: v for k, v in r.items() if k in RULE_KINDS},
                           sort_keys=True) for r in existing}
        if len(rs) < MIN_RUNS:
            n += 1
            items.append({"id": n, "agent": agent, "kind": "advice",
                          "text": f"only {len(rs)} runs: too few to suggest rules (need {MIN_RUNS}); collect more "
                                  "traces (production, or replay its eval set)"})
            continue
        if existing:
            n += 1
            items.append({"id": n, "agent": agent, "kind": "info",
                          "text": f"already configured: {', '.join(r['name'] for r in existing)} (not re-proposed)"})
        for s in mine_rules(rs, conf):
            sig = json.dumps({k: v for k, v in s["rule"].items() if k != "name"}, sort_keys=True)
            if sig in have:
                continue
            n += 1
            items.append({"id": n, "agent": agent, "kind": "rule", "key": _key(agent), **s})
        for lab in no_answer_labels(rs):
            n += 1
            items.append({"id": n, "agent": agent, "kind": "advice",
                          "text": f"{lab['runs']} runs ({100 * lab['share']:.0f}%) ended with \"{lab['text']}\" and "
                                  "no pass: label these as an explicit no_answer reason so they're reported apart"})
        if not any(isinstance(r.get("passed"), bool) for r in rs):
            n += 1
            items.append({"id": n, "agent": agent, "kind": "advice",
                          "text": "no graded runs: rules can be monitored but not tied to outcomes; add an eval set "
                                  "(start with `agentstats promote find` on its failures) and grade it"})
        if agent not in configs:
            n += 1
            items.append({"id": n, "agent": agent, "kind": "advice",
                          "text": "no monitoring config yet: accepting a rule below creates one"})
    return {"agents": {a: len(rs) for a, rs in sorted(by_agent.items())}, "items": items}


def render(p: dict) -> str:
    lines = []
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for it in p["items"]:
        by_agent[it["agent"]].append(it)
    for agent, n in p["agents"].items():
        lines.append(f"{agent}  ({n} runs)")
        for it in by_agent.get(agent, []):
            if it["kind"] == "rule":
                ev, r = it["evidence"], it["rule"]
                spec = f"called {r['called']}" if "called" in r else f"{r['before'][0]} before {r['before'][1]}"
                lines.append(f" [{it['id']}] {it['use'].upper():7} rule {r['name']}: {spec}; held "
                             f"{100 * ev['held'] / ev['runs']:.1f}% of {ev['runs']} runs. {it['why']}")
            elif it["kind"] == "info":
                lines.append(f"      ({it['text']})")
            else:
                lines.append(f" [{it['id']}] ADVICE  {it['text']}")
        lines.append("")
    return "\n".join(lines)


def apply(p: dict, config_dir: str | Path, accept: list[int]) -> list[str]:
    """Write accepted rule items into <config_dir>/<key>.json. Returns the files written."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    chosen = [it for it in p["items"] if it["id"] in set(accept) and it["kind"] == "rule"]
    written = []
    for agent in sorted({it["agent"] for it in chosen}):
        mine = [it for it in chosen if it["agent"] == agent]
        path = next((Path(c) for c in config_dir.glob("*.json")
                     if json.loads(c.read_text()).get("agent") == agent), config_dir / f"{mine[0]['key']}.json")
        cfg = json.loads(path.read_text()) if path.exists() else {"agent": agent, "rules": []}
        names = {r["name"] for r in cfg.get("rules", [])}
        for it in mine:
            rule = dict(it["rule"])
            if rule["name"] in names:
                continue
            rule["use"] = it["use"]
            rule["why"] = f"suggested by agentstats: {it['why']}"
            cfg.setdefault("rules", []).append(rule)
        path.write_text(json.dumps(cfg, indent=1) + "\n")
        written.append(str(path))
    return written
