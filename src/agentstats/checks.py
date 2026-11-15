"""
Deterministic trajectory checks: the rule-based raters that are cheap, fast and don't drift.

Rules (JSON list), each with a name:
    {"name": "looked_up_order", "called": "get_order_details"}
    {"name": "no_refund", "not_called": "issue_refund"}
    {"name": "lookup_first", "before": ["get_order_details", "issue_refund"]}   # if both happen, first one first
    {"name": "right_order", "args": {"tool": "get_order_details", "match": {"order_id": "42"}}}
    {"name": "few_searches", "max_calls": {"tool": "search", "max": 6}}         # "*" counts every tool call
    {"name": "no_flailing", "max_repeats": {"tool": "*", "max": 0}}             # calls identical to an earlier one
    {"name": "writes_verified", "verified": {"tool": ["suspend_user", "create_user"]}}   # or "*"

max_calls catches over-retrieval (more calls than a task needs); max_repeats catches tool flailing (the same call
again with the same arguments, after case and whitespace are normalised).

verified checks that each call to the named tools was read back after it wrote, by whatever executed it (a
gateway or the tool itself), not by the agent: the call's result must carry "verified": true. The read-back
happens where and when the write happens; agentstats only checks what was recorded. "verified": false (read back,
not there) or no "verified" at all (never checked) fails. "verified": null means the read-back itself couldn't run
(a timeout): that call is skipped, since it isn't evidence either way; record the run's infra_error if that matters.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

RULE_KINDS = ("called", "not_called", "before", "args", "max_calls", "max_repeats", "verified")


def load_rules(path: str | Path) -> list[dict]:
    rules = json.loads(Path(path).read_text())
    for r in rules:
        kinds = [k for k in RULE_KINDS if k in r]
        if "name" not in r or len(kinds) != 1:
            raise ValueError(f"each rule needs a name and exactly one of {RULE_KINDS}: {r}")
        validate(r)
    return rules


def validate(rule: dict) -> None:
    for kind in ("max_calls", "max_repeats"):
        if kind in rule:
            spec = rule[kind]
            if not (isinstance(spec, dict) and isinstance(spec.get("tool"), str)
                    and isinstance(spec.get("max"), int) and not isinstance(spec.get("max"), bool)
                    and spec["max"] >= 0):
                raise ValueError(f'{kind} needs {{"tool": name or "*", "max": a whole number >= 0}}: {rule}')
    if "verified" in rule:
        tool = rule["verified"].get("tool") if isinstance(rule["verified"], dict) else None
        if not (isinstance(tool, str) or (isinstance(tool, list) and tool and all(isinstance(t, str) for t in tool))):
            raise ValueError(f'verified needs {{"tool": name, [names] or "*"}}: {rule}')


def _normalise(v):
    """Argument values compared case- and whitespace-insensitively (trimmed, inner runs collapsed)."""
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip().lower()
    if isinstance(v, dict):
        return {k: _normalise(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_normalise(x) for x in v]
    return v


def _same_call_key(call: dict) -> tuple[str, str]:
    return call["name"], json.dumps(_normalise(call.get("args") or {}), sort_keys=True, default=str)


_NOT_RECORDED = object()


def _verified_flag(result):
    """The "verified" field of a call's result (a dict, or a JSON string of one); _NOT_RECORDED if there isn't one."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return _NOT_RECORDED
    return result.get("verified", _NOT_RECORDED) if isinstance(result, dict) else _NOT_RECORDED


def check(rule: dict, calls: list[dict]) -> bool:
    names = [c["name"] for c in calls]
    if "called" in rule:
        return rule["called"] in names
    if "not_called" in rule:
        return rule["not_called"] not in names
    if "before" in rule:
        first, second = rule["before"]
        if second not in names:
            return True
        return first in names and names.index(first) < names.index(second)
    if "args" in rule:
        tool, match = rule["args"]["tool"], rule["args"]["match"]
        return any(c["name"] == tool and all(str(c["args"].get(k)) == str(v) for k, v in match.items())
                   for c in calls)
    if "max_calls" in rule:
        tool, cap = rule["max_calls"]["tool"], rule["max_calls"]["max"]
        return sum(1 for n in names if tool in ("*", n)) <= cap
    if "max_repeats" in rule:
        tool, cap = rule["max_repeats"]["tool"], rule["max_repeats"]["max"]
        seen: Counter = Counter()
        repeats = 0
        for c in calls:
            if tool in ("*", c["name"]):
                key = _same_call_key(c)
                repeats += seen[key] > 0
                seen[key] += 1
        return repeats <= cap
    if "verified" in rule:
        tool = rule["verified"]["tool"]
        tools = [tool] if isinstance(tool, str) else tool
        for c in calls:
            if "*" in tools or c["name"] in tools:
                flag = _verified_flag(c.get("result"))
                if flag is None:  # the read-back couldn't run: not evidence either way
                    continue
                if flag is not True:
                    return False
        return True
    raise ValueError(f"unknown rule: {rule}")


def run_checks(rules: list[dict], calls: list[dict]) -> dict[str, bool]:
    return {r["name"]: check(r, calls) for r in rules}
