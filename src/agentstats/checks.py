"""
Deterministic trajectory checks: the rule-based raters that are cheap, fast and don't drift.

Rules (JSON list), each with a name:
    {"name": "looked_up_order", "called": "get_order_details"}
    {"name": "no_refund", "not_called": "issue_refund"}
    {"name": "lookup_first", "before": ["get_order_details", "issue_refund"]}   # if both happen, first one first
    {"name": "right_order", "args": {"tool": "get_order_details", "match": {"order_id": "42"}}}
"""
from __future__ import annotations

import json
from pathlib import Path

RULE_KINDS = ("called", "not_called", "before", "args")


def load_rules(path: str | Path) -> list[dict]:
    rules = json.loads(Path(path).read_text())
    for r in rules:
        kinds = [k for k in RULE_KINDS if k in r]
        if "name" not in r or len(kinds) != 1:
            raise ValueError(f"each rule needs a name and exactly one of {RULE_KINDS}: {r}")
    return rules


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
    raise ValueError(f"unknown rule: {rule}")


def run_checks(rules: list[dict], calls: list[dict]) -> dict[str, bool]:
    return {r["name"]: check(r, calls) for r in rules}
