"""
Agent traces as attempt rows: OpenTelemetry (OTLP JSON, GenAI semantic conventions) and Langfuse exports. One trace
is one attempt. Tool calls become trajectory rule checks; tokens, latency and cost become metrics.

    agentstats convert-traces traces.json --format otlp --rules rules.json --out rows.jsonl
    agentstats convert-traces langfuse-export.json --format langfuse --verdicts verdicts.jsonl --rules rules.json \\
        --arm main --out rows.jsonl

Pass/fail doesn't come from traces. Give it with --verdicts (JSONL, one row per trace:
{"trace_id": ..., "passed": true, "case": "...", "run": 1, "infra_error": null}), or leave it out: the rows are then
**rules-only** (passed = null), which is what production traces usually are, and the rule gates (`min_rule_rate`,
`no_rule_regression`) still work on them. That's the point: behaviour checks need no labels.

OpenTelemetry: spans are grouped by traceId. A tool call is a span with `gen_ai.operation.name = "execute_tool"`
(name in `gen_ai.tool.name`, arguments in `gen_ai.tool.call.arguments` when recorded). Tokens are summed from
`gen_ai.usage.input_tokens` / `output_tokens`. The case id comes from a span attribute (--case-key, default
`agentstats.case`) when no verdict names it, else from the trace id.

Langfuse: a JSON list (or JSONL) of traces as returned by GET /api/public/traces/{id}: observations of type TOOL are
tool calls (name, input as arguments), GENERATION usage gives tokens, the trace's latency and totalCost give the rest.
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

from agentstats.checks import run_checks
from agentstats.records import Attempt, from_dict

FORMATS = ("otlp", "langfuse")


# ---- reading files --------------------------------------------------------------------------------------------

def _read_json_or_jsonl(path: Path) -> list:
    text = Path(path).read_text()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _args(value) -> dict:
    """Tool arguments as a dict, from a dict, a JSON string, or a Python-repr string (Langfuse stores some that way)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(value)
                return parsed if isinstance(parsed, dict) else {}
            except (ValueError, SyntaxError):
                continue
    return {}


# ---- OpenTelemetry ----------------------------------------------------------------------------------------------

def _attr_value(v: dict):
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return int(v[k]) if k == "intValue" else v[k]
    if "arrayValue" in v:
        return [_attr_value(x) for x in v["arrayValue"].get("values", [])]
    return None


def _attrs(span: dict) -> dict:
    return {a["key"]: _attr_value(a.get("value", {})) for a in span.get("attributes", [])}


def otlp_traces(path: str | Path, case_key: str = "agentstats.case") -> dict[str, dict]:
    """trace_id -> {"calls": [...], "metrics": {...}, "case": str | None} from an OTLP JSON export."""
    spans = []
    for doc in _read_json_or_jsonl(Path(path)):
        for rs in doc.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                spans += ss.get("spans", [])
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_trace[s["traceId"]].append(s)
    out = {}
    for tid, ss in by_trace.items():
        ss.sort(key=lambda s: int(s.get("startTimeUnixNano", 0)))
        calls, tokens_in, tokens_out, case = [], 0, 0, None
        for s in ss:
            a = _attrs(s)
            if a.get("gen_ai.operation.name") == "execute_tool":
                calls.append({"name": a.get("gen_ai.tool.name") or s.get("name", ""),
                              "args": _args(a.get("gen_ai.tool.call.arguments"))})
            tokens_in += int(a.get("gen_ai.usage.input_tokens") or 0)
            tokens_out += int(a.get("gen_ai.usage.output_tokens") or 0)
            case = case or a.get(case_key)
        start = min(int(s.get("startTimeUnixNano", 0)) for s in ss)
        end = max(int(s.get("endTimeUnixNano", 0)) for s in ss)
        metrics = {"tool_calls": float(len(calls)), "latency_s": (end - start) / 1e9}
        if tokens_in or tokens_out:
            metrics.update(input_tokens=float(tokens_in), output_tokens=float(tokens_out))
        out[tid] = {"calls": calls, "metrics": metrics, "case": case}
    return out


# ---- Langfuse ---------------------------------------------------------------------------------------------------

def langfuse_traces(path: str | Path) -> dict[str, dict]:
    """trace_id -> {"calls": [...], "metrics": {...}, "case": None, "model": ...} from Langfuse trace exports."""
    out = {}
    for t in _read_json_or_jsonl(Path(path)):
        obs = sorted(t.get("observations") or [], key=lambda o: o.get("startTime") or "")
        calls = [{"name": o.get("name", ""), "args": _args(o.get("input"))} for o in obs if o.get("type") == "TOOL"]
        gens = [o for o in obs if o.get("type") == "GENERATION"]
        tokens_in = sum(int((o.get("usage") or {}).get("input") or 0) for o in gens)
        tokens_out = sum(int((o.get("usage") or {}).get("output") or 0) for o in gens)
        metrics = {"tool_calls": float(len(calls))}
        if t.get("latency") is not None:
            metrics["latency_s"] = float(t["latency"])
        if t.get("totalCost"):
            metrics["cost_usd"] = float(t["totalCost"])
        if tokens_in or tokens_out:
            metrics.update(input_tokens=float(tokens_in), output_tokens=float(tokens_out))
        models = sorted({o.get("model") for o in gens if o.get("model")})
        out[t["id"]] = {"calls": calls, "metrics": metrics, "case": None, "model": models[0] if models else None}
    return out


# ---- rows -------------------------------------------------------------------------------------------------------

def to_attempts(traces: dict[str, dict], *, rules: list[dict] | None = None, verdicts: dict[str, dict] | None = None,
                arm: str = "default") -> list[Attempt]:
    """One attempt per trace. Without a verdict for a trace, passed is null (rules-only)."""
    verdicts = verdicts or {}
    out = []
    for tid, t in sorted(traces.items()):
        v = verdicts.get(tid, {})
        row = {"arm": arm, "case": v.get("case") or t.get("case") or tid, "run": v.get("run", 1),
               "passed": v.get("passed"), "infra_error": v.get("infra_error"), "no_answer": v.get("no_answer"),
               "tags": list(v.get("tags") or []) + ([f"model:{t['model']}"] if t.get("model") else []),
               "checks": run_checks(rules, t["calls"]) if rules else {}, **t["metrics"]}
        out.append(from_dict(row))
    return out


def load_verdicts(path: str | Path) -> dict[str, dict]:
    rows = _read_json_or_jsonl(Path(path))
    missing = [r for r in rows if "trace_id" not in r]
    if missing:
        raise ValueError(f"{path}: every verdict needs a trace_id (e.g. {missing[0]})")
    return {r["trace_id"]: r for r in rows}


def convert(paths: list[str | Path], fmt: str, *, rules: list[dict] | None = None, verdicts: dict | None = None,
            arm: str = "default", case_key: str = "agentstats.case") -> list[Attempt]:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    traces: dict[str, dict] = {}
    for p in paths:
        traces.update(otlp_traces(p, case_key) if fmt == "otlp" else langfuse_traces(p))
    return to_attempts(traces, rules=rules, verdicts=verdicts, arm=arm)
