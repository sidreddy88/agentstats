"""
Monitor agents from their production traces: one config file per agent, one report per run.

    agentstats monitor --config-dir agents/ --days 7 --out weekly/                       # Langfuse (API keys in env)
    agentstats monitor --config-dir agents/ --otlp traces.json --out weekly/             # an OpenTelemetry export
    agentstats monitor --config-dir agents/ --days 7 --out weekly/ --judge --judge-model gpt-5-mini

Each agent is a config file, <config-dir>/<name>.json:
  agent          the agent's span name (Langfuse observation name, or the OTel `gen_ai.agent.name` / span name)
  rules          behaviour rules, checked per run:
                   {"name", "called": tool} / {"name", "not_called": tool} / {"name", "before": [a, b]} /
                   {"name", "args": {"tool", "match"}}  (rules on the run's tool calls)
                   {"name", "output": field, "in": [...]} / {"name", "output": field, "min": n}  (the agent's output)
                 optional "when": an adapter condition name, "called:<tool>", "output:<field>>=<n>" or
                 "input_json:<key>" (a truthy key in the agent's input); the rule is only checked when it holds
  adapter        name of an adapter (from --adapters) for logic a config can't express (default "default")
  observation_type   Langfuse observation type of the agent span (default AGENT)
  rubric, judge_fields, judge_sample, review_n   the rubric judge and the human review sample (optional)
  review_instructions   what a reviewer should check for this agent, printed atop its review sheet (optional)
  group_by       tag prefix for candidate eval cases (optional)
`agentstats suggest-apply` writes this format, so accepted suggestions are monitored the next run.

Adapters: a Python file passed with --adapters that defines ADAPTERS = {"name": instance}, each a subclass of
agentstats.monitor.Adapter overriding what it needs: parse_input, parse_output, checks, conditions, tags, source,
no_answer_reason, judge_item.

Steps: fetch each agent's runs and every tool and LLM call in the window, joined by trace id -> one agentstats row
per run, tagged by source (the trace's environment, e.g. production) and model, errored runs as infrastructure
failures -> behaviour rules per source and model (Wilson intervals) -> optional rubric judge on a sample (flagged
unreliable above a flip-rate limit) and a blind human review sheet -> production failures as candidate eval cases ->
suggested new rules -> report.md with a section per agent: aggregates and trace ids only, no raw text (that stays in
the output folder for the reviewer).

A trace, as the engine and adapters see it:
  {"trace_id", "model", "cost_usd",
   "agent": {"name", "input", "output", "level" ("ERROR" if the run errored), "statusMessage", "latency" (s),
             "environment", "metadata", "projectId"},
   "calls": [{"name", "input", "output"}, ...]}        # tool calls, in start order
"""
from __future__ import annotations

import base64
import contextlib
import datetime as dt
import importlib.util
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from agentstats.checks import RULE_KINDS, run_checks, validate

RULE_META = ("when", "why", "use")         # rule keys the engine reads; the rest go to the trajectory checks


def as_json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


# ---- adapters ---------------------------------------------------------------------------------------------------

class Adapter:
    """Default behaviour: the agent's output is a JSON object; no extra checks or conditions."""
    conditions: dict = {}

    def parse_input(self, trace: dict):
        return as_json(trace["agent"].get("input"))

    def parse_output(self, trace: dict) -> dict | None:
        """The agent's answer; None means no usable answer."""
        out = as_json(trace["agent"].get("output"))
        return out if isinstance(out, dict) else None

    def checks(self, trace: dict) -> dict[str, bool]:
        """Checks that need code; include a rule only when it applies to this run."""
        return {}

    def tags(self, trace: dict) -> list[str]:
        inp = self.parse_input(trace)
        et = inp.get("error_type") if isinstance(inp, dict) else None
        return [f"error_type:{et}"] if et else []

    def source(self, trace: dict) -> str:
        """Where the run came from: the trace's environment (production, staging, eval, ...)."""
        env = (trace["agent"].get("environment") or "").lower()
        return env if env and env != "default" else "unknown"

    def no_answer_reason(self, trace: dict) -> str:
        """Why a run has no usable answer (only asked when parse_output returned None)."""
        return "parse_error"

    def judge_item(self, trace: dict) -> dict | None:
        """The fields the rubric judge sees; None skips the run."""
        out = self.parse_output(trace)
        if out is None:
            return None
        return {"agent_input": json.dumps(self.parse_input(trace))[:1500],
                "agent_steps": "\n".join(f"{c['name']}: {str(c['output'])[:200]}" for c in trace["calls"])[:2000]
                or "no recorded steps",
                "agent_output": json.dumps(out)[:1500]}


def load_adapters(path: str | Path | None) -> dict[str, Adapter]:
    adapters: dict[str, Adapter] = {"default": Adapter()}
    if path:
        spec = importlib.util.spec_from_file_location("agentstats_user_adapters", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        found = getattr(mod, "ADAPTERS", None)
        if not isinstance(found, dict):
            raise ValueError(f"{path}: define ADAPTERS = {{'name': AdapterSubclass()}}")
        adapters.update(found)
    return adapters


def load_configs(config_dir: str | Path, adapters: dict[str, Adapter], names: list[str] | None = None) -> list[dict]:
    out = []
    paths = sorted(Path(config_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"no agent configs (*.json) in {config_dir}")
    for path in paths:
        if names and path.stem not in names:
            continue
        cfg = json.loads(path.read_text())
        cfg.setdefault("key", path.stem)
        if "agent" not in cfg:
            raise ValueError(f"{path}: 'agent' (the agent's span name) is required")
        if cfg.get("adapter", "default") not in adapters:
            raise ValueError(f"{path}: unknown adapter {cfg['adapter']!r}; known: {sorted(adapters)}")
        for r in cfg.get("rules", []):
            kinds = [k for k in (*RULE_KINDS, "output") if k in r]
            if "name" not in r or len(kinds) != 1:
                raise ValueError(f"{path}: each rule needs a name and exactly one of {(*RULE_KINDS, 'output')}: {r}")
            try:
                validate(r)
            except ValueError as e:
                raise ValueError(f"{path}: {e}") from None
        out.append(cfg)
    return out


# ---- sources: Langfuse ------------------------------------------------------------------------------------------

def langfuse_get(params: dict, retries: int = 6) -> dict:
    """One page from the Langfuse v2 observations API. Rate limits (429) and server errors are retried with backoff,
    honouring Retry-After: paging through a week of generations hits the limit."""
    base = (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
            or "https://cloud.langfuse.com").rstrip("/")
    # Stray quotes or whitespace (copied from a .env line like KEY="pk-lf-...") make Langfuse answer 401.
    try:
        public, secret = (os.environ[k].strip().strip("\"'") for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"))
    except KeyError as e:
        raise ValueError(f"set {e.args[0]} (or pass --otlp to read an OpenTelemetry export instead)") from None
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    url = f"{base}/api/public/v2/observations?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": "Basic " + token, "User-Agent": "agentstats-monitor"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == retries:
                raise
            wait = e.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else min(2 ** attempt * 2, 60))
    raise RuntimeError("unreachable")


def fetch_observations(obs_type: str, start: dt.datetime, end: dt.datetime, name: str | None = None,
                       get=langfuse_get, max_rows: int = 50_000, fields: str = "core,basic,io,metadata") -> list[dict]:
    flt = [{"type": "datetime", "column": "startTime", "operator": ">=", "value": start.isoformat()},
           {"type": "datetime", "column": "startTime", "operator": "<", "value": end.isoformat()}]
    if name:
        flt.append({"type": "string", "column": "name", "operator": "=", "value": name})
    params = {"type": obs_type, "limit": 1000, "fields": fields, "filter": json.dumps(flt)}
    rows: list[dict] = []
    while len(rows) < max_rows:
        page = get(params)
        rows += page.get("data", [])
        cursor = (page.get("meta") or {}).get("cursor")
        if not cursor:
            break
        params = {**params, "cursor": cursor}
    return rows


def join_traces(agents: list[dict], tools: list[dict], generations: list[dict] | None = None) -> list[dict]:
    """One trace per agent run: the agent observation, its tool calls in start order, and from its LLM calls the
    most used model and the summed cost. Tool and generation rows don't carry the trace name, so they're matched by
    trace id."""
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for t in tools:
        by_trace[t["traceId"]].append(t)
    models: dict[str, Counter] = defaultdict(Counter)
    cost: dict[str, float] = defaultdict(float)
    for g in generations or []:
        if g.get("model"):
            models[g["traceId"]][g["model"]] += 1
        cost[g["traceId"]] += float(g.get("totalCost") or 0)
    out = []
    for a in agents:
        calls = sorted(by_trace.get(a["traceId"], []), key=lambda t: t.get("startTime") or "")
        m = models.get(a["traceId"])
        out.append({"trace_id": a["traceId"], "agent": a, "model": m.most_common(1)[0][0] if m else "unknown",
                    "cost_usd": cost.get(a["traceId"]),
                    "calls": [{"name": t.get("name"), "input": as_json(t.get("input")) or {},
                               "output": str(as_json(t.get("output")) or "")} for t in calls]})
    return out


class LangfuseSource:
    def __init__(self, start: dt.datetime, end: dt.datetime, get=langfuse_get):
        self.start, self.end, self.get = start, end, get
        self._tools = self._gens = None

    def traces(self, cfg: dict) -> list[dict]:
        if self._tools is None:
            self._tools = fetch_observations("TOOL", self.start, self.end, get=self.get)
            # LLM calls are named by whichever code path made them: take every generation, match by trace id.
            self._gens = fetch_observations("GENERATION", self.start, self.end, get=self.get,
                                            fields="core,model,usage", max_rows=200_000)
        agents = fetch_observations(cfg.get("observation_type", "AGENT"), self.start, self.end, name=cfg["agent"],
                                    get=self.get)
        return join_traces(agents, self._tools, self._gens)

    def previous_count(self, cfg: dict, adapter: Adapter) -> Counter | None:
        """Runs of this agent in the window of the same length just before this one, by source. Tool calls aren't
        fetched for it, so the adapter's source() sees each run's input and environment only."""
        prev_start = self.start - (self.end - self.start)
        obs = fetch_observations(cfg.get("observation_type", "AGENT"), prev_start, self.start, name=cfg["agent"],
                                 get=self.get, fields="core,basic,io")
        return Counter(adapter.source({"trace_id": o.get("traceId"), "agent": o, "calls": []}) for o in obs)

    def agent_names(self, observation_type: str = "AGENT") -> Counter:
        """Agent span name -> runs in the window (for `agentstats init`)."""
        # The name is in the "basic" field group, not "core".
        return Counter(o.get("name") for o in fetch_observations(observation_type, self.start, self.end,
                                                                 get=self.get, fields="core,basic") if o.get("name"))

    def link(self, traces: list[dict]) -> str:
        """Langfuse UI link template for the review sheet ({case} is the trace id)."""
        base = (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
                or "https://cloud.langfuse.com").rstrip("/")
        project = next((t["agent"].get("projectId") for t in traces if t["agent"].get("projectId")), None)
        return f"{base}/project/{project}/traces/{{case}}" if project else ""


# ---- sources: OpenTelemetry (OTLP JSON, GenAI semantic conventions) -------------------------------------------

def _attr_value(v: dict):
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return int(v[k]) if k == "intValue" else v[k]
    if "arrayValue" in v:
        return [_attr_value(x) for x in v["arrayValue"].get("values", [])]
    return None


def _attrs(items: list[dict]) -> dict:
    return {a["key"]: _attr_value(a.get("value", {})) for a in items or []}


NOT_AGENT_OPS = ("execute_tool", "chat", "text_completion", "generate_content", "embeddings")


def _agent_name(span: dict) -> str | None:
    """The agent a span is the run of, or None: an `invoke_agent` span, or one naming `gen_ai.agent.name` that isn't
    a tool or model call (SDKs copy the agent name onto those too)."""
    op = span["_a"].get("gen_ai.operation.name")
    if op in NOT_AGENT_OPS:
        return None
    if op == "invoke_agent" or span["_a"].get("gen_ai.agent.name"):
        return span["_a"].get("gen_ai.agent.name") or span.get("name")
    return None


def _iso(nanos) -> str:
    return dt.datetime.fromtimestamp(int(nanos) / 1e9, dt.UTC).isoformat()


class OtlpSource:
    """OTLP JSON exports (a file, or JSONL of export requests). The agent run is the `invoke_agent` span (or one
    carrying `gen_ai.agent.name`, not a tool or model call) whose agent name, or failing that span name, is the
    config's agent; its tool calls are the `execute_tool` spans in the same
    trace. Input and output come from `gen_ai.input.messages` / `gen_ai.output.messages` (or `input.value` /
    `output.value`), the environment from the `deployment.environment.name` resource attribute."""

    def __init__(self, paths: list[str | Path], start: dt.datetime | None = None, end: dt.datetime | None = None):
        self.spans: dict[str, list[dict]] = defaultdict(list)
        self._before: dict[str, list[dict]] = defaultdict(list)     # the window just before, for previous_count
        self._windowed = bool(start and end)
        prev_start = start - (end - start) if start and end else None
        for p in paths:
            text = Path(p).read_text()
            try:
                docs = [json.loads(text)]
            except json.JSONDecodeError:
                docs = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            for doc in docs:
                for rs in doc.get("resourceSpans", []):
                    res = _attrs((rs.get("resource") or {}).get("attributes"))
                    for ss in rs.get("scopeSpans", []):
                        for s in ss.get("spans", []):
                            t0 = int(s.get("startTimeUnixNano", 0))
                            if start and t0 and _iso(t0) < start.isoformat():
                                if prev_start and _iso(t0) >= prev_start.isoformat():
                                    self._before[s["traceId"]].append({**s, "_a": _attrs(s.get("attributes")),
                                                                       "_r": res})
                                continue
                            if end and t0 and _iso(t0) >= end.isoformat():
                                continue
                            self.spans[s["traceId"]].append({**s, "_a": _attrs(s.get("attributes")), "_r": res})

    def traces(self, cfg: dict) -> list[dict]:
        out = []
        for tid, spans in self.spans.items():
            spans = sorted(spans, key=lambda s: int(s.get("startTimeUnixNano", 0)))
            agent = next((s for s in spans if _agent_name(s) == cfg["agent"]), None)
            if agent is None:
                continue
            a, r = agent["_a"], agent["_r"]
            models = Counter(s["_a"].get("gen_ai.response.model") or s["_a"].get("gen_ai.request.model")
                             for s in spans)
            models.pop(None, None)
            cost = sum(float(s["_a"].get("gen_ai.usage.cost") or 0) for s in spans)
            status = (agent.get("status") or {}).get("code")
            calls = [{"name": s["_a"].get("gen_ai.tool.name") or s.get("name", ""),
                      "input": as_json(s["_a"].get("gen_ai.tool.call.arguments")) or {},
                      "output": str(s["_a"].get("gen_ai.tool.call.result") or "")}
                     for s in spans if s["_a"].get("gen_ai.operation.name") == "execute_tool"]
            out.append({"trace_id": tid, "model": models.most_common(1)[0][0] if models else "unknown",
                        "cost_usd": cost or None, "calls": calls,
                        "agent": {"name": cfg["agent"],
                                  "input": a.get("gen_ai.input.messages") or a.get("input.value"),
                                  "output": a.get("gen_ai.output.messages") or a.get("output.value"),
                                  "level": "ERROR" if status in (2, "STATUS_CODE_ERROR") else None,
                                  "statusMessage": (agent.get("status") or {}).get("message"),
                                  "latency": (int(agent.get("endTimeUnixNano", 0))
                                              - int(agent.get("startTimeUnixNano", 0))) / 1e9,
                                  "environment": (r.get("deployment.environment.name")
                                                  or r.get("deployment.environment")),
                                  "metadata": a}})
        return out

    def agent_names(self, observation_type: str = "AGENT") -> Counter:
        """Agent name -> runs (for `agentstats init`): spans naming a `gen_ai.agent.name` or of operation
        `invoke_agent`, one per trace."""
        out: Counter = Counter()
        for spans in self.spans.values():
            out.update({n for n in (_agent_name(s) for s in spans) if n})
        return out

    def previous_count(self, cfg: dict, adapter: Adapter) -> Counter | None:
        """Runs of this agent in the window just before, by source (None when there's no window: --all-spans)."""
        if not self._windowed:
            return None
        out: Counter = Counter()
        for spans in self._before.values():
            agent = next((s for s in spans if _agent_name(s) == cfg["agent"]), None)
            if agent is not None:
                env = agent.get("_r", {}).get("deployment.environment.name") or agent.get("_r", {}).get(
                    "deployment.environment")
                out[adapter.source({"agent": {"environment": env, "input": None}, "calls": []})] += 1
        return out

    def link(self, traces: list[dict]) -> str:
        return ""


# ---- rows -------------------------------------------------------------------------------------------------------

def condition_holds(when: str | None, trace: dict, output: dict | None, adapter: Adapter) -> bool:
    if not when:
        return True
    names = [c["name"] for c in trace["calls"]]
    if when.startswith("called:"):
        return when.split(":", 1)[1] in names
    if when.startswith("output:"):
        m = re.match(r"output:(\w+)\s*>=\s*([\d.]+)$", when)
        if not m:
            raise ValueError(f"can't read condition {when!r}")
        v = (output or {}).get(m.group(1))
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= float(m.group(2))
    if when.startswith("input_json:"):
        inp = adapter.parse_input(trace)
        return bool(isinstance(inp, dict) and inp.get(when.split(":", 1)[1]))
    fn = adapter.conditions.get(when)
    if fn is None:
        raise ValueError(f"unknown condition {when!r}")
    return bool(fn(trace))


def output_check(rule: dict, output: dict | None) -> bool:
    v = (output or {}).get(rule["output"])
    if "in" in rule:
        return v in rule["in"]
    if "min" in rule:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= rule["min"]
    return v is not None


def to_row(trace: dict, cfg: dict, adapter: Adapter) -> dict:
    agent = trace["agent"]
    base_tags = [f"agent:{cfg['key']}", f"source:{adapter.source(trace)}", f"model:{trace.get('model', 'unknown')}"]
    if agent.get("level") == "ERROR":           # the run itself failed: infrastructure, not the agent's judgement
        row = {"arm": cfg["key"], "case": trace["trace_id"], "run": 1, "passed": None,
               "infra_error": str(agent.get("statusMessage") or "agent run errored")[:200],
               "tags": base_tags, "checks": {}}
        if trace.get("cost_usd"):
            row["cost_usd"] = round(trace["cost_usd"], 6)      # an errored run still cost money
        return row
    output = adapter.parse_output(trace)
    calls = [{"name": c["name"], "args": c["input"] if isinstance(c["input"], dict) else {}} for c in trace["calls"]]
    checks: dict[str, bool] = {}
    for r in cfg.get("rules", []):
        if not condition_holds(r.get("when"), trace, output, adapter):
            continue
        if "output" in r:
            if output is not None:
                checks[r["name"]] = output_check(r, output)
        else:
            checks.update(run_checks([{k: v for k, v in r.items() if k not in RULE_META}], calls))
    checks.update(adapter.checks(trace))
    meta = agent.get("metadata") or {}
    row = {"arm": cfg["key"], "case": trace["trace_id"], "run": 1, "passed": None, "infra_error": None,
           "tags": base_tags + adapter.tags(trace), "checks": checks, "tool_calls": len(calls)}
    if trace.get("cost_usd"):
        row["cost_usd"] = round(trace["cost_usd"], 6)
    if agent.get("latency") is not None:
        row["latency_s"] = float(agent["latency"])
    if isinstance(meta, dict) and meta.get("iterations") is not None:
        row["steps"] = float(meta["iterations"])
    if output is None:
        row["no_answer"] = adapter.no_answer_reason(trace)
    return row


def trajectories(cfg: dict, traces: list[dict]) -> list[dict]:
    """One line per run for `agentstats suggest`: the tools in order and the final text (ungraded)."""
    out = []
    for t in traces:
        if t["agent"].get("level") == "ERROR":
            continue
        end = as_json(t["agent"].get("output"))
        end = end.get("answer", "") if isinstance(end, dict) and "answer" in end else end
        out.append({"agent": cfg["agent"], "case": t["trace_id"], "passed": None,
                    "tools": [c["name"] for c in t["calls"]], "end": str(end or "")[:300]})
    return out


# ---- summaries --------------------------------------------------------------------------------------------------

def _pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _tag(row: dict, prefix: str) -> str:
    return next((t.split(":", 1)[1] for t in row["tags"] if t.startswith(prefix + ":")), "unknown")


def summarize_rules(rows: list[dict]) -> dict[str, dict]:
    from agentstats import analysis
    from agentstats.records import from_dict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[f"{_tag(r, 'source')} / {_tag(r, 'model')}"].append(r)
    out = {}
    for key in sorted(groups, key=lambda k: (not k.startswith("production"), k)):
        atts = [from_dict(r) for r in groups[key]]
        res = analysis.summarize(atts, design="sampled")
        costs = [r["cost_usd"] for r in groups[key] if r.get("cost_usd")]
        out[key] = {"runs": len(atts), "no_answer": sum(1 for a in atts if a.no_answer),
                    "infra": sum(1 for a in atts if a.infra_error),
                    "cost_per_run": sum(costs) / len(costs) if costs else None,
                    "cost_per_useful": _cost_per_useful(groups[key]), "checks": res.get("checks", {})}
    return out


def _cost_per_useful(rows: list[dict]) -> float | None:
    """All spend divided by the runs that produced something usable: passed ones when any run is graded, otherwise
    runs with an answer and no infrastructure error. Runs that fail still cost money; this charges it to the ones
    that didn't."""
    spend = sum(r.get("cost_usd") or 0 for r in rows)
    if not spend:
        return None
    graded = [r for r in rows if isinstance(r.get("passed"), bool)]
    useful = (sum(1 for r in graded if r["passed"]) if graded
              else sum(1 for r in rows if not r.get("no_answer") and not r.get("infra_error")))
    return spend / useful if useful else float("inf")


def flags(rule_summary: dict[str, dict], floor: float) -> list[str]:
    """A rule whose whole interval is below the floor is proven to hold too rarely; one whose estimate is below it is
    a lead to watch."""
    out = []
    for src, s in rule_summary.items():
        for rule, iv in s["checks"].items():
            if iv["high"] < floor:
                out.append(f"**{src}: `{rule}` holds only {_pct(iv['estimate'])} (CI {_pct(iv['low'])} to "
                           f"{_pct(iv['high'])}), proven below {_pct(floor)}**")
            elif iv["estimate"] < floor:
                out.append(f"{src}: `{rule}` at {_pct(iv['estimate'])} (CI {_pct(iv['low'])} to {_pct(iv['high'])}):"
                           " below the floor but not proven; watch it")
    return out


VOLUME_MIN_PREVIOUS = 10      # below this many runs last window, a drop can't be told from chance
VOLUME_DROP = 0.5             # flag when this window has under half the previous window's runs ...
VOLUME_P = 0.01               # ... and an exact test says the rate really fell


def volume_check(current: int, previous: int | None) -> dict:
    """Did an agent's run count fall? Under an unchanged rate, given both windows' total, this window's count is
    Binomial(total, 1/2); a one-sided exact test (the conditional test for two Poisson counts) says whether the drop
    is real. An agent that ran before and not at all now is always flagged: tracing may have stopped."""
    out = {"current": current, "previous": previous, "flag": False, "p": None, "note": ""}
    if previous is None:
        out["note"] = "no previous window"
        return out
    if previous and current == 0:
        out.update(flag=True, note="no runs this window after runs in the last one: check that tracing still works")
        return out
    if previous >= VOLUME_MIN_PREVIOUS and current < VOLUME_DROP * previous:
        from scipy.stats import binomtest
        p = binomtest(current, current + previous, 0.5, alternative="less").pvalue
        out["p"] = p
        if p < VOLUME_P:
            out.update(flag=True, note=f"runs fell {100 * (1 - current / previous):.0f}% (exact test p = {p:.2g}): if "
                                       "traffic didn't fall, check that tracing still works")
    return out


def _cli(*args: str) -> str:
    """Run another agentstats command in-process and return what it printed."""
    from agentstats import cli
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(list(args))
    if code not in (0, 1, 2, 3):
        raise RuntimeError(f"agentstats {' '.join(args[:2])} failed ({code}): {buf.getvalue()[-500:]}")
    return buf.getvalue()


# ---- judge and human review -------------------------------------------------------------------------------------

class JudgeSettings:
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", key_env: str = "OPENAI_API_KEY",
                 max_flips: float = 0.10, repeats: int = 2):
        self.model, self.base_url, self.key_env, self.max_flips, self.repeats = (model, base_url, key_env, max_flips,
                                                                                 repeats)


def write_review_helper(out: Path, items: dict[str, dict], link: str, fields: list[str],
                        instructions: str | None = None) -> None:
    """REVIEW.md next to the blind sheet: each sampled run's material (what the judge saw, never its verdict) with a
    trace link, numbered in the sheet's order. Raw text: it stays in the output folder, never in the report."""
    import csv
    sheet = out / "review" / "review.csv"
    if not sheet.exists():
        return
    rows = list(csv.DictReader(sheet.open()))
    md = ["# Review sheet helper", "", "For each run, decide whether the agent got it right. Write pass, fail or "
          "unsure in the human_verdict column of review.csv (same order). The judge's verdict is deliberately not "
          "shown.", ""]
    if instructions:
        md += ["**What to check:** " + instructions.strip(), ""]
    for n, r in enumerate(rows, 1):
        it = items.get(r["case"], {})
        url = link.format(case=r["case"]) if link else ""
        md += [f"## {n}. `{r['case'][:12]}`" + (f"  [open trace]({url})" if url else ""), ""]
        for f in fields:
            md += [f"**{f.replace('_', ' ')}**", "", "```", str(it.get(f, "")).strip()[:1500], "```", ""]
        md += ["---", ""]
    (out / "review" / "REVIEW.md").write_text("\n".join(md))


def judge(cfg: dict, traces: list[dict], adapter: Adapter, out: Path, settings: JudgeSettings, seed: int,
          link: str = "") -> tuple[str, str]:
    """(judge summary, review note) for one agent: rubric judge on a sample, then a blind review sheet."""
    items = []
    for t in traces:
        it = adapter.judge_item(t)
        if it:
            items.append({"item": t["trace_id"], "case": t["trace_id"], "tags": [f"source:{adapter.source(t)}"], **it})
    if not items:
        return "No judgeable runs.", "Not run (no judged runs)."
    rng = random.Random(seed)
    prod = [i for i in items if "source:production" in i["tags"]]
    rest = [i for i in items if "source:production" not in i["tags"]]
    rng.shuffle(rest)
    sample = (prod + rest)[:cfg.get("judge_sample", 40)]
    (out / "judge_items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in sample))
    fields = cfg.get("judge_fields") or [k for k in sample[0] if k not in ("item", "case", "tags")]
    _cli("grade", str(out / "judge_items.jsonl"), "--rubric", str(cfg["rubric"]), "--model", settings.model,
         "--base-url", settings.base_url, "--api-key-env", settings.key_env, "--fields", ",".join(fields),
         "--repeats", str(settings.repeats), "--out", str(out / "judge_labels.jsonl"),
         "--rows", str(out / "judge_rows.jsonl"))
    rows_path = out / "judge_rows.jsonl"
    jrows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()] if rows_path.exists() else []
    judged = [r for r in jrows if r["passed"] is not None]
    if not judged:
        return f"**The judge produced no verdicts** (check {settings.key_env} and the model).", "Not run (no verdicts)."
    per_q = Counter(k for r in judged for k, v in r["checks"].items() if not v)
    flip_rate = None
    if settings.repeats > 1:
        flip = re.search(r"flip rate ([\d.]+)%", _cli("judge", str(out / "judge_labels.jsonl")))
        flip_rate = float(flip.group(1)) / 100 if flip else None
    summary = (f"{len(judged)} of {len(jrows)} sampled runs judged; {sum(r['passed'] for r in judged)} passed every "
               "rubric question. Failed questions: "
               + (", ".join(f"`{k}` {v}" for k, v in per_q.most_common()) or "none")
               + (f". Judge flip rate over {settings.repeats} repeats: {_pct(flip_rate)}." if flip_rate is not None
                  else "."))
    if flip_rate is not None and flip_rate > settings.max_flips:
        summary = (f"**The judge is unreliable** (flip rate {_pct(flip_rate)} over {settings.repeats} repeats, above "
                   f"{_pct(settings.max_flips)}): treat its verdicts as noise and fix the rubric before trusting them. "
                   + summary)
    n = cfg.get("review_n", 20)
    _cli("review", "sample", str(rows_path), "--n", str(n), "--stratify-by", "source", "--seed", str(seed),
         "--out", str(out / "review"), *(["--link", link] if link else []))
    write_review_helper(out, {i["case"]: i for i in sample}, link, fields, cfg.get("review_instructions"))
    return summary, (f"{n} judged runs sampled for review: read `{cfg['key']}/review/REVIEW.md` (each run's material "
                     f"and a trace link) and fill `{cfg['key']}/review/review.csv`; the judge's verdicts and sampling "
                     f"weights are in `{cfg['key']}/review/key.jsonl`.")


# ---- report -----------------------------------------------------------------------------------------------------

def render_agent(cfg: dict, rows: list[dict], rule_summary: dict, judge_summary: str, review_note: str,
                 promote_table: str, failures: list[str], floor: float) -> list[str]:
    by_src = Counter(_tag(r, "source") for r in rows)
    lines = [f"## {cfg['agent']}", ""]
    if cfg.get("description"):
        lines += [cfg["description"], ""]
    if not rows:
        return lines + ["No runs in this window.", ""]
    lines += [f"{len(rows)} runs: " + ", ".join(f"{k} {v}" for k, v in sorted(by_src.items())) + "."]
    fl = flags(rule_summary, floor)
    lines += ["", "**Needs attention**", ""] + ([f"- {f}" for f in fl] or
                                              [f"- Nothing: every behaviour rule is at or above {_pct(floor)}."])
    lines += ["", "| source / model | runs | errored (infrastructure) | no usable answer | cost per run | "
              "cost per usable answer |", "|---|---|---|---|---|---|"]
    for key, s in rule_summary.items():
        cpr = f"${s['cost_per_run']:.4f}" if s.get("cost_per_run") else "n/a"
        cpu = s.get("cost_per_useful")
        cpu = "n/a" if cpu is None else "no usable answers" if cpu == float("inf") else f"${cpu:.4f}"
        lines.append(f"| {key} | {s['runs']} | {s['infra']} | {s['no_answer']} | {cpr} | {cpu} |")
    errors = Counter(r["infra_error"] for r in rows if r.get("infra_error"))
    if errors:
        lines += ["", "Errored runs are kept out of the rule rates. Causes: "
                  + "; ".join(f"{msg} ({n})" for msg, n in errors.most_common(3)) + "."]
    checked = [(k, rule, iv) for k, s in rule_summary.items() for rule, iv in sorted(s["checks"].items())]
    if checked:
        lines += ["", "| source / model | rule | runs checked | held | 95% CI |", "|---|---|---|---|---|"]
        lines += [f"| {k} | `{rule}` | {iv['n']} | {_pct(iv['estimate'])} | {_pct(iv['low'])} to {_pct(iv['high'])} |"
                  for k, rule, iv in checked]
    else:
        lines += ["", "No behaviour rule applied to these runs."]
    if failures:
        lines += ["", "Runs that broke a rule (trace ids): " + ", ".join(f"`{t}`" for t in failures[:15])
                  + (f" and {len(failures) - 15} more" if len(failures) > 15 else "")]
    if cfg.get("rubric"):
        lines += ["", "**Rubric judge**", "", judge_summary or "Not run (pass `--judge`).",
                  "", "**Human review**", "", review_note]
    lines += ["", "**Candidate eval cases (production failures, deduplicated)**", "", promote_table or "None.", ""]
    return lines


def suggestions(out: Path, config_dir: str | Path) -> str:
    traj = out / "trajectories.jsonl"
    if not traj.exists() or not traj.read_text().strip():
        return "No trajectories in this window."
    text = _cli("suggest", str(traj), "--config-dir", str(config_dir), "--out", str(out / "proposal.json"))
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("wrote "))
    return ("agentstats looked at these runs against the configured rules and suggests (evidence on each line; "
            "production runs are ungraded, so new rules here are monitor-only):\n\n```\n" + body.strip() + "\n```\n\n"
            f"To accept some: `agentstats suggest-apply proposal.json --config-dir {config_dir}` (it asks item by "
            "item and writes only the rules you accept).")


def render_volume(volume: list[tuple[str, dict]]) -> list[str]:
    """Run counts per agent vs the window before, first in the report: a quiet report is only good news if the
    traces are still arriving."""
    flagged = [(a, v) for a, v in volume if v["flag"]]
    lines = ["## Run volume", ""]
    lines += [f"- **{a}: {v['note']}** ({v['current']} vs {v['previous']})" for a, v in flagged] or \
        ["- No agent's run count fell significantly against the window before (eval runs aren't checked)."]
    lines += ["", "| agent / source | runs this window | window before | |", "|---|---|---|---|"]
    for a, v in volume:
        prev = "n/a" if v["previous"] is None else v["previous"]
        lines.append(f"| {a} | {v['current']} | {prev} | {'⚠️' if v['flag'] else ''} |")
    return lines + [""]


def render(start, end, sections: list[list[str]], config_dir: str | Path, labels_dir: str) -> str:
    lines = [f"# agentstats monitor, {start:%Y-%m-%d} to {end:%Y-%m-%d}", ""]
    for s in sections:
        lines += s
    lines += ["## How to close the loop", "",
              "1. Open the output folder (one subfolder per agent).",
              "2. Fill `<agent>/review/review.csv` (human_verdict: pass / fail / unsure), then `agentstats review "
              "import <agent>/review/review.csv --key <agent>/review/key.jsonl --rater <you> --out "
              f"{labels_dir}/<agent>_review_labels.jsonl`; it prints how often the judge agrees with you.",
              "3. In `<agent>/candidates.jsonl`, set `\"approved\": true` on cases worth keeping, then `agentstats "
              f"promote accept <agent>/candidates.jsonl --into {labels_dir}/<agent>_cases.jsonl`.", "",
              f"_Generated by `agentstats monitor`; agents are configured in `{config_dir}`. No raw run text is "
              "included here; it's in the output folder and your trace store._"]
    return "\n".join(lines) + "\n"


# ---- run --------------------------------------------------------------------------------------------------------

def run_agent(cfg: dict, traces: list[dict], adapter: Adapter, out: Path, *, judge_settings: JudgeSettings | None,
              seed: int, floor: float, link: str = "") -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    rows = [to_row(t, cfg, adapter) for t in traces]
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if not rows:
        return render_agent(cfg, rows, {}, "", "", "", [], floor)
    rule_summary = summarize_rules(rows)
    (out / "rules.json").write_text(json.dumps(rule_summary, indent=1))
    failures = [r["case"] for r in rows if (r["checks"] and not all(r["checks"].values())) or r.get("no_answer")]
    judge_summary, review_note = "", "Not run (needs the judge: `--judge`)."
    if judge_settings and cfg.get("rubric"):
        judge_summary, review_note = judge(cfg, traces, adapter, out, judge_settings, seed, link)
    prod_rows = [r for r in rows if "source:production" in r["tags"]]
    table = ""
    if prod_rows:
        (out / "production_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prod_rows))
        args = ["promote", "find", str(out / "production_rows.jsonl"), "--source", "production", "--label-from",
                "human", "--out", str(out / "candidates.jsonl")]
        if cfg.get("group_by"):
            args += ["--group-by", cfg["group_by"]]
        table = "\n".join(ln for ln in _cli(*args).splitlines() if ln.startswith("|"))
    return render_agent(cfg, rows, rule_summary, judge_summary, review_note, table, failures, floor)


def run(config_dir: str | Path, out: str | Path, *, source, start: dt.datetime, end: dt.datetime,
        adapters: dict[str, Adapter] | None = None, agents: list[str] | None = None,
        judge_settings: JudgeSettings | None = None, seed: int = 0, floor: float = 0.95,
        labels_dir: str = "evals", link: str | None = None, log=print) -> Path:
    """Monitor every configured agent over one window; returns the report path."""
    adapters = adapters or {"default": Adapter()}
    configs = load_configs(config_dir, adapters, agents)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sections, traj, volume = [], [], []
    for cfg in configs:
        traces = source.traces(cfg)
        log(f"{cfg['agent']}: {len(traces)} runs")
        adapter = adapters[cfg.get("adapter", "default")]
        prev = source.previous_count(cfg, adapter) if hasattr(source, "previous_count") else None
        now = Counter(adapter.source(t) for t in traces)
        for src in sorted(set(now) | set(prev or {}), key=lambda k: (k != "production", k)) or ["all"]:
            v = volume_check(now.get(src, 0), None if prev is None else prev.get(src, 0))
            if src == "eval":                          # eval runs come and go by design: shown, never flagged
                v.update(flag=False, note="eval runs: not checked")
            volume.append((f"{cfg['agent']} / {src}", v))
        trace_link = link if link is not None else source.link(traces)
        sections.append(run_agent(cfg, traces, adapters[cfg.get("adapter", "default")], out / cfg["key"],
                                  judge_settings=judge_settings, seed=seed, floor=floor, link=trace_link))
        traj += trajectories(cfg, traces)
    (out / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in traj))
    sections.append(["## Suggestions", "", suggestions(out, config_dir), ""])
    sections.insert(0, render_volume(volume))
    (out / "volume.json").write_text(json.dumps({a: v for a, v in volume}, indent=1, default=str))
    report = out / "report.md"
    report.write_text(render(start, end, sections, config_dir, labels_dir))
    return report
