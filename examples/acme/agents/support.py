"""
Acme's support agent: a small tool-calling loop over a fake order database.

    python agents/support.py --prompt v1 --model deepseek-ai/DeepSeek-V4.1-Flash --trials 2 --out data/support-v1
    python agents/support.py --prompt v2 ...     # the "simplified" prompt: drops the look-up-first rule

Writes, per run:
- rows.jsonl     agentstats attempt rows (pass/fail from the order database, tool-call checks from rules.json)
- traces.json    the same conversations as OpenTelemetry (OTLP JSON) spans, as a tracing backend would export them
- items.jsonl    the customer message and the agent's final reply, for `agentstats grade`

Any OpenAI-compatible endpoint (default Together; key in TOGETHER_API_KEY). Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentstats.checks import run_checks

HERE = Path(__file__).resolve().parent.parent
ORDERS = {f"A{100 + i}": {"total": t, "status": s} for i, (t, s) in enumerate([
    (20, "delivered"), (45, "delivered"), (12, "shipped"), (30, "delivered"), (18, "delivered"), (60, "processing"),
    (25, "delivered"), (9, "delivered"), (75, "shipped"), (40, "delivered")])}

PROMPTS = {
    "v1": ("You are Acme's customer support agent. Use the tools. Before issuing any refund, ALWAYS call lookup_order "
           "first and refund exactly the order total it returns. Only orders with status 'delivered' can be "
           "refunded; for others, explain that and do not refund. If the customer asks for a human, call "
           "escalate_to_human. Never offer anything beyond a refund. Keep replies short and polite."),
    "v2": ("You are Acme's customer support agent. Help customers quickly using the tools. If the customer asks for "
           "a human, call escalate_to_human. Keep replies short and polite."),
}
TOOLS = [
    {"type": "function", "function": {"name": "lookup_order", "description": "Look up an order's total and status.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Refund an order.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                    "required": ["order_id", "amount"]}}},
    {"type": "function", "function": {"name": "escalate_to_human", "description": "Hand the chat to a person.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]


def load_cases(name: str = "support_cases.jsonl") -> list[dict]:
    return [json.loads(x) for x in (HERE / "config" / name).read_text().splitlines() if x.strip()]


def chat(base_url: str, key: str, model: str, messages: list[dict], retries: int = 4,
         temperature: float | None = 0.7) -> dict:
    payload = {"model": model, "messages": messages, "tools": TOOLS, "max_tokens": 600}
    if temperature is not None:
        payload["temperature"] = temperature
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "acme-example/1"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if (e.code not in (408, 429) and e.code < 500) or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def run_tool(name: str, args: dict, state: dict) -> str:
    if name == "lookup_order":
        o = ORDERS.get(str(args.get("order_id", "")).upper())
        return json.dumps(o) if o else "no such order"
    if name == "issue_refund":
        state["refunds"].append((str(args.get("order_id", "")).upper(), float(args.get("amount", 0))))
        return "refund issued"
    if name == "escalate_to_human":
        state["escalated"] = True
        return "a person will take over"
    return "unknown tool"


def converse(case: dict, prompt: str, model: str, base_url: str, key: str, max_turns: int = 5,
             temperature: float | None = 0.7) -> dict:
    messages = [{"role": "system", "content": PROMPTS[prompt]}, {"role": "user", "content": case["message"]}]
    state = {"refunds": [], "escalated": False}
    calls, usage, t0, reply, error = [], {"in": 0, "out": 0}, time.time(), "", None
    try:
        for _ in range(max_turns):
            resp = chat(base_url, key, model, messages, temperature=temperature)
            u = resp.get("usage") or {}
            usage["in"] += u.get("prompt_tokens", 0)
            usage["out"] += u.get("completion_tokens", 0)
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             **({"tool_calls": tool_calls} if tool_calls else {})})
            if not tool_calls:
                reply = msg.get("content") or ""
                break
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls.append({"name": name, "args": args, "t": time.time()})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": run_tool(name, args, state)})
    except Exception as exc:                                   # provider or network failure: an infra error
        error = f"{type(exc).__name__}: {exc}"[:200]
    return {"calls": calls, "state": state, "reply": reply, "usage": usage, "seconds": time.time() - t0,
            "t0": t0, "error": error}


def outcome(case: dict, conv: dict) -> bool | None:
    """The ground truth from the order database (not a judge): did the agent do the right thing?"""
    if conv["error"]:
        return None
    refunds, want = conv["state"]["refunds"], case["expect"]
    if want == "refund":
        total = ORDERS[case["order"]]["total"]
        return refunds == [(case["order"], float(total))]
    if want == "no_refund":
        return not refunds
    if want == "escalate":
        return conv["state"]["escalated"] and not refunds
    return None                                       # production traffic: no ground truth


def otlp_trace(case: dict, run: int, model: str, conv: dict) -> list[dict]:
    """One trace as OTLP spans (GenAI semantic conventions)."""
    trace_id = hashlib.md5(f"{case['case']}-{run}-{model}".encode()).hexdigest()

    def attr(k, v):
        return {"key": k, "value": {"intValue": v} if isinstance(v, int) else {"stringValue": str(v)}}
    spans = [{"traceId": trace_id, "spanId": trace_id[:16], "name": "chat",
              "startTimeUnixNano": str(int(conv["t0"] * 1e9)),
              "endTimeUnixNano": str(int((conv["t0"] + conv["seconds"]) * 1e9)),
              "attributes": [attr("gen_ai.operation.name", "chat"), attr("gen_ai.request.model", model),
                             attr("gen_ai.usage.input_tokens", conv["usage"]["in"]),
                             attr("gen_ai.usage.output_tokens", conv["usage"]["out"]),
                             attr("agentstats.case", case["case"])]}]
    for i, c in enumerate(conv["calls"]):
        spans.append({"traceId": trace_id, "spanId": f"{trace_id[:14]}{i:02d}", "name": f"execute_tool {c['name']}",
                      "startTimeUnixNano": str(int(c["t"] * 1e9)), "endTimeUnixNano": str(int(c["t"] * 1e9)),
                      "attributes": [attr("gen_ai.operation.name", "execute_tool"), attr("gen_ai.tool.name", c["name"]),
                                     attr("gen_ai.tool.call.arguments", json.dumps(c["args"]))]})
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--base-url", default="https://api.together.xyz/v1")
    ap.add_argument("--key-env", default="TOGETHER_API_KEY")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--price-in", type=float, default=0.30, help="$ per million input tokens")
    ap.add_argument("--price-out", type=float, default=1.20, help="$ per million output tokens")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-temperature", action="store_true", help="for models that reject one (Claude Haiku 5.5)")
    ap.add_argument("--cases", default="support_cases.jsonl", help="cases file in config/")
    ap.add_argument("--limit", type=int, help="only the first N cases (for a quick check)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    key = os.environ[args.key_env]
    cases = load_cases(args.cases)[:args.limit] if args.limit else load_cases(args.cases)
    rules = json.loads((HERE / "config" / "rules.json").read_text())
    jobs = [(c, t) for t in range(1, args.trials + 1) for c in cases]
    with ThreadPoolExecutor(args.workers) as pool:
        temp = None if args.no_temperature else 0.7
        convs = list(pool.map(lambda j: converse(j[0], args.prompt, args.model, args.base_url, key,
                                                 temperature=temp), jobs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, spans, items, cost = [], [], [], 0.0
    for (case, run), conv in zip(jobs, convs, strict=True):
        c = (conv["usage"]["in"] * args.price_in + conv["usage"]["out"] * args.price_out) / 1e6
        cost += c
        rows.append({"arm": f"{args.model.split('/')[-1]}:{args.prompt}", "case": case["case"], "run": run,
                     "passed": outcome(case, conv), "infra_error": conv["error"],
                     "tags": [f"intent:{case['intent']}"], "checks": run_checks(rules, conv["calls"]),
                     "cost_usd": round(c, 6), "input_tokens": conv["usage"]["in"],
                     "output_tokens": conv["usage"]["out"], "tool_calls": len(conv["calls"]),
                     "latency_s": round(conv["seconds"], 2)})
        spans += otlp_trace(case, run, args.model, conv)
        if not conv["error"]:
            order = ORDERS.get(case.get("order", ""), {})
            items.append({"item": f"{case['case']}#{run}", "case": case["case"], "run": run,
                          "tags": [f"intent:{case['intent']}"], "customer": case["message"],
                          "order_record": json.dumps(order) if order else "no order mentioned",
                          "agent_reply": conv["reply"]})
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "traces.json").write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}))
    (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    passed = [r["passed"] for r in rows if r["passed"] is not None]
    print(f"{args.prompt} on {args.model}: {sum(passed)}/{len(passed)} passed, "
          f"{sum(1 for r in rows if r['infra_error'])} infra errors, ${cost:.4f}")


if __name__ == "__main__":
    main()
