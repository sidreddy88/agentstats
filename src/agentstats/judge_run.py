"""
Run an LLM judge on binary rubric questions.

    agentstats grade items.jsonl --rubric rubric.json --model deepseek-ai/DeepSeek-V4.1-Flash \\
        --base-url https://api.together.xyz/v1 --api-key-env TOGETHER_API_KEY --repeats 3 \\
        --out judge_labels.jsonl --rows judged_rows.jsonl
    agentstats judge judge_labels.jsonl human_labels.jsonl      # consistency, and agreement with people

Items (JSONL), one per answer to grade: {"item": "conv-07", "input": "...", "output": "..."}. Every field except
`item`, `case`, `run`, `arm` and `tags` is shown to the judge (or only --fields). `case`/`run`/`arm`/`tags` are carried
into the judged rows.

Rubric (JSON): a list of yes/no questions and the answer that counts as passing:
    [{"name": "correct_amount", "question": "Did the reply state the correct refund amount?", "pass": "yes"},
     {"name": "label_when_eligible", "question": "If the order was eligible, did it generate a label?",
      "pass": "yes", "conditional": true}]
An item passes when every applicable question gets its passing answer. A question marked "conditional" may be
answered n/a (its condition doesn't hold) and is then left out; n/a on any other question makes the reply unusable.
Any question may be answered "unknown" when the material doesn't show enough to answer it (e.g. the tool results the
question is about weren't passed in): the judge isn't pushed into guessing yes or no. Unknown never passes: it counts
as a failed question, and the summary reports how many answers were unknown, since many of them usually mean the
judge is missing material (--fields) rather than that the agent failed.
Every answer comes with a one-sentence reason (in the label rows), and every verdict records the judge version
(model, rubric hash, prompt format) so results from a changed judge aren't compared as if nothing changed.
Yes/no questions are easier for a model to answer consistently than a 1-10 score, and each failed question says what
went wrong.

The judge is any OpenAI-compatible chat endpoint (Together, OpenAI, vLLM, Ollama, ...), called with the standard
library: no new dependency. A reply that isn't the JSON asked for gets "na" for that item (recorded, never a silent
pass); network or server errors are retried, then recorded as an infrastructure error.

Outputs: judge label rows ({"item", "rater": "judge", "verdict", "run"}) for `agentstats judge`, and with --rows,
attempt rows whose checks are the rubric answers (true = the passing answer), so a rubric question can be gated like
a behaviour rule (`min_rule_rate`, `no_rule_regression`).
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CARRIED = ("item", "case", "run", "arm", "tags")
PROMPT_VERSION = 3   # bump when SYSTEM or the reply format changes: it's part of the judge version on every verdict
SYSTEM = ("You grade an AI agent's work. Answer each question about the material with yes or no, judging only what "
          "the material shows. A question marked [conditional] may be answered n/a when the material shows its "
          "condition doesn't hold; never answer n/a otherwise. If the material doesn't show enough to answer a "
          "question, answer unknown rather than guess. Give each answer a one-sentence reason that points to "
          "the evidence in the material. Reply with one JSON object mapping each question's name to "
          "{\"answer\": \"yes\" | \"no\" | \"n/a\" | \"unknown\", \"reason\": \"...\"}, and nothing else.")

Chat = Callable[[list[dict]], str]


NEGATED = ("no_", "not_", "non_", "never_", "without_")


def rubric_warnings(rubric: list[dict]) -> list[str]:
    """Question names that read as the opposite of their question. A judge sometimes answers the name instead of the
    question: "no_extra_promises" with question "Did it promise anything extra?" and pass "no" is a double negative,
    and in a real run the judge flipped between yes and no on the same reply. Name questions after what they ask."""
    return [f"question {q['name']!r} has a negated name and pass 'no': a judge may answer the name instead of the "
            f"question. Rename it to what the question asks (e.g. {q['name'].split('_', 1)[-1]!r})"
            for q in rubric if q.get("pass", "yes") == "no" and q["name"].lower().startswith(NEGATED)]


def load_rubric(path: str | Path) -> list[dict]:
    rubric = json.loads(Path(path).read_text())
    if not isinstance(rubric, list) or not rubric:
        raise ValueError("the rubric must be a non-empty JSON list of questions")
    names = set()
    for q in rubric:
        if not {"name", "question"} <= set(q) or q.get("pass", "yes") not in ("yes", "no"):
            raise ValueError(f"each question needs name, question and pass (yes or no): {q}")
        if q["name"] in names:
            raise ValueError(f"duplicate question name {q['name']!r}")
        if not isinstance(q.get("conditional", False), bool):
            raise ValueError(f"'conditional' must be true or false: {q}")
        names.add(q["name"])
    return rubric


def judge_version(model: str, rubric: list[dict]) -> str:
    """What produced a verdict: the judge model, the rubric's content and the prompt format. Recorded on every label
    and judged row, so a change in judge results can be traced to a change in the judge, and verdicts from different
    judge versions aren't compared as if they were the same instrument."""
    import hashlib
    digest = hashlib.sha256(json.dumps(rubric, sort_keys=True).encode()).hexdigest()[:10]
    return f"{model}|rubric:{digest}|prompt:{PROMPT_VERSION}"


def load_items(path: str | Path) -> list[dict]:
    items = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    missing = [i for i in items if "item" not in i]
    if missing:
        raise ValueError(f"every item needs an 'item' id (e.g. {str(missing[0])[:80]})")
    return items


def prompt(item: dict, rubric: list[dict], fields: list[str] | None = None) -> list[dict]:
    keys = fields or [k for k in item if k not in CARRIED]
    material = "\n\n".join(f"## {k}\n{item[k] if isinstance(item[k], str) else json.dumps(item[k])}"
                           for k in keys if k in item)
    questions = "\n".join(f"- {q['name']}{' [conditional]' if q.get('conditional') else ''}: {q['question']}"
                          for q in rubric)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{material}\n\n## Questions\n{questions}"}]


def parse(reply: str, rubric: list[dict]) -> tuple[dict[str, bool | None], dict[str, str], list[str]] | None:
    """({name: answered the passing answer, or None for n/a}, {name: reason}, [names answered unknown]), or None if
    the reply isn't usable. An unknown answer is recorded as not passing (False) and listed.
    Accepts {"q": {"answer": "yes", "reason": "..."}} and the older {"q": "yes"}. n/a only counts for a question
    marked conditional; on any other question it makes the reply unusable (a judge mustn't dodge a question)."""
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    answers: dict[str, bool | None] = {}
    reasons: dict[str, str] = {}
    unknown: list[str] = []
    for q in rubric:
        raw = data.get(q["name"])
        value, reason = (raw.get("answer"), raw.get("reason")) if isinstance(raw, dict) else (raw, None)
        v = str(value or "").strip().lower()
        if v in ("n/a", "na", "not applicable") and q.get("conditional"):
            answers[q["name"]] = None
        elif v in ("unknown", "can't tell", "cannot tell"):
            answers[q["name"]] = False
            unknown.append(q["name"])
        elif v in ("yes", "no"):
            answers[q["name"]] = v == q.get("pass", "yes")
        else:
            return None
        if reason:
            reasons[q["name"]] = str(reason)[:300]
    return answers, reasons, unknown


def item_verdict(answers: dict[str, bool | None]) -> str:
    """pass when every applicable question passed; na when none applied (every question n/a)."""
    applicable = [v for v in answers.values() if v is not None]
    if not applicable:
        return "na"
    return "pass" if all(applicable) else "fail"


def http_chat(base_url: str, model: str, api_key: str | None, *, temperature: float | None = 0.0,
              timeout: float = 120.0, retries: int = 3) -> Chat:
    """An OpenAI-compatible chat call (POST {base_url}/chat/completions), standard library only.

    Sends its own User-Agent: some providers' edge (Cloudflare error 1010, seen on Together) blocks Python's default
    "Python-urllib" one with a 403."""
    import agentstats

    url = base_url.rstrip("/") + "/chat/completions"

    def chat(messages: list[dict]) -> str:
        body = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": f"agentstats/{agentstats.__version__}",
                                              **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429) and e.code < 500 or attempt == retries:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == retries:
                    raise
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("unreachable")
    return chat


def judge_one(item: dict, rubric: list[dict], chat: Chat, fields: list[str] | None = None) -> dict:
    try:
        reply = chat(prompt(item, rubric, fields))
    except Exception as exc:                       # transport or server failure after retries
        return {"answers": None, "error": f"{type(exc).__name__}: {exc}"[:200], "unparsed": False}
    parsed = parse(reply, rubric)
    answers, reasons, unknown = parsed if parsed else (None, {}, [])
    return {"answers": answers, "reasons": reasons, "unknown": unknown, "error": None, "unparsed": parsed is None,
            "reply": None if parsed else reply[:300]}


def _majority(answers: list[dict[str, bool | None]]) -> dict[str, bool]:
    """Per question, the answer most repeats gave, over the repeats where it applied (a tie counts as not passing:
    the cautious reading). A question n/a in every repeat is left out."""
    out = {}
    for k in answers[0]:
        applied = [a[k] for a in answers if a.get(k) is not None]
        if applied:
            out[k] = sum(applied) * 2 > len(applied)
    return out


def run(items: list[dict], rubric: list[dict], chat: Chat, *, repeats: int = 1, workers: int = 4,
        fields: list[str] | None = None, version: str | None = None) -> tuple[list[dict], list[dict], dict]:
    """(judge label rows, attempt rows with rubric checks, summary).

    Every repeat is a judge label row (so `agentstats judge` can measure the judge's flip rate). Attempt rows are one
    per item: the per-question majority over the repeats that parsed. Repeats are the judge's, not the agent's, so
    they never become extra attempts.
    """
    jobs = [(i, r) for r in range(1, repeats + 1) for i in range(len(items))]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda job: judge_one(items[job[0]], rubric, chat, fields), jobs))
    labels: list[dict] = []
    per_item: dict[int, list[dict]] = {i: [] for i in range(len(items))}
    errors: dict[int, str] = {}
    counts = {"calls": len(jobs), "judged": 0, "unparsed": 0, "errors": 0, "unknown_answers": 0}
    for (i, r), res in zip(jobs, results, strict=True):
        if res["error"]:
            counts["errors"] += 1
            errors.setdefault(i, res["error"])
            continue
        stamp = {"judge": version} if version else {}
        if res["answers"] is None:
            counts["unparsed"] += 1
            labels.append({"item": str(items[i]["item"]), "rater": "judge", "verdict": "na", "run": r, **stamp})
            continue
        counts["judged"] += 1
        counts["unknown_answers"] += len(res["unknown"])
        per_item[i].append(res["answers"])
        labels.append({"item": str(items[i]["item"]), "rater": "judge", "run": r,
                       "verdict": item_verdict(res["answers"]), "reasons": res["reasons"],
                       **({"unknown": res["unknown"]} if res["unknown"] else {}), **stamp})
    rows = []
    for i, item in enumerate(items):
        tags = list(item.get("tags") or []) + ([f"judge:{version}"] if version else [])
        row = {"case": str(item.get("case", item["item"])), "run": int(item.get("run", 1)),
               "arm": item.get("arm", "default"), "tags": tags,
               "passed": None, "infra_error": None, "checks": {}}
        if per_item[i]:
            row["checks"] = _majority(per_item[i])
            row["passed"] = all(row["checks"].values()) if row["checks"] else None
        else:                                      # no usable verdict: the judge failed, not the agent
            row["infra_error"] = errors.get(i, "judge reply not parsed")
        rows.append(row)
    return labels, rows, counts
