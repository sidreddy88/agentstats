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
     {"name": "no_forbidden_promise", "question": "Did the reply promise anything the policy forbids?", "pass": "no"}]
An item passes when every question gets its passing answer. Yes/no questions are easier for a model to answer
consistently than a 1-10 score, and each failed question says what went wrong.

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
SYSTEM = ("You grade an AI agent's work. Answer each question about the material with yes or no, judging only what "
          "the material shows. Reply with one JSON object mapping each question's name to \"yes\" or \"no\", and "
          "nothing else.")

Chat = Callable[[list[dict]], str]


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
        names.add(q["name"])
    return rubric


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
    questions = "\n".join(f"- {q['name']}: {q['question']}" for q in rubric)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{material}\n\n## Questions\n{questions}"}]


def parse(reply: str, rubric: list[dict]) -> dict[str, bool] | None:
    """Rubric answers as {name: answered the passing answer}, or None if the reply isn't usable."""
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    out = {}
    for q in rubric:
        v = str(data.get(q["name"], "")).strip().lower()
        if v not in ("yes", "no"):
            return None
        out[q["name"]] = v == q.get("pass", "yes")
    return out


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
    answers = parse(reply, rubric)
    return {"answers": answers, "error": None, "unparsed": answers is None, "reply": None if answers else reply[:300]}


def _majority(answers: list[dict[str, bool]]) -> dict[str, bool]:
    """Per question, the answer most repeats gave (a tie counts as not passing: the cautious reading)."""
    return {k: sum(a[k] for a in answers) * 2 > len(answers) for k in answers[0]}


def run(items: list[dict], rubric: list[dict], chat: Chat, *, repeats: int = 1, workers: int = 4,
        fields: list[str] | None = None) -> tuple[list[dict], list[dict], dict]:
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
    counts = {"calls": len(jobs), "judged": 0, "unparsed": 0, "errors": 0}
    for (i, r), res in zip(jobs, results, strict=True):
        if res["error"]:
            counts["errors"] += 1
            errors.setdefault(i, res["error"])
            continue
        if res["answers"] is None:
            counts["unparsed"] += 1
            labels.append({"item": str(items[i]["item"]), "rater": "judge", "verdict": "na", "run": r})
            continue
        counts["judged"] += 1
        per_item[i].append(res["answers"])
        labels.append({"item": str(items[i]["item"]), "rater": "judge", "run": r,
                       "verdict": "pass" if all(res["answers"].values()) else "fail"})
    rows = []
    for i, item in enumerate(items):
        row = {"case": str(item.get("case", item["item"])), "run": int(item.get("run", 1)),
               "arm": item.get("arm", "default"), "tags": list(item.get("tags") or []),
               "passed": None, "infra_error": None, "checks": {}}
        if per_item[i]:
            row["checks"] = _majority(per_item[i])
            row["passed"] = all(row["checks"].values())
        else:                                      # no usable verdict: the judge failed, not the agent
            row["infra_error"] = errors.get(i, "judge reply not parsed")
        rows.append(row)
    return labels, rows, counts
