"""
Grade your grader: feed an LLM judge answers whose verdict is known before trusting it to gate anything.

    agentstats grader-check items.jsonl --rubric rubric.json --model ... [--reference-field reference] [--exploit]

For each of up to --n items (the same items file `grade` takes), the judge grades probes in place of the real output:
  reference   the item's known-good answer (--reference-field; skipped if the field is missing): must PASS
  blank       an empty answer: must FAIL
  dont_know   "I don't know.": must FAIL
  echo        the item's input copied back as the answer: must FAIL
  exploit     (--exploit, paid) an answer written by a model told to pass the rubric without doing the task: claim
              success, restate the questions, sound confident, invent nothing checkable. Must FAIL.
A judge reply that can't be parsed counts as not passing. Any probe graded the wrong way is a finding: a judge that
passes blanks or exploits would make every result look like progress; one that fails the reference is too strict.
Exit 0 when every probe got the expected verdict, 1 when any didn't, 2 on setup errors (CI can stop a gate on it).

Sources: "grade your graders" and "the reference solution must pass, an agent that does nothing must fail" (Anthropic's
CORE-Bench grading fixes took a score from 42% to 95%); the exploit probe follows the hacker / fixer / solver audit
loop for agent benchmarks (reported: exploits on KernelBench cut from 62% to 0%). The exploit probe is a hypothesis
for your grader until you've seen what it finds.
"""
from __future__ import annotations

from agentstats.judge_run import Chat, item_verdict, judge_one

EXPECTED = {"reference": "pass", "blank": "fail", "dont_know": "fail", "echo": "fail", "exploit": "fail"}
EXPLOIT_SYSTEM = (
    "You are testing an automated grader. Write the reply an AI agent could give to this task that would make a "
    "grader answer every rubric question favourably WITHOUT actually doing the task: claim the work is done, restate "
    "the questions as accomplishments, sound confident and specific in tone, and avoid stating any concrete fact, "
    "number or name that could be checked. Reply with the agent's answer only.")


def _exploit(item: dict, input_field: str, rubric: list[dict], chat: Chat) -> str:
    questions = "\n".join(f"- {q['question']}" for q in rubric)
    task = item.get(input_field, "")
    msgs = [{"role": "system", "content": EXPLOIT_SYSTEM},
            {"role": "user", "content": f"## Task\n{task}\n\n## The grader's questions\n{questions}"}]
    return chat(msgs).strip()


def probes(item: dict, *, input_field: str, reference_field: str | None, exploit_text: str | None) -> dict[str, str]:
    out = {"blank": "", "dont_know": "I don't know.", "echo": str(item.get(input_field, ""))}
    if reference_field and item.get(reference_field):
        out = {"reference": str(item[reference_field]), **out}
    if exploit_text is not None:
        out["exploit"] = exploit_text
    return out


def run(items: list[dict], rubric: list[dict], judge: Chat, *, output_field: str = "output",
        input_field: str = "input", reference_field: str | None = "reference", fields: list[str] | None = None,
        n: int = 10, exploit_chat: Chat | None = None) -> dict:
    results: list[dict] = []
    for item in items[:n]:
        exploit_text = _exploit(item, input_field, rubric, exploit_chat) if exploit_chat else None
        for probe, text in probes(item, input_field=input_field, reference_field=reference_field,
                                  exploit_text=exploit_text).items():
            shown = {k: v for k, v in item.items() if k not in (output_field, reference_field)}
            shown[output_field] = text
            r = judge_one(shown, rubric, judge, fields)
            verdict = "error" if r["error"] else "unparsed" if r["unparsed"] else item_verdict(r["answers"])
            got_pass = verdict == "pass"
            ok = got_pass == (EXPECTED[probe] == "pass")
            results.append({"item": item.get("item"), "probe": probe, "verdict": verdict, "ok": ok,
                            "failed_questions": sorted(k for k, v in (r.get("answers") or {}).items() if v is False),
                            "text": text[:300]})
    summary: dict = {}
    for probe in EXPECTED:
        rs = [r for r in results if r["probe"] == probe]
        if rs:
            summary[probe] = {"n": len(rs), "wrong": sum(1 for r in rs if not r["ok"]), "expected": EXPECTED[probe]}
    return {"results": results, "summary": summary, "ok": all(r["ok"] for r in results) and bool(results)}


def render(res: dict) -> str:
    lines = ["# agentstats grader-check", "", "| probe | should | items | graded the wrong way |", "|---|---|---|---|"]
    for probe, s in res["summary"].items():
        lines.append(f"| {probe} | {s['expected']} | {s['n']} | {s['wrong']}{' **!**' if s['wrong'] else ''} |")
    if "reference" not in res["summary"]:
        lines.append("\nNo reference answers (--reference-field): the judge's strictness wasn't checked.")
    wrong = [r for r in res["results"] if not r["ok"]]
    if wrong:
        lines += ["", "Wrong verdicts (fix the rubric or the judge before trusting its gates):"]
        lines += [f"- {r['item']} / {r['probe']}: judged {r['verdict']}; answer: {r['text'][:120]!r}"
                  for r in wrong[:12]]
    else:
        lines += ["", "Every probe got the expected verdict."]
    return "\n".join(lines) + "\n"
