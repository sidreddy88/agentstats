"""grader-check: known-answer probes for an LLM judge. No network: the judge is a scripted function."""
import json

from agentstats import cli, grader_check

RUBRIC = [{"name": "answered", "question": "Did the reply answer the user's question with the right amount?",
           "pass": "yes"}]
ITEMS = [{"item": f"i{k}", "input": f"What is the refund for order {k}?", "output": f"${k}0",
          "reference": f"The refund is ${k}0."} for k in range(3)]


def judge_that(passes):
    """A fake judge: passes(answer_text) -> bool decides its yes/no."""
    def chat(messages):
        user = messages[-1]["content"]
        answer = user.split("## output\n", 1)[1].split("\n\n## ", 1)[0] if "## output\n" in user else ""
        return json.dumps({"answered": {"answer": "yes" if passes(answer) else "no", "reason": "r"}})
    return chat


def test_a_good_judge_passes_every_probe():
    res = grader_check.run(ITEMS, RUBRIC, judge_that(lambda a: a.startswith("The refund is $")))
    assert res["ok"] and set(res["summary"]) == {"reference", "blank", "dont_know", "echo"}
    assert all(s["wrong"] == 0 for s in res["summary"].values())


def test_a_lenient_judge_is_caught_on_blanks_and_echoes():
    res = grader_check.run(ITEMS, RUBRIC, judge_that(lambda a: True))
    assert not res["ok"]
    assert res["summary"]["blank"]["wrong"] == 3 and res["summary"]["echo"]["wrong"] == 3
    assert "graded the wrong way" in grader_check.render(res) and "**!**" in grader_check.render(res)


def test_a_too_strict_judge_is_caught_on_the_reference():
    res = grader_check.run(ITEMS, RUBRIC, judge_that(lambda a: False))
    assert not res["ok"] and res["summary"]["reference"]["wrong"] == 3 and res["summary"]["blank"]["wrong"] == 0


def test_exploit_probe_uses_the_exploit_writer():
    writer_calls = []

    def chat(messages):
        if messages[0]["content"].startswith("You are testing an automated grader"):
            writer_calls.append(messages)
            return "Done! I answered your question about the refund with the correct amount."
        return judge_that(lambda a: a.startswith("Done!"))(messages)      # a judge fooled by confident claims
    res = grader_check.run(ITEMS, RUBRIC, chat, exploit_chat=chat)
    assert len(writer_calls) == 3 and res["summary"]["exploit"]["wrong"] == 3 and not res["ok"]


def test_unparsed_replies_count_as_not_passing_and_no_reference_is_reported():
    items = [{k: v for k, v in it.items() if k != "reference"} for it in ITEMS]
    res = grader_check.run(items, RUBRIC, lambda m: "I refuse")
    assert "reference" not in res["summary"] and res["ok"]               # blanks etc. "not passing": correct
    assert "strictness wasn't checked" in grader_check.render(res)


def test_cli_exit_codes(tmp_path, capsys):
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in ITEMS))
    (tmp_path / "rubric.json").write_text(json.dumps(RUBRIC))
    args = cli.argparse.Namespace(items=str(tmp_path / "items.jsonl"), rubric=str(tmp_path / "rubric.json"),
                                  model="m", base_url="x", api_key_env="K", temperature=0.0, n=10,
                                  output_field="output", input_field="input", reference_field="reference",
                                  fields=None, exploit=False, json=False)
    assert cli._grader_check(args, chat=judge_that(lambda a: a.startswith("The refund is $"))) == 0
    assert cli._grader_check(args, chat=judge_that(lambda a: True)) == 1
