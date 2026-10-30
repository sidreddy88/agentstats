"""LLM judge on yes/no rubric questions (judge_run.py, `agentstats grade`). No real API calls: a fake chat function,
and a local HTTP server for the OpenAI-compatible client."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentstats import cli, judge, judge_run

RUBRIC = [{"name": "correct_amount", "question": "Did the reply state the correct refund amount?", "pass": "yes"},
          {"name": "forbidden_promise", "question": "Did the reply promise anything policy forbids?", "pass": "no"}]
ITEMS = [{"item": "conv-1", "case": "c1", "tags": ["intent:refund"], "input": "Refund order A1",
          "output": "Refunded $20"},
         {"item": "conv-2", "case": "c2", "input": "Refund order A2", "output": "Refunded $20, and a free TV"},
         {"item": "conv-3", "case": "c3", "input": "Refund order A3", "output": "Refunded $5 (wrong amount)"}]


def fake(answers_by_output: dict[str, str]):
    """A chat function that answers from the item's output text, as a judge would."""
    def chat(messages):
        text = messages[-1]["content"]
        for key, reply in answers_by_output.items():
            if key in text:
                return reply
        return "not json"
    return chat


GOOD = {"free TV": '{"correct_amount": "yes", "forbidden_promise": "yes"}',
        "wrong amount": 'Here you go: {"correct_amount": "no", "forbidden_promise": "no"}',
        "$20": '{"correct_amount": "yes", "forbidden_promise": "no"}'}


def test_rubric_and_items_are_validated(tmp_path):
    bad = tmp_path / "r.json"
    bad.write_text(json.dumps([{"name": "a", "question": "q", "pass": "maybe"}]))
    with pytest.raises(ValueError, match="pass"):
        judge_run.load_rubric(bad)
    bad.write_text(json.dumps([{"name": "a", "question": "q"}, {"name": "a", "question": "r"}]))
    with pytest.raises(ValueError, match="duplicate"):
        judge_run.load_rubric(bad)
    items = tmp_path / "i.jsonl"
    items.write_text(json.dumps({"input": "x"}) + "\n")
    with pytest.raises(ValueError, match="'item' id"):
        judge_run.load_items(items)


def test_negated_question_names_with_pass_no_are_flagged():
    rubric = [{"name": "no_extra_promises", "question": "Did it promise anything extra?", "pass": "no"},
              {"name": "extra_promises", "question": "Did it promise anything extra?", "pass": "no"},
              {"name": "not_rude", "question": "Was it polite?", "pass": "yes"}]
    warnings = judge_run.rubric_warnings(rubric)
    assert len(warnings) == 1 and "'no_extra_promises'" in warnings[0] and "'extra_promises'" in warnings[0]


def test_prompt_shows_material_but_not_ids_or_tags():
    msgs = judge_run.prompt(ITEMS[0], RUBRIC)
    user = msgs[-1]["content"]
    assert "## input\nRefund order A1" in user and "## output\nRefunded $20" in user
    assert "conv-1" not in user and "intent:refund" not in user
    assert "- forbidden_promise: Did the reply promise" in user
    assert "## output" not in judge_run.prompt(ITEMS[0], RUBRIC, fields=["input"])[-1]["content"]


def test_parse_maps_answers_to_pass_and_rejects_bad_replies():
    assert judge_run.parse('{"correct_amount": "Yes", "forbidden_promise": "no"}', RUBRIC) == \
        {"correct_amount": True, "forbidden_promise": True}
    assert judge_run.parse('{"correct_amount": "yes", "forbidden_promise": "yes"}', RUBRIC)["forbidden_promise"] \
        is False                                            # "yes" to a pass=no question fails it
    assert judge_run.parse("I think yes", RUBRIC) is None
    assert judge_run.parse('{"correct_amount": "yes"}', RUBRIC) is None           # a question unanswered
    assert judge_run.parse('{"correct_amount": "maybe", "forbidden_promise": "no"}', RUBRIC) is None


def test_run_verdicts_rows_and_checks():
    labels, rows, counts = judge_run.run(ITEMS, RUBRIC, fake(GOOD))
    assert counts == {"calls": 3, "judged": 3, "unparsed": 0, "errors": 0}
    assert [lab["verdict"] for lab in labels] == ["pass", "fail", "fail"]
    by_case = {r["case"]: r for r in rows}
    assert by_case["c1"]["passed"] is True and by_case["c1"]["tags"] == ["intent:refund"]
    assert by_case["c2"]["checks"] == {"correct_amount": True, "forbidden_promise": False}
    assert by_case["c3"]["checks"] == {"correct_amount": False, "forbidden_promise": True}


def test_repeats_are_judge_labels_not_extra_attempts_and_rows_use_the_majority():
    replies = iter(['{"correct_amount": "yes", "forbidden_promise": "no"}'] * 2
                   + ['{"correct_amount": "no", "forbidden_promise": "no"}'])
    labels, rows, counts = judge_run.run(ITEMS[:1], RUBRIC, lambda m: next(replies), repeats=3, workers=1)
    assert len(labels) == 3 and {lab["run"] for lab in labels} == {1, 2, 3}
    assert len(rows) == 1 and rows[0]["passed"] is True                           # 2 of 3 said the amount was right
    assert judge.analyze(labels)["consistency"]["flipped"] == ["conv-1"]


def test_unparsed_reply_and_transport_error_are_judge_failures_not_agent_failures():
    def chat(messages):
        if "A2" in messages[-1]["content"]:
            raise ConnectionError("down")
        if "A3" in messages[-1]["content"]:
            return "no idea"
        return GOOD["$20"]
    labels, rows, counts = judge_run.run(ITEMS, RUBRIC, chat)
    assert counts == {"calls": 3, "judged": 1, "unparsed": 1, "errors": 1}
    by_case = {r["case"]: r for r in rows}
    assert by_case["c2"]["infra_error"].startswith("ConnectionError") and by_case["c2"]["passed"] is None
    assert by_case["c3"]["infra_error"] == "judge reply not parsed"
    assert {lab["item"]: lab["verdict"] for lab in labels} == {"conv-1": "pass", "conv-3": "na"}


class _Handler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.calls.append((self.path, self.headers.get("Authorization"), body, self.headers.get("User-Agent")))
        if len(_Handler.calls) == 1:                       # first call: a transient server error, then success
            self.send_response(503)
            self.end_headers()
            return
        reply = {"choices": [{"message": {"content": '{"correct_amount": "yes", "forbidden_promise": "no"}'}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_http_client_against_a_local_openai_compatible_server(monkeypatch):
    monkeypatch.setattr(judge_run.time, "sleep", lambda s: None)
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        chat = judge_run.http_chat(f"http://127.0.0.1:{server.server_port}/v1", "m", "secret", temperature=0.0)
        assert "correct_amount" in chat([{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()
    path, auth, body, agent = _Handler.calls[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer secret"
    assert agent.startswith("agentstats/")                 # not Python-urllib, which some edges block (403)
    assert body["model"] == "m" and body["temperature"] == 0.0 and len(_Handler.calls) == 2   # retried the 503


def test_cli_grade_with_a_fake_judge(tmp_path, monkeypatch):
    items, rubric = tmp_path / "items.jsonl", tmp_path / "rubric.json"
    items.write_text("".join(json.dumps(i) + "\n" for i in ITEMS))
    rubric.write_text(json.dumps(RUBRIC))
    out, rows = tmp_path / "labels.jsonl", tmp_path / "rows.jsonl"
    args = cli.argparse.Namespace(rubric=str(rubric), items=str(items), model="m", base_url="x", api_key_env="K",
                                  temperature=0.0, repeats=1, workers=1, fields=None, out=str(out), rows=str(rows))
    assert cli._grade(args, chat=fake(GOOD)) == 0
    assert len(out.read_text().splitlines()) == 3
    assert cli.main(["summary", str(rows)]) == 0
    monkeypatch.delenv("K", raising=False)
    assert cli.main(["grade", str(items), "--rubric", str(rubric), "--model", "m", "--api-key-env", "K",
                     "--out", str(out)]) == 2                                    # no API key: a clear message
