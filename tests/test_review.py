"""Sampled human review (review.py): who gets reviewed, blindness, and agreement reweighted to the population."""
from __future__ import annotations

import csv
import json

import pytest

from agentstats import cli, judge, review
from agentstats.records import from_dict, write_jsonl


def population():
    """200 graded attempts: 10 pass with a rule violated (disagree), 40 fail, 150 pass; plus 5 infra failures."""
    rows = []
    for i in range(200):
        if i < 10:
            passed, ok = True, False
        elif i < 50:
            passed, ok = False, False
        else:
            passed, ok = True, True
        rows.append(from_dict({"case": f"c{i}", "passed": passed, "tags": [f"intent:{'ab'[i % 2]}"],
                               "checks": {"looked_up": ok}}))
    rows += [from_dict({"case": f"x{i}", "passed": None, "infra_error": "503"}) for i in range(5)]
    return rows


def fill(sheet_path, key, verdict_for):
    """Fill the sheet as a reviewer would: verdict_for(key row) -> 'pass' / 'fail' / 'unsure' / ''."""
    with sheet_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_item = {k["item"]: k for k in key}
    for r in rows:
        r["human_verdict"] = verdict_for(by_item[r["item"]])
    with sheet_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_strata_and_allocation_favour_disagreements():
    sheet, key, summary = review.sample(population(), 40, stratify_by="intent")
    assert summary["strata"] == {"disagree": 10, "fail": 40, "pass": 150} and summary["graded"] == 200
    assert summary["sampled"] == {"disagree": 10, "fail": 15, "pass": 15} and len(sheet) == 40
    weights = {k["stratum"]: k["weight"] for k in key}
    assert weights == {"disagree": 1.0, "fail": 40 / 15, "pass": 10.0}
    assert not any(k["case"].startswith("x") for k in key)        # infrastructure failures aren't reviewed


def test_failed_with_every_rule_satisfied_is_also_a_disagreement():
    a = from_dict({"case": "c", "passed": False, "checks": {"looked_up": True}})
    assert review.stratum(a) == "disagree"
    assert review.stratum(from_dict({"case": "c", "passed": False})) == "fail"   # no rules: no disagreement


def test_unused_places_move_on_and_the_sample_is_deterministic():
    _, _, summary = review.sample(population(), 300)
    assert summary["sampled"] == {"disagree": 10, "fail": 40, "pass": 150}       # asked for more than exist
    a = [s["item"] for s in review.sample(population(), 40, seed=1)[0]]
    assert a == [s["item"] for s in review.sample(population(), 40, seed=1)[0]]
    assert a != [s["item"] for s in review.sample(population(), 40, seed=2)[0]]


def test_sheet_is_blind_and_links_are_filled(tmp_path):
    sheet, key, summary = review.sample(population(), 20, link="https://traces/{case}?run={run}")
    sheet_path, key_path = review.write(tmp_path / "r", sheet, key, summary)
    header = sheet_path.read_text().splitlines()[0].split(",")
    assert header == review.SHEET_COLUMNS                          # no verdict, stratum or weight columns
    assert sheet[0]["link"] == f"https://traces/{sheet[0]['case']}?run=1"
    with pytest.raises(FileExistsError):
        review.write(tmp_path / "r", sheet, key, summary)          # a filled sheet is never overwritten


def test_reweighting_recovers_the_population_agreement(tmp_path):
    # The judge passed every "disagree" attempt; the reviewer fails them all and agrees everywhere else.
    # On the sample: 30/40 = 75%. On the population: (40 + 150) / 200 = 95%.
    sheet, key, summary = review.sample(population(), 40)
    sheet_path, key_path = review.write(tmp_path / "r", sheet, key, summary)
    fill(sheet_path, key, lambda k: "fail" if k["stratum"] == "disagree" else k["judge_verdict"])
    labels, rep = review.import_labels(sheet_path, key_path, rater="alice")
    assert rep["labelled"] == 40
    assert rep["on_sample"]["agreement"] == pytest.approx(0.75)
    assert rep["population"]["agreement"] == pytest.approx(0.95)
    assert rep["population"]["strictness"] == 0.0
    assert rep["population"]["leniency"] == pytest.approx(10 / (10 + 40))  # human fails: 10 disagree + 40 fail
    ci = rep["population_ci"]["agreement"]
    assert ci["low"] <= 0.95 <= ci["high"]
    assert rep["by_stratum"]["disagree"]["agreement"] == 0.0 and rep["by_stratum"]["pass"]["agreement"] == 1.0
    assert "reweighted to the population" in review.render(rep)
    # the labels feed `agentstats judge` unchanged
    out = judge.analyze(labels)
    assert out["alignment"]["n"] == 40 and out["alignment"]["fp"] == 10


def test_blank_and_unsure_rows_and_bad_input(tmp_path):
    sheet, key, summary = review.sample(population(), 10)
    sheet_path, key_path = review.write(tmp_path / "r", sheet, key, summary)
    answers = iter(["", "unsure"] + ["pass"] * 8)
    fill(sheet_path, key, lambda k: next(answers))
    labels, rep = review.import_labels(sheet_path, key_path)
    assert rep["blank"] == 1 and rep["unsure"] == 1 and rep["labelled"] == 8
    assert {r["verdict"] for r in labels if r["rater"] == "human:human"} == {"pass", "na"}
    fill(sheet_path, key, lambda k: "maybe")
    with pytest.raises(ValueError, match="use pass, fail or unsure"):
        review.import_labels(sheet_path, key_path)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"item": "nope"}) + "\n")
    with pytest.raises(ValueError, match="aren't in the key"):
        review.import_labels(sheet_path, other)


def test_cli_sample_then_import(tmp_path):
    rows = tmp_path / "rows.jsonl"
    write_jsonl(population(), rows)
    assert cli.main(["review", "sample", str(rows), "--n", "12", "--stratify-by", "intent",
                     "--out", str(tmp_path / "wk1")]) == 0
    key = [json.loads(x) for x in (tmp_path / "wk1" / "key.jsonl").read_text().splitlines()]
    fill(tmp_path / "wk1" / "review.csv", key, lambda k: k["judge_verdict"])
    labels = tmp_path / "labels.jsonl"
    wk1 = tmp_path / "wk1"
    assert cli.main(["review", "import", str(wk1 / "review.csv"), "--key", str(wk1 / "key.jsonl"),
                     "--rater", "bob", "--out", str(labels)]) == 0
    assert cli.main(["judge", str(labels)]) == 0
    assert len(labels.read_text().splitlines()) == 24              # one human + one judge row per item



def test_precision_note_says_what_a_sample_size_buys():
    note = review.precision_note(20)
    assert "With 20 reviewed items" in note and "+-16 points" in note and "+-10 points needs 49" in note


def test_cases_that_fail_every_run_are_their_own_group():
    rows = [from_dict({"case": f"n{i}", "run": r, "passed": False}) for i in range(4) for r in (1, 2)]   # never pass
    rows += [from_dict({"case": f"f{i}", "run": r, "passed": r == 1}) for i in range(10) for r in (1, 2)]  # flaky
    rows += [from_dict({"case": "once", "passed": False})]                       # one run: not enough to call it
    sheet, key, summary = review.sample(rows, 20)
    assert summary["strata"] == {"never": 8, "fail": 11, "pass": 10}
    assert {k["case"] for k in key if k["stratum"] == "never"} <= {f"n{i}" for i in range(4)}
    assert review.never_passed(rows) == frozenset(("default", f"n{i}") for i in range(4))


def test_a_rule_disagreement_outranks_never():
    a = from_dict({"case": "n", "run": 1, "passed": False, "checks": {"looked_up": True}})
    assert review.stratum(a, frozenset({("default", "n")})) == "disagree"


def test_report_explains_the_never_group():
    report = {"labelled": 3, "unsure": 0, "blank": 0, "population": {}, "population_ci": {}, "on_sample": {},
              "by_stratum": {"never": {"n": 3, "agreement": 0.33}}}
    assert "check the task and the grader, not the agent" in review.render(report)
