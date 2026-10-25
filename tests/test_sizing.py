import random

import pytest

from agentstats import cli, sizing
from agentstats.records import Attempt, write_jsonl


def baseline(cases=120, trials=4, seed=1):
    """Mostly easy cases plus some that flip, like a real agent eval."""
    r = random.Random(seed)
    rates = [r.choice([0.98, 0.98, 0.95, 0.6, 0.3, 0.02]) for _ in range(cases)]
    return [Attempt(case=f"c{i}", passed=r.random() < p, run=t)
            for t in range(1, trials + 1) for i, p in enumerate(rates)]


def test_no_change_false_alarms_near_nominal_for_drop_rule():
    res = sizing.simulate(baseline(), trials=2, rule="drop", conf=0.90, shifts=(0.0,), sims=600, seed=3)
    assert 0.02 <= res["rows"][0]["FAIL"] <= 0.10     # one-sided 5% at a 90% interval


def test_power_grows_with_the_drop():
    rows = sizing.simulate(baseline(), trials=2, rule="drop", shifts=(0.0, 0.75, 1.5), sims=300, seed=4)["rows"]
    fails = [r["FAIL"] for r in rows]
    assert fails[0] < fails[1] < fails[2] and fails[2] > 0.9
    assert rows[1]["mean_drop"] > 0 and rows[2]["mean_drop"] > rows[1]["mean_drop"]


def test_noninferiority_is_often_inconclusive_with_few_trials():
    row = sizing.simulate(baseline(cases=60), trials=1, rule="noninferiority", margin=0.02, shifts=(0.0,),
                          sims=300, seed=5)["rows"][0]
    assert row["INCONCLUSIVE"] > 0.3


def test_deterministic_and_validates_rule():
    a = sizing.simulate(baseline(cases=30), trials=2, shifts=(0.5,), sims=50, seed=9)
    b = sizing.simulate(baseline(cases=30), trials=2, shifts=(0.5,), sims=50, seed=9)
    assert a == b
    with pytest.raises(ValueError):
        sizing.simulate(baseline(cases=30), trials=2, rule="vibes")


def test_cli_size(tmp_path, capsys):
    write_jsonl(baseline(cases=40), tmp_path / "base.jsonl")
    assert cli.main(["size", "--baseline", str(tmp_path / "base.jsonl"), "--sims", "40"]) == 0
    out = capsys.readouterr().out
    assert "agentstats size" in out and "No change:" in out
