"""agentstats coverage: is the eval set like production?"""
from agentstats import coverage


def _eval(spec: dict[str, int], runs: int = 2, prefix: str = "template") -> list[dict]:
    rows, k = [], 0
    for value, n in spec.items():
        for _ in range(n):
            rows += [{"case": f"c{k}", "run": r, "tags": [f"{prefix}:{value}"]} for r in range(1, runs + 1)]
            k += 1
    return rows


def _prod(spec: dict[str, int]) -> list[dict]:
    return [{"case": f"t{v}{i}", "tags": [f"error_type:{v}"]} for v, n in spec.items() for i in range(n)]


def _row(res, value):
    return next(r for r in res["prefixes"]["error_type"] if r["value"] == value)


def test_cases_count_once_and_rename_maps_prefixes():
    res = coverage.compare(_eval({"A": 30, "B": 30}), _prod({"A": 50, "B": 50}), ["error_type"],
                           {"template": "error_type"})
    assert res["eval_cases"] == 60 and res["production_runs"] == 100
    assert _row(res, "A")["eval"] == 30 and _row(res, "A")["eval_share"] == 0.5
    assert not any(r.get("flag") for r in res["prefixes"]["error_type"])          # same mix: no gap


def test_a_value_production_sees_but_the_eval_set_lacks_is_missing():
    res = coverage.compare(_eval({"A": 60}), _prod({"A": 5, "GENERIC": 5}), ["error_type"], {"template": "error_type"})
    assert _row(res, "GENERIC")["flag"] == "missing"


def test_under_and_over_need_an_interval_that_excludes_zero():
    res = coverage.compare(_eval({"A": 95, "B": 5}), _prod({"A": 100, "B": 100}), ["error_type"],
                           {"template": "error_type"})
    assert _row(res, "B")["flag"] == "under" and _row(res, "A")["flag"] == "over"
    assert _row(res, "B")["difference"]["high"] < 0                                  # eval minus production
    small = coverage.compare(_eval({"A": 95, "B": 5}), _prod({"A": 1, "B": 1}), ["error_type"],
                             {"template": "error_type"})
    assert "flag" not in _row(small, "B")                                            # 2 runs prove nothing


def test_eval_only_values_are_listed_not_flagged():
    res = coverage.compare(_eval({"A": 50, "RARE": 10}), _prod({"A": 100}), ["error_type"], {"template": "error_type"})
    assert "flag" not in _row(res, "RARE")


def test_a_prefix_the_eval_set_never_tags_is_reported_not_flagged():
    res = coverage.compare(_eval({"A": 10}), _prod({"A": 10}), ["severity"], {"template": "error_type"})
    assert res["untagged"] == ["severity"] and "severity" not in res["prefixes"]
    assert "no `severity:` tags" in coverage.render(res)
