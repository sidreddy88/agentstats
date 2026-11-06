import pytest
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa

from agentstats import judge


def test_cohen_kappa_matches_statsmodels():
    a = ["pass", "pass", "fail", "pass", "fail", "fail", "pass", "pass", "fail", "pass"]
    b = ["pass", "fail", "fail", "pass", "fail", "pass", "pass", "pass", "fail", "pass"]
    cats = ("pass", "fail")
    table = [[sum(1 for x, y in zip(a, b, strict=True) if x == r and y == c) for c in cats] for r in cats]
    assert judge.cohen_kappa(a, b) == pytest.approx(cohens_kappa(table).kappa)


def test_fleiss_kappa_matches_statsmodels():
    table = [[3, 0], [2, 1], [0, 3], [1, 2], [3, 0], [3, 0], [0, 3]]
    assert judge.fleiss_kappa(table) == pytest.approx(fleiss_kappa(table, method="fleiss"))


def test_pairwise_disagreement_binary_formula():
    for s, n in [(0, 4), (1, 4), (2, 4), (3, 5)]:
        v = ["pass"] * s + ["fail"] * (n - s)
        assert judge.pairwise_disagreement(v) == pytest.approx(2 * s * (n - s) / (n * (n - 1)))


def test_pairwise_delta():
    assert judge.pairwise_delta([1, 3, 5]) == pytest.approx((2 + 4 + 2) / 3)


def _rows():
    rows = []
    human = {"i1": "pass", "i2": "pass", "i3": "fail", "i4": "fail", "i5": "pass", "i6": "fail"}
    for item, v in human.items():
        for r in ("human:a", "human:b", "human:c"):
            rows.append({"item": item, "rater": r, "verdict": v})
    rows.append({"item": "i6", "rater": "human:c", "verdict": "pass"})  # overwrites c's label: 2 fail, 1 pass
    judged = {"i1": ["pass", "pass"], "i2": ["fail", "fail"], "i3": ["fail", "pass"], "i4": ["pass", "pass"],
              "i5": ["pass", "pass"], "i6": ["fail", "fail"]}
    for item, vs in judged.items():
        for run, v in enumerate(vs, 1):
            rows.append({"item": item, "rater": "judge", "verdict": v, "run": run})
    return rows


def test_analyze_alignment_and_consistency():
    out = judge.analyze(_rows())
    al = out["alignment"]
    # human majority: i1 p, i2 p, i3 f, i4 f, i5 p, i6 f
    # judge (most common, first on ties): i1 p, i2 f, i3 f, i4 p, i5 p, i6 f
    assert (al["tp"], al["fn"], al["fp"], al["tn"]) == (2, 1, 1, 2)
    assert al["strictness"] == pytest.approx(1 / 3)
    assert al["leniency"] == pytest.approx(1 / 3)
    assert out["consistency"]["flip_rate"] == pytest.approx(1 / 6)
    assert out["consistency"]["flipped"] == ["i3"]
    assert out["human_agreement"]["raters_per_item"] == 3


def test_analyze_uses_heldout_when_marked():
    rows = _rows()
    for r in rows:
        if r["item"] in ("i1", "i3"):
            r["split"] = "heldout"
    al = judge.analyze(rows)["alignment"]
    assert al["scope"] == "heldout" and al["n"] == 2


def test_krippendorff_alpha_matches_the_published_example():
    # Krippendorff, "Computing Krippendorff's Alpha-Reliability": 4 coders, 12 units, missing values; nominal 0.743.
    coders = [[1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
              [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
              [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
              [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]]
    units = [[c[u] for c in coders if c[u] is not None] for u in range(12)]
    assert judge.krippendorff_alpha(units) == pytest.approx(0.743, abs=5e-4)


def test_alpha_uses_items_with_uneven_rater_counts():
    rows = _rows()
    rows += [{"item": "i7", "rater": r, "verdict": "pass"} for r in ("human:a", "human:b")]
    h = judge.analyze(rows)["human_agreement"]
    assert h["raters_per_item"] == 3 and h["items"] == 6          # Fleiss: only the 3-rater items
    assert h["alpha_items"] == 7                                   # alpha: every item with 2+ human labels


def _ceiling_rows(judge_wrong: int, n: int = 40):
    """n items, 3 humans who agree except on every 5th item (2-1 split); the judge gets `judge_wrong` items wrong."""
    rows = []
    for k in range(n):
        item, truth = f"c{k}", ("pass" if k % 2 else "fail")
        other = "fail" if truth == "pass" else "pass"
        labels = [truth, truth, other if k % 5 == 0 else truth]
        rows += [{"item": item, "rater": f"human:{r}", "verdict": v} for r, v in zip("abc", labels, strict=True)]
        rows.append({"item": item, "rater": "judge", "verdict": other if k < judge_wrong else truth})
    return rows


def test_ceiling_judge_as_good_as_people_is_within_noise():
    c = judge.analyze(_ceiling_rows(judge_wrong=0))["ceiling"]
    assert c["n"] == 40 and c["human_human"] == pytest.approx(1 - 8 * (2 / 3) / 40)
    assert c["judge_human"] >= c["human_human"]
    assert c["difference"]["high"] >= 0 and not c["verdict"].startswith("below")


def test_ceiling_a_judge_that_often_disagrees_is_below_human_agreement():
    c = judge.analyze(_ceiling_rows(judge_wrong=16))["ceiling"]
    assert c["difference"]["high"] < 0 and c["verdict"].startswith("below human agreement")


def test_ceiling_needs_two_human_labels():
    rows = [{"item": "x", "rater": "human:a", "verdict": "pass"}, {"item": "x", "rater": "judge", "verdict": "pass"}]
    assert "ceiling" not in judge.analyze(rows)
