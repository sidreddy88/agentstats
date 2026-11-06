"""
Can you trust your autorater (LLM judge)? Three checks.

Input rows: {"item": "...", "rater": "human:alice" | "judge", "verdict": "pass" | "fail" | "na",
             "run": 1, "split": "tune" | "heldout"}

1. Human agreement (are the labels themselves trustworthy?): Krippendorff's alpha over every item with >= 2 human
   labels (any number of raters per item), and Fleiss' kappa over the items sharing the most common rater count.
2. Judge vs humans (alignment): judge verdict vs the human majority. accuracy, strictness (false fails among
   human passes), leniency (false passes among human fails: the dangerous one when a pass justifies a release),
   Cohen's kappa. Reported on the held-out split when one is marked, so tuning the judge doesn't flatter it.
3. Judge consistency (is the judge itself stable?): over repeated judge runs on the same item, verdict flip rate
   and mean pairwise disagreement (or mean pairwise |delta| for numeric scores).
4. Judge vs the human ceiling: on items with >= 2 human labels, how often the judge agrees with each person versus
   how often people agree with each other, and the difference with an item-bootstrap interval. A judge can't be
   expected to beat human agreement; "within noise of it" is the practical target for subjective criteria.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from agentstats.intervals import bootstrap, wilson


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _norm(v) -> str | float:
    if isinstance(v, bool):
        return "pass" if v else "fail"
    if isinstance(v, (int, float)):
        return float(v)
    return str(v).strip().lower()


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(table: list[list[int]]) -> float:
    """table[i][j]: how many raters put item i in category j. Every item needs the same number of raters."""
    n_items = len(table)
    if n_items == 0:
        return float("nan")
    m = sum(table[0])
    if m < 2 or any(sum(r) != m for r in table):
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of raters on every item")
    p_items = [(sum(c * c for c in r) - m) / (m * (m - 1)) for r in table]
    p_bar = sum(p_items) / n_items
    totals = [sum(r[j] for r in table) for j in range(len(table[0]))]
    p_cats = [t / (n_items * m) for t in totals]
    pe = sum(p * p for p in p_cats)
    return 1.0 if pe == 1 else (p_bar - pe) / (1 - pe)


def krippendorff_alpha(units: list[list]) -> float:
    """Nominal Krippendorff's alpha. units[i]: the labels item i received (any number; items with fewer than 2
    are not pairable and are skipped). Unlike Fleiss' kappa it needs no fixed number of raters per item."""
    pairable = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in pairable)
    if n < 2:
        return float("nan")
    totals: Counter = Counter()
    d_obs = 0.0
    for u in pairable:
        counts = Counter(u)
        totals.update(counts)
        m = len(u)
        d_obs += sum(counts[c] * counts[k] for c in counts for k in counts if c != k) / (m - 1)
    d_obs /= n
    d_exp = sum(totals[c] * totals[k] for c in totals for k in totals if c != k) / (n * (n - 1))
    return 1.0 if d_exp == 0 else 1 - d_obs / d_exp


def pairwise_disagreement(verdicts: list) -> float:
    """Probability two distinct verdicts on the same item differ (categorical). For pass/fail with s passes
    out of n this is 2 s (n - s) / (n (n - 1))."""
    n = len(verdicts)
    if n < 2:
        return float("nan")
    same = sum(c * (c - 1) for c in Counter(verdicts).values())
    return 1 - same / (n * (n - 1))


def pairwise_delta(scores: list[float]) -> float:
    n = len(scores)
    if n < 2:
        return float("nan")
    total = sum(abs(scores[i] - scores[j]) for i in range(n) for j in range(i + 1, n))
    return 2 * total / (n * (n - 1))


def analyze(rows: list[dict], positive: str = "pass") -> dict:
    humans: dict[str, dict[str, str]] = defaultdict(dict)
    judge: dict[str, list] = defaultdict(list)
    split: dict[str, str] = {}
    for r in rows:
        item, rater, v = str(r["item"]), str(r["rater"]), _norm(r["verdict"])
        if r.get("split"):
            split[item] = str(r["split"])
        if rater.startswith("human"):
            humans[item][rater] = v
        else:
            judge[item].append(v)
    out: dict = {"items": len(set(humans) | set(judge))}

    # 1. Human agreement
    multi = {i: list(h.values()) for i, h in humans.items() if len(h) >= 2}
    if multi:
        cats = sorted({v for vs in multi.values() for v in vs}, key=str)
        sizes = Counter(len(vs) for vs in multi.values())
        m = sizes.most_common(1)[0][0]
        rows_same = [vs for vs in multi.values() if len(vs) == m]
        table = [[vs.count(c) for c in cats] for vs in rows_same]
        out["human_agreement"] = {
            "items": len(rows_same), "raters_per_item": m, "fleiss_kappa": fleiss_kappa(table),
            "unanimous": sum(1 for vs in rows_same if len(set(vs)) == 1),
            "alpha_items": len(multi), "krippendorff_alpha": krippendorff_alpha(list(multi.values())),
        }

    # Human label per item: majority; ties dropped.
    labels = {}
    ties = 0
    for i, h in humans.items():
        top = Counter(h.values()).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ties += 1
            continue
        labels[i] = top[0][0]

    # 2. Judge vs humans (first judge run per item, or its majority over runs)
    def judge_label(vs: list):
        top = Counter(vs).most_common()
        return top[0][0]

    eval_items = [i for i in labels if i in judge and labels[i] != "na" and judge_label(judge[i]) != "na"]
    heldout = [i for i in eval_items if split.get(i) == "heldout"]
    scope = heldout if heldout else eval_items
    if scope:
        h = [labels[i] for i in scope]
        j = [judge_label(judge[i]) for i in scope]
        tp = sum(1 for x, y in zip(h, j, strict=True) if x == positive and y == positive)
        fn = sum(1 for x, y in zip(h, j, strict=True) if x == positive and y != positive)
        fp = sum(1 for x, y in zip(h, j, strict=True) if x != positive and y == positive)
        tn = len(scope) - tp - fn - fp
        out["alignment"] = {
            "scope": "heldout" if heldout else "all labelled items", "n": len(scope), "human_ties_dropped": ties,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": (tp + tn) / len(scope),
            "strictness": fn / (tp + fn) if tp + fn else None,
            "leniency": fp / (tn + fp) if tn + fp else None,
            "cohen_kappa": cohen_kappa(h, j),
            "ci": {"accuracy": wilson(tp + tn, len(scope)).to_json(),
                   "strictness": wilson(fn, tp + fn).to_json() if tp + fn else None,
                   "leniency": wilson(fp, tn + fp).to_json() if tn + fp else None},
            "disagreements": [{"item": i, "human": labels[i], "judge": judge_label(judge[i])}
                              for i in scope if labels[i] != judge_label(judge[i])],
        }

    # 4. Judge vs the human ceiling
    ceiling_items = [i for i in multi if i in judge and judge_label(judge[i]) != "na"
                     and all(v != "na" for v in multi[i])]
    ceil_heldout = [i for i in ceiling_items if split.get(i) == "heldout"]
    ceil_scope = ceil_heldout if ceil_heldout else ceiling_items
    if ceil_scope:
        units = []
        for i in ceil_scope:
            hs, jl = multi[i], judge_label(judge[i])
            pairs = len(hs) * (len(hs) - 1) / 2
            agree_hh = sum(1 for a in range(len(hs)) for b in range(a + 1, len(hs)) if hs[a] == hs[b]) / pairs
            agree_jh = sum(1 for v in hs if v == jl) / len(hs)
            units.append((agree_hh, agree_jh))
        diff = bootstrap(units, lambda us: sum(j - h for h, j in us) / len(us))
        if diff.high < 0:
            verdict = "below human agreement: the judge disagrees with people more than they disagree with each other"
        elif diff.low > 0:
            verdict = "above human agreement (unusual: check that the judge isn't agreeing with one rater's habits)"
        else:
            verdict = "within noise of human agreement"
        out["ceiling"] = {
            "scope": "heldout" if ceil_heldout else "all items with 2+ human labels", "n": len(ceil_scope),
            "human_human": sum(h for h, _ in units) / len(units),
            "judge_human": sum(j for _, j in units) / len(units),
            "difference": diff.to_json(), "verdict": verdict,
            "note": "few items: the interval is wide" if len(ceil_scope) < 30 else "",
        }

    # 3. Judge consistency
    repeated = {i: vs for i, vs in judge.items() if len(vs) >= 2}
    if repeated:
        numeric = all(isinstance(v, float) for vs in repeated.values() for v in vs)
        cons = {"items": len(repeated), "runs_per_item": min(len(vs) for vs in repeated.values()),
                "flip_rate": sum(1 for vs in repeated.values() if len(set(vs)) > 1) / len(repeated),
                "flipped": sorted(i for i, vs in repeated.items() if len(set(vs)) > 1)}
        if numeric:
            cons["mean_pairwise_delta"] = sum(pairwise_delta(vs) for vs in repeated.values()) / len(repeated)
        else:
            cons["mean_pairwise_disagreement"] = (sum(pairwise_disagreement(vs) for vs in repeated.values())
                                                  / len(repeated))
        out["consistency"] = cons
    return out
