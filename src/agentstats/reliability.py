"""
Capability and consistency from repeated attempts per case.

    pass@k  P(at least one of k attempts passes): capability ([Chen et al. 2021], arXiv 2107.03374).
    pass^k  P(all k attempts pass): consistency, what a user relying on the agent sees (tau-bench, Yao et al.,
            arXiv 2406.12045).

Unbiased per-case estimates from n attempts with c passes, averaged over cases:
    pass@k = 1 - C(n - c, k) / C(n, k)        pass^k = C(c, k) / C(n, k)

A large pass@k - pass@1 means retries help; a large pass@1 - pass^k means success depends on luck.
CIs come from a bootstrap over cases. Use a deterministic grader here: an LLM judge adds its own randomness.
"""
from __future__ import annotations

from math import comb

from agentstats.intervals import Interval, bootstrap, mean


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        raise ValueError(f"need n >= k attempts (n={n}, k={k})")
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    if n < k:
        raise ValueError(f"need n >= k attempts (n={n}, k={k})")
    return comb(c, k) / comb(n, k)


def summarize(counts: dict[str, tuple[int, int]], conf: float = 0.95, seed: int = 0) -> dict:
    """counts: case -> (attempts n, passes c). pass@k and pass^k for every k up to the smallest n."""
    if not counts:
        return {"k_max": 0, "pass_at": {}, "pass_hat": {}, "cases": {}}
    k_max = min(n for n, _ in counts.values())
    units = list(counts.values())
    out: dict = {"k_max": k_max, "pass_at": {}, "pass_hat": {}}
    for k in range(1, k_max + 1):
        out["pass_at"][k] = bootstrap(units, lambda us, k=k: mean([pass_at_k(n, c, k) for n, c in us]),
                                      conf, seed=seed, method="bootstrap over cases").to_json()
        out["pass_hat"][k] = bootstrap(units, lambda us, k=k: mean([pass_hat_k(n, c, k) for n, c in us]),
                                       conf, seed=seed, method="bootstrap over cases").to_json()
    out["cases"] = {
        "always": sorted(c for c, (n, p) in counts.items() if p == n),
        "never": sorted(c for c, (n, p) in counts.items() if p == 0),
        "flaky": sorted(c for c, (n, p) in counts.items() if 0 < p < n),
    }
    return out


def flake_rate(counts: dict[str, tuple[int, int]]) -> Interval | None:
    multi = [(n, c) for n, c in counts.values() if n >= 2]
    if not multi:
        return None
    return bootstrap(multi, lambda us: mean([1.0 if 0 < c < n else 0.0 for n, c in us]),
                     method="bootstrap over cases")


def suite_health(counts: dict[str, tuple[int, int]]) -> dict | None:
    """What kind of eval set this is, from how its cases score. A regression suite (nearly everything passes) guards
    against breakage but can't show an improvement; a frontier suite (30-70%) can move both ways. A frontier suite
    nobody refreshes turns into a regression suite as the agent improves. counts: case -> (attempts, passes)."""
    if not counts:
        return None
    n_cases = len(counts)
    rate = mean([c / n for n, c in counts.values()])
    always = sum(1 for n, c in counts.values() if c == n) / n_cases
    never = sum(1 for n, c in counts.values() if c == 0) / n_cases
    if rate >= 0.9 or always >= 0.8:
        kind = "regression"
        advice = ("most cases pass: it guards against breakage but can't show an improvement; to measure progress, "
                  "keep a separate frontier set the agent passes 30-70% of the time")
    elif 0.3 <= rate <= 0.7:
        kind = "frontier"
        advice = "room to move both ways: an improvement or a regression can show"
    elif rate < 0.3 or never >= 0.5:
        kind = "too_hard"
        advice = ("most cases fail: check the never-pass cases for broken tasks or graders before reading anything "
                  "into the rate")
    else:
        kind = "between"
        advice = "between a regression and a frontier set: fine for gating, weak for showing gains"
    return {"kind": kind, "pass_rate": rate, "always_share": always, "never_share": never, "cases": n_cases,
            "advice": advice}
