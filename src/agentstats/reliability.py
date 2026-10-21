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
