# Changelog

## 0.2.0 (unreleased)

**Gates for evals in CI**
- `agentstats.toml` policy with tiers (smoke / PR / nightly / release); `agentstats gate` exits 0 pass, 1 fail,
  2 inconclusive, 3 unhealthy and writes the GitHub job summary.
- Two regression rules: `no_significant_drop` (fails only on a proven drop, for pull requests) and `no_regression`
  (non-inferiority at a margin, for releases: an under-powered run is INCONCLUSIVE, with a runs estimate).
- `agentstats sample`: a deterministic, stratified smoke sample (critical cases always in).
- `agentstats init-ci`: a starter policy and smoke / nightly / release GitHub workflows.
- `agentstats size`: simulate a gate rule on your own baseline (pass / fail / inconclusive rates for a range of true
  drops) to choose trials and margins.

**Statistics**
- Fixed-cases test for hand-picked eval sets, now the default for `curated`: per-case pass rates vs the baseline's,
  noise from repeated trials per case, with a small-sample correction.
- Run-wide shock check (whole runs varying more than per-case randomness predicts): falls back to run-level statistics
  with 3+ runs; with fewer, the gate asks for a rerun.
- Runs are compared on the cases attempted in every run; an infrastructure failure no longer removes its case.
- Pass-rate intervals clipped to the possible range.
- No graded attempts (every attempt an infrastructure failure) is UNHEALTHY, not a crash.

## 0.1.0

First version: Harbor / ATIF / JSONL inputs; pass rate with intervals chosen by eval-set design (curated / sampled /
clustered); pass@k and pass^k; infrastructure failures kept apart; efficiency with CIs; trajectory rule checks; judge
checks (Fleiss / Cohen kappa, strictness, leniency, flip rate); `summary`, `compare`, `judge`, `check`, `convert`.
