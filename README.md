# agentstats

Statistics for agent evals. Point it at your agent's runs (Harbor jobs, ATIF trajectories or a JSONL file) and
it tells you how good the agent is, how reliable it is, and whether version B is really better than A. Every
number comes with a confidence interval, and the report says which claim the numbers support.

```bash
pip install -e .        # not on PyPI yet
agentstats summary jobs/run1 jobs/run2 jobs/run3
agentstats compare --a jobs/baseline --b jobs/candidate
agentstats judge labels.jsonl
agentstats gate --policy agentstats.toml --tier nightly --candidate results/ --baseline baseline/
```

## Why

Agents are stochastic, so a single eval run is a sample, not a measurement. On 100 SWE-bench Verified tasks, the
same agent with the same model and setup (mini-swe-agent, DeepSeek-V4.1-Flash) scored **48%, then 40%**. Eight
tasks flipped between identical runs. If you compare two versions on one run each, that noise is your result.

Three other things go wrong in practice:

- **Infrastructure failures get scored as agent failures.** A quota error or a billing outage looks like a bad
  agent. agentstats keeps them out of pass rates, counts them, and warns if one arm had more.
- **The wrong interval for how the eval set was built.** Most eval sets are hand-picked. Treating each case as an
  independent sample of real traffic gives intervals that are too narrow, and a claim the data can't support.
- **An unchecked LLM judge.** A lenient judge passes behaviour humans would reject, and that's how regressions
  ship.

## What it does differently

Other good tools exist (see [Related tools](#related-tools)). agentstats is built around agent runs:

1. **Reads trajectories, not score tables.** Harbor job directories and ATIF trajectories directly: pass/fail,
   infrastructure errors, tokens, cost, latency, steps and tool calls, plus deterministic rule checks on the
   tool calls (`called`, `not_called`, `before`, `args`).
2. **Picks the statistics from how the eval set was built, and says what claim they support.**

   | `--design` | Your eval set | Interval / comparison | Claim |
   |---|---|---|---|
   | `curated` (default) | hand-picked cases, most eval sets | fixed cases: noise from repeated trials of each case (falls back to Student's t / Welch's t over complete runs when runs vary more than that predicts) | "on this eval set" |
   | `sampled` | drawn at random from real traffic | per-case t / paired per-case differences | "on the population the cases came from" |
   | `clustered` | sampled, in groups that fail together | cluster-robust standard errors | "on the population the groups came from" |

   For a hand-picked set the cases are fixed, so the only randomness is the agent's own: the fixed-cases test
   measures it from repeated trials of each case, and cases that always (or never) pass contribute nothing. It's
   the paired test from a production diagnosis gate, plus a small-sample correction that brought simulated false
   alarms from 6.8% to 5.8% at a nominal 5% (the rest is the discreteness of pass/fail; a simulated threshold
   would close it). It checks for run-wide shocks (whole runs varying more than per-case randomness predicts, as
   when a provider has a bad hour) and then falls back to treating each complete run as one observation. With one
   run it uses a Wilson interval or an exact McNemar test. Overlapping groups can't be modelled credibly, so `clustered` refuses them and explains why.
3. **Reliability as its own result.** pass@k (capability) and pass^k (consistency) with CIs, and lists of
   always-pass, never-pass and flaky cases.
4. **Gates that can say "I can't tell".** Two regression rules: `no_significant_drop` fails only on a proven drop
   (for pull requests); `no_regression` passes only when the change is shown to be no worse than a margin
   (non-inferiority, for releases), so an under-powered run is INCONCLUSIVE, with an estimate of the runs that would
   settle it, never PASS. A run-wide shock with too few runs to correct for it is also INCONCLUSIVE ("rerun").
   Infrastructure failures (Harbor exceptions or an `infra_error` field) are kept out of pass rates and make a run
   UNHEALTHY (exit 3), as eval-sprint also does.
5. **Judge checks:** human agreement (Fleiss' κ), judge vs humans (accuracy, strictness, leniency, Cohen's κ,
   each with a CI, on a held-out split if you mark one) and judge consistency over repeated runs (flip rate, mean
   pairwise disagreement).

## Example: did giving the fixing agent a diagnosis help?

mini-swe-agent on 100 SWE-bench Verified tasks, once plain and once with a diagnosis from a separate diagnosis
agent added to the task as a hint. One Harbor job each.

```
$ agentstats compare --a jobs-plain100/... --b jobs-hint100/... --a-name plain --b-name with-diagnosis

NO DETECTABLE DIFFERENCE: with-diagnosis minus plain = +1.0 pts (95% CI -6.2 pts to +8.2 pts),
on this eval set, from a single run per arm (indicative: run-to-run noise is unmeasured).
Method: exact McNemar test + paired differences. 100 matched cases; plain 48.0%, with-diagnosis 49.0%.

- McNemar (exact): 7 cases pass only with with-diagnosis, 6 only with plain; p = 1.
- Smallest difference this setup can reliably detect (80% power): about +10.2 pts.
- Cost per attempt: plain $0.0858, with-diagnosis $0.0765; difference $-0.0093 (95% CI -0.0182 to -0.0005).
```

The +1 point is noise: 13 tasks moved, in both directions, and this setup can't see anything smaller than about
10 points. The cost difference is real (its CI excludes zero): the hint made the agent about 11% cheaper.

The same plain setup, run twice:

```
$ agentstats summary jobs-plain100/... jobs-all500/...

Pass rate: 44.0% (95% CI 41.2% to 46.8%), on this eval set.
Method: fixed cases, noise from repeated trials per case. 100 cases, 2 run(s).
- Not used: Student's t over complete runs would give 44.0% (95% CI 0.0% to 94.8%) (for hand-picked sets
  with 3+ runs that show run-wide shocks).
- Note: Whole runs vary 8.0x more than per-case randomness predicts (p = 0.005): a run-wide shock. With fewer
  than 3 runs there's nothing safer to fall back to; treat this with care.

| k | pass@k (capability)           | pass^k (consistency)          |
| 1 | 44.0% (95% CI 34.5% to 53.5%) | 44.0% (95% CI 34.5% to 53.5%) |
| 2 | 48.0% (95% CI 38.0% to 58.0%) | 40.0% (95% CI 30.0% to 50.0%) |
Always pass: 40, never pass: 52, flaky: 8
```

48% and 40% is more than per-case randomness explains: something changed across the whole run between the two
(they ran at different times, in jobs of different sizes). A narrow interval from these two runs would be false
confidence, and the report says so. A third run lets it fall back to run-level statistics.

## In production: an incident-triage agent's CI gate

agentstats runs the regression gate for a production incident-triage agent (classifies alerts as real / noise /
duplicate with a P0-P3 severity): 114 held-out cases, 2 trials per pull request, compared case by case with a 4-trial
baseline using the fixed-cases test, `no_significant_drop` at a 90% interval.

**Sized by simulation from the baseline's own per-case rates** (each case's true rate drawn from its baseline trials,
then candidate and baseline simulated from it; `agentstats size` does the same for your baseline), against the gate it
replaced (fail if more than 6% of 80 filtered cases miss, one run):

| True drop | agentstats gate catches it | Old threshold gate |
|---|---|---|
| none (false alarm) | 5% | ~0% |
| ~5 points | 64% | 48% |
| ~8 points | 94% | 82% |
| ~12 points | 100% | 97% |

The old gate also had to drop 32 noisy cases to get under its threshold; the paired test keeps them, since each case is
compared with its own baseline rate.

**What it caught in its first week:**
- **An environment offset.** Six CI trials of unchanged code averaged 89.9% against a baseline measured on a laptop
  (93.2%): -3.3 points, 95% CI -6.1 to -0.5, spread over many cases, no run-wide shock. One pull request had already
  failed for it. The baseline is now measured in CI; the first gate run against it: -0.7 points (CI -3.4 to +2.1), PASS.
- **Outages, as UNHEALTHY instead of FAIL.** Once when the API credit ran out mid-run (69 of 228 attempts failed:
  exit 3, the cause listed, the remaining attempts compared separately), once when a new major SDK release rejected a
  request parameter on every call.
- **A doubtful "same setup".** Two runs of the same coding agent on the same 100 tasks scored 48% and 40%: the shock
  check flagged it (whole runs varying 8x more than per-case randomness predicts, p = 0.005), so the narrow interval
  wasn't trusted (see the example above).



## Evals in CI, in tiers

`agentstats init-ci` writes a policy (`agentstats.toml`) and three GitHub workflows. You replace one step in
each, running your agent, and agentstats decides:

| Tier | When | What runs | Typical requirements |
|---|---|---|---|
| smoke | every pull request, minutes | a stratified sample (`agentstats sample`) + all rule checks | `rules`, `min_pass_rate` (a crash threshold) |
| nightly | every night | the whole eval set, k runs per case | `no_regression` vs the baseline, `max_infra` |
| release | before shipping | the whole set against the production baseline | `no_regression`, `max_cost_ratio`, `max_p95_latency_ratio`, `min_pass_hat_k` |

`agentstats gate --policy agentstats.toml --tier nightly --candidate results/ --baseline baseline/` exits
**0 pass, 1 fail, 2 inconclusive, 3 unhealthy** (too many infrastructure failures to judge the agent), and writes
its table to the GitHub job summary.

Comparisons with the baseline are **non-inferiority tests**. "The interval includes zero" doesn't show that
nothing regressed, so `no_regression = 0.02` passes only when the lower confidence bound of (candidate − baseline)
is at least −2 points. An under-powered run is INCONCLUSIVE, with an estimate of how many runs would settle it,
never PASS. Small smoke tiers say what they can't see: "with 20 cases and 1 run, this tier can only see drops
larger than about 40 points: it catches breakage, not regressions."

On the real runs above (with-diagnosis as the candidate, plain as the baseline):

```
## agentstats gate: nightly ⚠️ INCONCLUSIVE

| requirement | measured | threshold | result | detail |
| runs | 1 | >= 3 | ⚠️ INCONCLUSIVE | the nightly tier expects 3 complete runs |
| max_infra | 0.0% | <= 5.0% | ✅ PASS | |
| no_regression | +1.0 pts (CI -6.2 pts to +8.2 pts, mcnemar) | lower bound >= -2.0 pts | ⚠️ INCONCLUSIVE | about 6 runs per arm would settle it if the estimate holds |
| max_cost_ratio | 0.89x (CI 0.80x to 0.99x) | <= 1.10x | ✅ PASS | |
```

## Input

One row per attempt (`agentstats convert` writes this from Harbor jobs):

```json
{"arm": "baseline", "run": 1, "case": "django-11099", "passed": true, "infra_error": null,
 "cost_usd": 0.12, "input_tokens": 41000, "output_tokens": 2100, "steps": 14, "tool_calls": 9,
 "latency_s": 33.0, "tags": ["feature:search"], "checks": {"no_refund_before_lookup": true}}
```

A *run* is one full pass over the eval set. Several paths for one arm are several runs. Runs are compared on the
cases attempted in every run.

Judge labels (`agentstats judge`), one row per verdict:

```json
{"item": "conv-07", "rater": "human:alice", "verdict": "pass", "split": "heldout"}
{"item": "conv-07", "rater": "judge", "verdict": "fail", "run": 2}
```

Trajectory rules (`agentstats check JOB --rules rules.json --out results.jsonl`): see `examples/rules.json`.
A synthetic judge-labels file is in `examples/judge_labels.jsonl`.

## Methods and sources

- Error bars for evals, clustered standard errors: Miller 2024, [arXiv 2411.00640](https://arxiv.org/abs/2411.00640).
- Choosing the method by how the eval set was built, evals in CI tiers, judge alignment and consistency metrics:
  Yesha Shah's "Agentic Evals" series (see [Credits](#credits)).
- pass@k: Chen et al. 2021, [arXiv 2107.03374](https://arxiv.org/abs/2107.03374). pass^k: τ-bench, Yao et al.,
  [arXiv 2406.12045](https://arxiv.org/abs/2406.12045). Randomness in agentic evals: Bjarnason et al. 2026,
  [arXiv 2602.07150](https://arxiv.org/abs/2602.07150).
- Judge self-inconsistency: Haldar & Hockenmaier 2025 (EMNLP Findings); Yagubyan 2026,
  [arXiv 2606.13685](https://arxiv.org/abs/2606.13685).

Every statistic is tested against scipy or statsmodels (`tests/test_stats.py`, `tests/test_judge.py`). The only
runtime dependency is scipy.

## Credits

The framing behind several features comes from Yesha Shah's **Agentic Evals** series (Google Cloud Agent Quality;
Part 1, "A Software Engineer's Guide", 2026-09-21; Part 2, "Statistical Tools for Measuring Agent Performance",
2026-09-30; Part 3, "Introduction to Autoraters", 2026-10-19):

- choosing the statistics by how the eval set was built (benchmark vs generalised performance; curated, sampled and
  correlated eval sets);
- pass@k and pass^k for agents, and reporting intervals rather than single runs;
- evals in CI in tiers (smoke, nightly, release), a stratified smoke sample, and rule-based checks on trajectories;
- the judge checks: strictness, leniency, Cohen's and Fleiss' κ, a held-out split, verdict flip rate and mean pairwise
  disagreement.

The statistical methods themselves come from the papers listed under [Methods and sources](#methods-and-sources),
which the series also cites.

Original to agentstats: the fixed-cases test with its small-sample correction (from a production diagnosis
regression gate) and the run-wide shock check; the gate rules (`no_significant_drop` and non-inferiority, INCONCLUSIVE
with a runs estimate, a rerun on a shock, UNHEALTHY on infrastructure failures); reading Harbor jobs and ATIF
trajectories; and the production results above.

## Related tools

- [evalstats](https://github.com/ianarawjo/evalstats) ([arXiv 2609.35815](https://arxiv.org/abs/2609.35815)):
  judge-corrected tests (PPI) and methods tuned for small samples. Use it when your verdicts come from an LLM
  judge with a small human-labelled set.
- [evalci](https://github.com/Shreyaskc/evalci) ([arXiv 2607.04429](https://arxiv.org/abs/2607.04429)): CIs,
  paired tests, power analysis and multiple-comparison correction for per-item tables (lm-eval, HELM).
- [eval-sprint](https://github.com/renanpgonzalez/eval-sprint): an eval kit with deterministic and LLM judges, a
  labelling UI, and a CI regression gate (exact McNemar on paired flips, errors kept out of pass rates, an
  unhealthy-run exit code). Its judge calibration (blind expert labels, Horvitz-Thompson weights, Rogan-Gladen
  correction) is more thorough than agentstats' judge check.

## Limitations

- Pass rates need pass/fail verdicts; numeric scores are used for efficiency metrics and judge consistency.
- No power analysis beyond the "smallest detectable difference" estimate.
- The judge check uses the judge's most common verdict per item; it does not correct downstream statistics for
  judge error (evalstats does).

MIT licence.
