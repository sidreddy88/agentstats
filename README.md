# agentstats

Statistics for agent evals. Point it at your agent's runs (Harbor jobs, ATIF trajectories or a JSONL file) and
it tells you how good the agent is, how reliable it is, and whether version B is really better than A. Every
number comes with a confidence interval, and the report says which claim the numbers support.

```bash
pip install -e .        # not on PyPI yet
agentstats summary jobs/run1 jobs/run2 jobs/run3
agentstats compare --a jobs/baseline --b jobs/candidate
agentstats judge labels.jsonl
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
   | `curated` (default) | hand-picked cases, most eval sets | Student's t over complete runs / Welch's t | "on this eval set" |
   | `sampled` | drawn at random from real traffic | per-case t / paired per-case differences | "on the population the cases came from" |
   | `clustered` | sampled, in groups that fail together | cluster-robust standard errors | "on the population the groups came from" |

   With one run per arm it falls back to a Wilson interval or an exact McNemar test and labels the result
   indicative. Overlapping groups can't be modelled credibly, so `clustered` refuses them and explains why.
3. **Reliability as its own result.** pass@k (capability) and pass^k (consistency) with CIs, and lists of
   always-pass, never-pass and flaky cases.
4. **Infrastructure failures kept apart**, from Harbor exceptions or an `infra_error` field.
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

Pass rate: 44.0% (95% CI 0.0% to 94.8%), on this eval set.
Method: Student's t over complete runs. 100 cases, 2 run(s). Interval clipped to the possible range.
- Not used: t over per-case pass rates would give 44.0% (95% CI 34.5% to 53.5%); it doesn't fit a curated set.
- Note: 2 runs: Student's t with 1 degree(s) of freedom gives a very wide interval; 3+ runs recommended.
- Per run: run 1 48.0%, run 2 40.0%

| k | pass@k (capability)           | pass^k (consistency)          |
| 1 | 44.0% (95% CI 34.5% to 53.5%) | 44.0% (95% CI 34.5% to 53.5%) |
| 2 | 48.0% (95% CI 38.0% to 58.0%) | 40.0% (95% CI 30.0% to 50.0%) |
Always pass: 40, never pass: 52, flaky: 8
```

Two runs of a hand-picked set say almost nothing about the pass rate, and the report says so instead of quoting
the narrower per-case interval that doesn't fit.

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
- Choosing the method by how the eval set was built, pass^k for agents, judge alignment and consistency metrics:
  Yesha Shah's "Agentic Evals" series (2026), Parts 2 and 3.
- pass@k: Chen et al. 2021, [arXiv 2107.03374](https://arxiv.org/abs/2107.03374). pass^k: τ-bench, Yao et al.,
  [arXiv 2406.12045](https://arxiv.org/abs/2406.12045). Randomness in agentic evals: Bjarnason et al. 2026,
  [arXiv 2602.07150](https://arxiv.org/abs/2602.07150).
- Judge self-inconsistency: Haldar & Hockenmaier 2025 (EMNLP Findings); Yagubyan 2026,
  [arXiv 2606.13685](https://arxiv.org/abs/2606.13685).

Every statistic is tested against scipy or statsmodels (`tests/test_stats.py`, `tests/test_judge.py`). The only
runtime dependency is scipy.

## Related tools

- [evalstats](https://github.com/ianarawjo/evalstats) ([arXiv 2609.35815](https://arxiv.org/abs/2609.35815)):
  judge-corrected tests (PPI) and methods tuned for small samples. Use it when your verdicts come from an LLM
  judge with a small human-labelled set.
- [evalci](https://github.com/Shreyaskc/evalci) ([arXiv 2607.04429](https://arxiv.org/abs/2607.04429)): CIs,
  paired tests, power analysis and multiple-comparison correction for per-item tables (lm-eval, HELM).
- [eval-sprint](https://github.com/renanpgonzalez/eval-sprint): an eval kit with deterministic and LLM judges, a
  labelling UI, judge calibration and a CI regression gate.

## Limitations

- Pass rates need pass/fail verdicts; numeric scores are used for efficiency metrics and judge consistency.
- No power analysis beyond the "smallest detectable difference" estimate.
- The judge check uses the judge's most common verdict per item; it does not correct downstream statistics for
  judge error (evalstats does).

MIT licence.
