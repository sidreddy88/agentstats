# agentstats

Statistics for agent evals. Point it at your agent's runs (Harbor jobs, ATIF trajectories or a JSONL file) and
it tells you how good the agent is, how reliable it is, whether version B is really better than A, and whether a
cheaper model is shown to be no worse. Every number comes with a confidence interval, and the report says which
claim the numbers support.

```bash
pip install agentstats        # Python 3.11+
agentstats summary jobs/run1 jobs/run2 jobs/run3
agentstats compare --a jobs/baseline --b jobs/candidate
agentstats review sample rows.jsonl --n 40 --stratify-by intent --out review/wk1   # blind sheet for a person
agentstats grade items.jsonl --rubric rubric.json --model MODEL --base-url URL --repeats 3 --out labels.jsonl
agentstats judge labels.jsonl
agentstats promote find prod_rows.jsonl --source production --existing evals/cases.jsonl --out candidates.jsonl
agentstats gate --policy agentstats.toml --tier nightly --candidate results/ --baseline baseline/
agentstats convert-traces traces.json --format otlp --rules rules.json --out rows.jsonl   # or --format langfuse
agentstats summary logs/my-task.eval                     # Inspect AI logs are read directly
agentstats preregister --name my-switch --margin 0.03 --provenance independent --out decisions/my-switch
agentstats switch --prereg decisions/my-switch/prereg.json --current current.jsonl --candidate cheap.jsonl \
    --out decisions/my-switch
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

1. **Reads trajectories, not score tables.** Harbor job directories, ATIF trajectories, Inspect AI logs and
   OpenTelemetry/Langfuse traces directly: pass/fail,
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
6. **Model switches decided by a rule written down first.** `agentstats switch` answers SWITCH / KEEP / NOT YET for
   moving to a cheaper model: non-inferiority at your margin plus a cost-ratio limit, the model as-is and with an
   adapted harness, per-slice results, and a warning when the eval set came from the current model.
   `agentstats preregister` fixes the rule before any result exists, and the decision only counts as pre-registered
   if the registration was committed before the results and its inputs are unchanged. See
   [Deciding a model switch](#deciding-a-model-switch-agentstats-preregister-and-agentstats-switch).

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

**A skipped tool the labels rewarded.** The triage agent (then on Claude Haiku 4.5) skipped a required lookup tool
and wrote an invented count instead. The eval labels had been built from its own answers, so the accuracy gate couldn't
see it. With behaviour rules, Haiku 4.5 as the candidate against Haiku 5.5: accuracy +0.9 pts (CI -2.1 to +3.9), PASS;
`min_rule_rate` FAIL (lookup called in 25.4% of attempts, CI 21.0% to 30.5%); `no_rule_regression` FAIL (-74.6 pts vs
100%). On 61 real-incident traces exported from Langfuse and read with `convert-traces` (rules-only, no labels): the
lookup was called in 52.5% (CI 40.2% to 64.5%). The rules now run in that agent's CI gate.

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



## Deciding a model switch: `agentstats preregister` and `agentstats switch`

A cheaper model comes out. "No significant difference" on one eval run isn't "no worse": a small run usually just has
a wide interval. `agentstats switch` asks the question the other way round, as a non-inferiority test: is the cheaper
model **shown** to lose no more than your margin, at no higher cost? It answers **SWITCH**, **KEEP** or **NOT YET**
and writes a decision record (`decision.json` + `decision.md`) with the rule, the numbers and hashes of the inputs.

```bash
agentstats preregister --name my-switch --margin 0.03 --provenance independent \
    --input labels=evals/labels.jsonl --slice-by template --out decisions/my-switch
git add decisions/my-switch && git commit -m "pre-register my-switch"     # before running anything
# ... run both models on the same cases, a few trials each, same session ...
agentstats switch --prereg decisions/my-switch/prereg.json \
    --current current.jsonl --candidate cheap.jsonl [--adapted cheap-adapted.jsonl] --out decisions/my-switch
```

| Interval for (candidate - current) | Decision |
|---|---|
| lower bound above -margin, and the cost ratio's upper bound at most `--max-cost-ratio` (default 1.0) | SWITCH |
| upper bound below -margin (worse even at the optimistic end), or more expensive | KEEP |
| anything else | NOT YET |

- **Pre-registration is checked, not trusted.** With `--prereg` the rule comes from the file (contradicting flags are
  an error), and the decision only counts as pre-registered if the registration is committed and unchanged, was
  committed before every result file was last modified, and every declared `--input` still has its registered hash.
- **Two steps:** the cheaper model as-is, then with an adapted harness (`--adapted`, e.g. rows from
  `harness-evolve export`); the better decision wins.
- **Slices** (`--slice-by`) are reported per slice; `--strict-slices` makes the decision no better than the worst one.
- **Provenance:** `--provenance current-model` says the eval set was made or filtered with the current model, and the
  record warns that the comparison favours it.
- **Cost per accepted result** is reported for each arm. Exit codes: 0 SWITCH, 1 KEEP, 2 NOT YET, 3 UNHEALTHY.

**In production:** my triage agent moved from Claude Haiku 4.5 to Haiku 5.5 on a pre-registered decision: +2.2 pts
(95% CI -0.7 to +5.1) at 0.09x the cost per accepted result. An earlier comparison had said 10 points worse; the gap
came from labels that repeated counts the old model made up. Rule, rows and record:
[examples/switch-triage-haiku-5-5](https://github.com/sidreddy88/agentstats/tree/main/examples/switch-triage-haiku-5-5).

These two commands started as a separate package, stepdown. They're a thin layer on agentstats' own tests, so they
live here for now. If the model-switch side grows into its own system (suggesting candidate models, adapting the
harness with harness-evolve, deciding as runs accumulate, shadow runs on live traffic), it may become a separate
package again.

## Evals in CI, in tiers

`agentstats init-ci` writes a policy (`agentstats.toml`) and three GitHub workflows. You replace one step in
each, running your agent, and agentstats decides:

| Tier | When | What runs | Typical requirements |
|---|---|---|---|
| smoke | every pull request, minutes | a stratified sample (`agentstats sample`) + all rule checks | `rules`, `min_pass_rate` (a crash threshold) |
| nightly | every night | the whole eval set, k runs per case | `no_regression` vs the baseline, `max_infra` |
| release | before shipping | the whole set against the production baseline | `no_regression`, `max_cost_ratio`, `max_p95_latency_ratio`, `min_pass_hat_k` |

**Behaviour rules as a regression gate (no labels needed).** `min_rule_rate` requires every trajectory rule (the
`checks` in each row, e.g. `{"called_lookup_tool": true}`) to hold at least that often, judged on its interval: FAIL
only when the whole interval is below, INCONCLUSIVE when it can't tell. `no_rule_regression` compares each rule with
the baseline, case by case (fixed-cases test, Bonferroni across rules), and fails when a rule is proven to hold less
often. These catch what an accuracy gate can't when the labels are wrong: a triage agent that skipped a required
lookup tool half the time and invented the number passed every accuracy check, because the labels had been built
from its own answers. The gate prints a per-rule table either way.

Two more requirements, both from real misses: `max_no_answer` caps the share of attempts with no usable answer, and
`max_slice_drop` (with `slice_by = "<tag prefix>"` in the tier) fails when any slice, e.g. one error template, is
proven worse than the baseline by more than the limit. Slice intervals are Bonferroni-adjusted across slices, slices
whose estimate is past the limit but not proven are named as leads, and with `slice_by` set the per-slice table is
printed even without the requirement.

`agentstats gate --policy agentstats.toml --tier nightly --candidate results/ --baseline baseline/` exits
**0 pass, 1 fail, 2 inconclusive, 3 unhealthy** (too many infrastructure failures to judge the agent), and writes
its table to the GitHub job summary.

**Size a gate before trusting it:** `agentstats size --baseline baseline.jsonl --trials 2 --rule drop` simulates the
rule on your baseline's own per-case rates and prints how often it would pass or fail for a range of true drops, so you
can pick trials and margins (e.g. no change passes >= 90%, an 8-point drop fails >= 90%).

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

`no_answer` (optional, a reason such as `"step_limit"` or `"parse_error"`) marks an attempt where the agent produced
no usable answer at all. It counts as a failure, like a wrong answer, but is reported apart: a jump in no-answer
attempts usually means the harness or the agent loop broke, not that the agent's judgement got worse. Rows are
validated on load (the error names the file, line and field), and an optional `schema_version` (currently 1) lets a
newer format be refused instead of half-read. `harness-evolve export` and agent-platform's triage gate both write this
format; `tests/test_hardening.py` loads samples from each.

**From traces** (`agentstats convert-traces`): an OpenTelemetry OTLP JSON export (tool calls are spans with
`gen_ai.operation.name = "execute_tool"`, name in `gen_ai.tool.name`, arguments in `gen_ai.tool.call.arguments`;
tokens from `gen_ai.usage.*`) or a Langfuse export (traces with their observations; TOOL observations are tool
calls). One trace is one attempt; `--rules` fills its checks. Pass/fail comes from `--verdicts` (JSONL with
`trace_id`, `passed`, optional `case`, `run`, `infra_error`); without it the rows are **rules-only**: no pass rate, but
the rule gates work, which is what production traces without labels need.

**From Inspect AI logs** (`.eval` or `.json`; read directly by every command, or
`agentstats convert --format inspect LOG --rules rules.json --out rows.jsonl`): one row per sample and epoch (case =
sample id, run = epoch, arm = model or `--arm`). Pass/fail comes from the headline scorer or `--scorer`: C passes; I
and P fail; N fails as `no_answer`; numbers pass at `--pass-threshold` (default 1.0). A sample that errored is an
infrastructure failure by default (`--errors fail` counts it as a failure); a sample that hit a limit and didn't pass
is `no_answer: limit_<type>`. Tool calls from the assistant messages fill `--rules` checks; tokens, time, steps and
tool calls become metrics; task, model and scalar sample metadata become tags (so `slice_by = "intent"` works).
`.eval` files are zstd-compressed: they're read with inspect-ai's own reader when it's installed, or the standard
library on Python 3.14+; otherwise `inspect log convert --to json` first.

**LLM judge on yes/no rubric questions** (`agentstats grade items.jsonl --rubric rubric.json --model MODEL
--base-url URL --api-key-env VAR --repeats 3 --out labels.jsonl --rows rows.jsonl`): items are
`{"item", "input", "output", ...}` (every field but ids and tags is shown to the judge, or `--fields`); the rubric is a
list of `{"name", "question", "pass": "yes"|"no"}`, and an item passes when every question gets its passing answer.
Any OpenAI-compatible endpoint, called with the standard library. A reply that isn't the JSON asked for is recorded
as `na`, never a silent pass; transport errors are retried, then recorded as infrastructure errors. Repeats are judge
labels (for `agentstats judge`'s flip rate), never extra attempts; `--rows` writes one attempt row per item with the
per-question majority as checks, so a rubric question can be gated like a behaviour rule.

**Sampled human review** (`agentstats review sample ROWS --n 40 --stratify-by intent --link URL --out DIR`, then
`agentstats review import DIR/review.csv --key DIR/key.jsonl --rater NAME --out labels.jsonl`): picks graded attempts
for a person to check, favouring those where the verdict and the behaviour rules disagree (passed with a rule
violated, or failed with every rule satisfied), then failures, then passes, spread over a tag. The sheet is blind
(no verdicts, no reason for the pick); the key keeps each item's verdict, stratum and sampling weight. Import appends
label rows for `agentstats judge` and reports the grader's agreement, leniency and strictness both on the sample and
**reweighted to the population** (over-sampling disagreements would otherwise make the grader look worse than it is),
with stratified bootstrap intervals.

**Failures become eval cases** (`agentstats promote find ROWS --source production --label-from none --existing
evals/cases.jsonl --group-by intent --out candidates.jsonl`, then `agentstats promote accept candidates.jsonl --into
evals/cases.jsonl`): failures from eval runs, or rule violations on rules-only production rows, are grouped by
signature (the rules that broke, the no-answer reason, optionally a slice tag), so one failure seen thirty times is one
signature with up to `--per-signature` candidate cases (the most frequent first). Cases already in the eval set are
skipped. Each candidate is tagged `source:`, `found:` and `label:` (`human`, `none`, or `model:<name>`); a person sets
`"approved": true` and `accept` appends those to the cases file (the one `agentstats sample` reads) and prints the eval
set's provenance mix, warning when most labels came from one model: comparisons against that model are then tilted
in its favour, because its own answers are the answer key. agentstats keeps only case ids and tags; the inputs and
expected answers go in your runner's dataset.

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
- Pre-registration (fix the analysis before seeing the data): Nosek et al. 2018, "The preregistration revolution",
  [PNAS](https://doi.org/10.1073/pnas.1708274114). The switch rule is a standard non-inferiority test (lower interval
  bound above -margin) on the same fixed-cases interval the gates use.
- Cost per accepted result rather than cost per call: from Grep AI's "The Managed Agent Wars".

Every statistic is tested against scipy or statsmodels (`tests/test_stats.py`, `tests/test_judge.py`). The only
runtime dependency is scipy.

## Credits

The framing behind several features comes from Yesha Shah's **Agentic Evals** series:
[Part 1, "A Software Engineer's Guide"](https://yeshas93.substack.com/p/agentic-evals-a-software-engineers)
(2026-09-21); [Part 2, "Statistical Tools for Measuring Agent Performance"](https://yeshas93.substack.com/p/agentic-evals-statistical-tools-for)
(2026-09-30); [Part 3, "Introduction to Autoraters"](https://substack.com/home/post/p-220996535) (2026-10-19):

- choosing the statistics by how the eval set was built (benchmark vs generalised performance; curated, sampled and
  correlated eval sets);
- pass@k and pass^k for agents, and reporting intervals rather than single runs;
- evals in CI in tiers (smoke, nightly, release), a stratified smoke sample, and rule-based checks on trajectories;
- the judge checks: strictness, leniency, Cohen's and Fleiss' κ, a held-out split, verdict flip rate and mean pairwise
  disagreement.

The statistical methods themselves come from the papers listed under [Methods and sources](#methods-and-sources),
which the series also cites.

Original to agentstats:

- the fixed-cases test with its small-sample correction (from a production diagnosis regression gate), and the
  run-wide shock check;
- the gate rules: `no_significant_drop` and non-inferiority, INCONCLUSIVE with a runs estimate, a rerun on a shock;
- gate sizing by simulation on your own baseline (`agentstats size`);
- **behaviour rules as a regression gate**: `no_rule_regression` compares each trajectory rule with the baseline, case
  by case with intervals, and `min_rule_rate` sets an interval-based floor. They need no labels, which is what caught a
  production bug an accuracy gate passed (see the production section);
- the **no-answer outcome** (a broken loop or unparseable output reported apart from wrong answers) and **per-slice
  gates** with Bonferroni-adjusted intervals and named leads;
- **rules-only rows from traces**: OpenTelemetry (GenAI conventions) and Langfuse exports, so behaviour can be gated on
  production traces that have no labels;
- **model-switch decisions** (`agentstats switch`): non-inferiority plus cost per accepted result, as-is and adapted
  steps, an eval-set provenance warning, and **checked pre-registration** (committed before the results, inputs
  hashed), with a decision record;
- **calibration tests** that fail if the statistics drift (e.g. the drop gate's false-alarm rate);
- reading Harbor jobs and ATIF trajectories, and the production results above.

UNHEALTHY on infrastructure failures and keeping them out of pass rates are also in eval-sprint.

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
- Power analysis is by simulation (`agentstats size`) for the two gate rules; no closed-form sample-size planner.
- The judge check uses the judge's most common verdict per item; it does not correct downstream statistics for
  judge error (evalstats does).
- `agentstats switch` decides on fixed batches: run both models, then decide. Deciding as runs accumulate
  (anytime-valid intervals) isn't built. The margin is your judgement; agentstats makes you state it and checks you
  stated it first. Cost is whatever your rows record (my runs: token counts at list prices, no cache discounts).

MIT licence.
