# Changelog

## 0.5.0 (2026-10-31)

**Suggestions**
- `agentstats suggest` / `suggest-apply`: per-agent behaviour rules mined from trajectories with outcome evidence
  (gate when breaking it is proven to hurt, monitor otherwise), no-answer label advice from repeated template endings,
  advice for agents with few or no graded runs; accepted rules are written into the agent's config, nothing else is.

## 0.4.0 (2026-10-30)

**Statistics**
- The sampled design no longer crashes on rules-only rows (no verdicts at all); it reports the rule checks like the
  curated design does. Found by agent-platform's weekly monitoring job.
- Few failures: when either outcome is rare (under 10 failures, or passes, across both arms) the fixed-cases test
  falls back to Newcombe's hybrid score interval on the pooled counts. Before, no failures at all gave a zero-width
  interval (+0.0 to +0.0), so `switch` could say SWITCH on 120 vs 120 passes and rule comparisons looked exact. In
  simulation (40 cases x 3 trials, base 0.99 vs candidate 0.95) coverage of the true difference rose from about 92%
  to 97%. Found by the Acme example's model switch, which now says NOT YET.

**Example**
- `examples/acme/`: "a week at Acme", two real agents (a tool-calling support agent with OpenTelemetry traces, an
  Inspect AI coding agent) through the whole back half: smoke and nightly gates, behaviour rules on production traces,
  an unavailable model reading as UNHEALTHY, the rubric judge, sampled human review, promote, and a model switch.
  Replays from recorded runs with no API keys (`bash week.sh`). The README records what went wrong while building it.

**Inputs**
- Inspect AI eval logs (`.eval` and `.json`): read directly by every command, or `agentstats convert --format inspect`
  with `--scorer`, `--pass-threshold`, `--errors {infra,fail}`, `--rules`, `--arm`. One row per sample and epoch;
  sample errors are infrastructure failures by default; limits and N scores become `no_answer`; tool calls feed rule
  checks; metadata becomes tags. Uses inspect-ai's reader when installed (no new dependency). A log where no sample
  was scored (e.g. every sample errored) reads as infrastructure failures instead of failing on scorer choice.

**Eval sets**
- `agentstats promote find` / `promote accept`: failures (failed verdicts, or rule violations on rules-only production
  rows) grouped by signature and deduplicated into candidate cases with provenance tags (source, found, label);
  accepting appends approved candidates to the cases file and prints the provenance mix, warning when one model
  labelled most of the eval set.

**Judges**
- `agentstats grade`: an LLM judge on binary rubric questions over any OpenAI-compatible endpoint (standard library,
  own User-Agent: Together's edge returns 403 to Python's default one). Writes judge label rows (each repeat, for
  consistency) and, with `--rows`, attempt rows whose checks are the per-question majority. Unparsed replies are `na`,
  transport failures infrastructure errors. Warns when a question's name is negated (`no_...`) and its passing answer
  is "no": on a real run such a double negative made the judge flip between yes and no on the same reply.
- `agentstats review sample` / `review import`: sampled human review of graded attempts. The sample favours attempts
  where the verdict and the behaviour rules disagree, then failures, then passes, spread over a tag (`--stratify-by`);
  the sheet (CSV) is blind, with verdicts, strata and sampling weights in a separate key. Import writes label rows for
  `agentstats judge` and reports agreement, leniency and strictness on the sample and reweighted to the population
  (stratified bootstrap 95% intervals).

## 0.3.0 (2026-10-29)

First release on PyPI.

**Model switches** (moved in from the stepdown package)
- `agentstats preregister`: write the switch rule (margin, confidence, cost ratio, design, provenance, slices, input
  hashes) to `prereg.json` + `prereg.md` before any result exists.
- `agentstats switch`: SWITCH / KEEP / NOT YET for moving to a cheaper model (non-inferiority at a margin + cost
  ratio, as-is and adapted steps, slices, provenance warning), with a decision record; with `--prereg` the rule comes
  from the registration and is verified (committed, unchanged, before the results, inputs unchanged).
- Example: `examples/switch-triage-haiku-5-5/` (a production decision).

**Gates**
- `no_answer` outcome: attempts with no usable answer (step limit, unparseable output) count as failures but are
  reported apart in `summary` and the gate; new requirement `max_no_answer`. A gate without it still notes a jump.
- Per-slice comparisons: `slice_by` in a tier prints candidate vs baseline per slice (fixed-cases, Bonferroni-adjusted
  across slices, intervals clipped to the possible range); `max_slice_drop` fails on a proven slice drop and names
  large unproven drops as leads.

- Rule gates: `min_rule_rate` (each trajectory rule holds at least this often, interval-based) and
  `no_rule_regression` (each rule vs the baseline, fixed-cases, Bonferroni across rules). Behaviour checks that need
  no labels; the gate prints a per-rule table.

**Inputs**
- `agentstats convert-traces`: OpenTelemetry (OTLP JSON, GenAI conventions) and Langfuse trace exports to rows; tool
  calls become rule checks; `--verdicts` joins pass/fail by trace id, otherwise rows are rules-only and the rule gates
  (and `summary`'s rule table) still work, while pass-rate requirements report INCONCLUSIVE instead of UNHEALTHY.
- Rows are validated on load with the file, line and field in the error (e.g. `'steps' must be a number, got list`).
- Optional `schema_version` (1); newer versions are refused. Contract tests load rows written by harness-evolve and by
  agent-platform's triage gate.

**Statistics**
- Calibration tests: Wilson coverage, fixed-cases coverage, the drop gate's false-alarm rate (12,000 simulations; the
  6% bound fails if the small-sample correction is removed) and power, and the shock check's false-positive rate.

## 0.2.0 (not published)

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
