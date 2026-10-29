# Pre-registration: move agent-platform's TriageAgent from Claude Haiku 4.5 to Claude Haiku 5.5

Written and committed **before any run** for this decision (the commit timestamp is the record). Nothing below changes
after results are seen; deviations get listed in the decision record, not edited here.

## Question
Is TriageAgent on Haiku 5.5 no more than 3 points worse than on Haiku 4.5, at no higher cost?

## Arms (same code, same commit of agent-platform; only the model setting differs)
- **current:** `claude-haiku-4-5-20251001`, temperature 0 (production today).
- **candidate:** `claude-haiku-5-5`, no temperature (the model rejects it), default effort. The harness (prompt,
  tools, settings) is unchanged: no adaptation step.

## Cases and labels
- The 114 held-out cases in `app/evals/triage_heldout.jsonl`.
- Labels relaxed by rules A and B exactly as written in `stepdown/results/2026-10-25-relaxed-labels/README.md`
  (commit 99a6e07): 18 cases accept either answer. The label file's sha256 is recorded with the results.
- Provenance: `current-model` (the cases were generated and filtered with Haiku 4.5), so the comparison is tilted toward
  the current arm.

## Runs
- 4 trials per case per arm, both arms in the **same local session** (run concurrently on one machine), via
  `scripts/triage_gate.py run`. Infrastructure failures (API errors) are kept apart and capped at 5%.
- Cost per attempt from token counts at list prices in `config/llm_routing.json` (input and output only; prompt-cache
  discounts are not counted, which overstates Haiku 5.5's cost, i.e. is conservative for the candidate).

## Rule
`stepdown decide --margin 0.03 --conf 0.95 --max-cost-ratio 1.0 --provenance current-model --preregistered`
(agentstats fixed-cases test on the curated set):
- **SWITCH** if the lower 95% bound of (candidate - current) is at least -3 points and the cost ratio's upper bound is at
  most 1.0;
- **KEEP** if the upper bound is below -3 points, or it costs more;
- **NOT YET** otherwise. A NOT YET is reported as is; no reruns to get a different answer.
- Slices by error template are reported, not enforced (`--slice-by template`).

## What a SWITCH leads to
An agent-platform PR that sets the triage model to Haiku 5.5, which then goes through the regular CI triage gate
against a refreshed baseline.
