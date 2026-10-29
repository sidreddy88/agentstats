# Example: a production model switch, pre-registered (triage agent, Claude Haiku 4.5 -> Haiku 5.5)

A real decision for the incident-triage agent in my agent platform, made on 2026-10-26 and now in production.

- [preregistration.md](preregistration.md): the rule, written and committed before either model was run.
- [rows/](rows/): the attempt rows (114 held-out cases x 4 trials per model, both models in the same session).
- [decision/decision.md](decision/decision.md): the decision record. **SWITCH**: +2.2 pts (95% CI -0.7 to +5.1),
  cost per accepted result $0.0071 -> $0.0006 (0.09x).
- [analysis.md](analysis.md): checks done afterwards (weak slices, parse failures), reported but not used to change the
  decision.

These files were made with stepdown 0.1.0, a separate package at the time; its commands are now `agentstats
preregister` and `agentstats switch`. The files are kept exactly as recorded, so they still say "stepdown" and show the
paths they were made from.

Reproduce the decision from the rows:

```bash
agentstats switch --current rows/current.jsonl --candidate rows/candidate.jsonl \
    --margin 0.03 --provenance current-model --slice-by template --preregistered \
    --name triage-haiku45-to-haiku55 --out /tmp/triage-switch
```

Why "current-model" provenance: the eval cases were generated and filtered with Haiku 4.5, so the comparison is tilted
toward it. An earlier comparison on the original labels said Haiku 5.5 was about 10 points worse; those labels repeated
occurrence counts that Haiku 4.5 had made up after skipping a required lookup. The labels used here accept either
answer where the triage guide doesn't settle the case.
