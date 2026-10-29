# Triage Haiku 4.5 -> 5.5: result and checks after the decision

**Decision (pre-registered rule, run as written): SWITCH.** +2.2 pts (95% CI -0.7 to +5.1), cost 0.09x (CI 0.09x to
0.10x). Current 90.8% / $0.0071 per accepted result; candidate 93.0% / $0.0006. `decision/decision.md`.
No deviations from `preregistration.md`; no reruns.

## Checked afterwards (reported, not used to change the decision)

- **HEALTH_CHECK_TIMEOUT slice: KEEP** (-41.7 pts, CI -70.3 to -13.0; candidate 7/24, current 17/24). All 6 cases'
  descriptions say the timeout was "expected, routine transient network jitter -- not a real problem, no action
  needed, false alarm"; label noise P3. Haiku 5.5 answered real P3 in 17 of 24. Its reasoning (2 extra runs, <$0.01):
  "the reporter's noise claim is unverified, so it is classified real at low severity pending confirmation". So it
  doesn't stand down on a claim made inside the alert text, which the prompt's "use 'real' if the error is happening
  and no open PR was found" also pushes toward, and which is the safer behaviour if log text can be attacker-influenced.
  Real CloudWatch text wouldn't editorialise like these synthetic descriptions. Cost of the behaviour: some false
  alarms become P3 investigations; no incident is missed. A prompt line can tune it if it matters.
- **Parse failures:** 2 per arm ("non-JSON answer" on JSON-looking answers), so not model-specific. A small parser fix
  for later.
- **Other slices:** S3_NO_SUCH_KEY -6.2 (CI -11.4 to -1.1, NOT YET), ECS_ERROR -1.8 (CI -5.3 to +1.7); gains on
  HEAP_OUT_OF_MEMORY +18.2, MONGO_CONNECTION_REFUSED +66.7, PROCESS_KILLED +15.0.
- The eval set still favours Haiku 4.5 (generated and filtered with it), so the bias runs against this SWITCH.

## Next
agent-platform PR A (labels that accept either answer, scorer, routed model, cost in the gate; behaviour unchanged),
then a CI baseline refresh, then PR B (routing.triage.model -> claude-haiku-5-5), gated against the refreshed baseline.
