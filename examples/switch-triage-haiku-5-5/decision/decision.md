# stepdown decision: triage-haiku45-to-haiku55

**SWITCH**: as-is: shown no more than 3 points worse: +2.2 pts (CI -0.7 pts to +5.1 pts, fixed_cases); cost 0.09x (CI 0.09x to 0.10x)

Rule: switch only if shown no more than 3 points worse (95% interval, curated eval set) at a cost ratio of at most 1x. Margin declared before the results: yes. Eval-set provenance: current-model.

| step | decision | change (pts) | 95% CI | cost ratio |
|---|---|---|---|---|
| as-is | SWITCH | +2.2 | -0.7 to +5.1 | 0.09x (CI 0.09x to 0.10x) |
| &nbsp;&nbsp;as-is [template:APP_CRASHED] | SWITCH | +0.0 | +0.0 to +0.0 | 0.10x (CI 0.09x to 0.10x) |
| &nbsp;&nbsp;as-is [template:AXIOSERROR] | SWITCH | +0.0 | +0.0 to +0.0 | 0.08x (CI 0.08x to 0.09x) |
| &nbsp;&nbsp;as-is [template:CASTERROR] | SWITCH | +0.0 | +0.0 to +0.0 | 0.10x (CI 0.10x to 0.10x) |
| &nbsp;&nbsp;as-is [template:DEPRECATIONWARNING] | NOT YET | +12.5 | -12.0 to +37.0 | 0.09x (CI 0.08x to 0.09x) |
| &nbsp;&nbsp;as-is [template:ECONNREFUSED] | NOT YET | +100.0 | +25.9 to +100.0 | n/a |
| &nbsp;&nbsp;as-is [template:ECS_ERROR] | NOT YET | -1.8 | -5.3 to +1.7 | 0.09x (CI 0.08x to 0.09x) |
| &nbsp;&nbsp;as-is [template:HEALTH_CHECK_TIMEOUT] | KEEP | -41.7 | -70.3 to -13.0 | 0.10x (CI 0.09x to 0.11x) |
| &nbsp;&nbsp;as-is [template:HEAP_OUT_OF_MEMORY] | SWITCH | +18.2 | +6.0 to +30.3 | 0.08x (CI 0.08x to 0.09x) |
| &nbsp;&nbsp;as-is [template:MODULE_NOT_FOUND] | SWITCH | +10.0 | -2.8 to +22.8 | 0.11x (CI 0.09x to 0.12x) |
| &nbsp;&nbsp;as-is [template:MONGO_CONNECTION_REFUSED] | SWITCH | +66.7 | +27.6 to +100.0 | 0.09x (CI 0.07x to 0.11x) |
| &nbsp;&nbsp;as-is [template:NULL_PTR] | SWITCH | +0.0 | +0.0 to +0.0 | 0.08x (CI 0.07x to 0.09x) |
| &nbsp;&nbsp;as-is [template:PROCESS_KILLED] | SWITCH | +15.0 | -2.0 to +32.0 | 0.11x (CI 0.10x to 0.11x) |
| &nbsp;&nbsp;as-is [template:S3_NO_SUCH_KEY] | NOT YET | -6.2 | -11.4 to -1.1 | 0.09x (CI 0.08x to 0.10x) |
| &nbsp;&nbsp;as-is [template:SYNTAXERROR] | NOT YET | +0.0 | -7.7 to +7.7 | 0.09x (CI 0.08x to 0.09x) |
| &nbsp;&nbsp;as-is [template:TOKENEXPIREDERROR] | NOT YET | +0.0 | -8.6 to +8.6 | 0.10x (CI 0.09x to 0.12x) |

| arm | attempts | pass rate | $ per attempt | $ per accepted result |
|---|---|---|---|---|
| current | 456 | 90.8% | 0.0065 | 0.0071 |
| candidate | 456 | 93.0% | 0.0006 | 0.0006 |

Warnings:
- slices that don't support the decision (reported, not enforced): as-is [template:DEPRECATIONWARNING], as-is [template:ECONNREFUSED], as-is [template:ECS_ERROR], as-is [template:HEALTH_CHECK_TIMEOUT], as-is [template:S3_NO_SUCH_KEY], as-is [template:SYNTAXERROR], as-is [template:TOKENEXPIREDERROR]
- the eval set was created, filtered or labelled by the current model, so the comparison favours it: a SWITCH still holds, a KEEP or NOT YET may overstate the gap

Note: Pre-registered in preregistration.md (stepdown 48019b0). agent-platform commit 4d60f56 (branch feat/triage-haiku-5-5); labels sha256 3125868d99de2502; both arms same local session, 4 trials x 114 cases.

Inputs:
- current: `~/Production/stepdown/decisions/2026-10-25-triage-haiku-5-5/rows/current.jsonl` (sha256 eaadfcb37f9d6fce)
- candidate: `~/Production/stepdown/decisions/2026-10-25-triage-haiku-5-5/rows/candidate.jsonl` (sha256 863982cc73550280)

Decided 2026-10-26T02:44:15+00:00 with stepdown 0.1.0, agentstats 0.2.0.
