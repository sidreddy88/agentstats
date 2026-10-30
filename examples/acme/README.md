# A week at Acme: agentstats as the back half of an eval harness

Acme runs two agents:

- a **support agent** (refunds, order status, hand-off to a person) that calls tools and exports OpenTelemetry traces;
- a **coding agent** that fixes small Python bugs, evaluated with **Inspect AI** against hidden tests.

Acme keeps its own runners: a script for support, Inspect for coding, a tracing backend for production. agentstats
doesn't run agents. It reads what the runners produce and does the back half: verdicts, aggregation, the release gate
and the failure loop.

```bash
pip install agentstats
bash week.sh          # replays the whole week from recorded runs: no API keys, no cost
```

## The week

| When | What happens | Step in `week.sh` |
|---|---|---|
| Every PR | Smoke gate: PASS, or INCONCLUSIVE with a run estimate | 1 |
| Nightly | Full gate on both agents; behaviour rules also checked on yesterday's production traces | 2, 2b, 2c |
| Monday | 30 judged conversations sampled for human review; judge agreement updated | 3 |
| Monday | Last week's production failures promoted as candidate cases; a person approves them | 4 |
| Release | Release gate: non-inferiority plus rules plus slices; the decision record goes in the release notes | (tier `release` in `config/agentstats.toml`) |
| New model out | Pre-registered switch decision, so it's not settled by a gut call | 5 |

## What happened this week (real runs)

All numbers below are from real model calls, recorded in `data/` (support and judge: DeepSeek-V4.1-Flash on Together;
the candidate model: Claude Haiku 5.5; coding: Inspect on Together). Recording everything cost about $0.40.

**1. PR #42 "simplify the support prompt" is blocked at smoke.** On 16 sampled cases the pass rate dropped by
31 points (95% CI -58.6 to -3.9): a proven drop, so `no_significant_drop` fails and the PR can't merge.
Lesson from building this: the first policy used a 5-point margin for smoke, and the PR *passed*, because the
interval's upper end (-3.9) didn't prove a drop of more than 5 points. For smoke, Acme now fails on any proven drop
(margin 0). A gate's thresholds need checking too.

**2. Nightly confirms it on all 40 cases x 3 trials:** -18.3 points (CI -24.9 to -11.8), and the refund slice -30.6
(CI -43.9 to -17.3). The behaviour rule (`lookup_before_refund`) held 100% in both versions. The bug we planted (a
prompt without "look up the order first") never happened: the model looks orders up anyway. What did happen: v2
refunded orders that weren't delivered yet (9 of 120), and asked for confirmation or a reason ("Would you like me to
refund the full $45?") instead of refunding (13 of 120). Whether asking first is a failure is a judgement call; in a
one-message eval it leaves the task undone.
The outcome checks caught both; the rule written in advance caught neither. You can't predict every failure mode,
which is what steps 3 and 4 are for.

**2c. A model that isn't available reads as UNHEALTHY, not as a bad model.** An earlier candidate
(`openai/gpt-oss-20b`) turned out not to be served per token: all 120 calls failed with 400/500 errors. agentstats
keeps infrastructure failures out of pass rates, so the gate says UNHEALTHY instead of reporting 0%.

**3. Monday review: the judge is far too lenient.** The rubric judge (`agentstats grade`, `config/rubric.json`: polite,
no extra promises, consistent with the order record) passed 119 of v2's 120 replies. A review sample of 30, filled in
by a reviewer, finds the judge agrees only **79.5%** of the time once reweighted to the population (CI 62.4% to
93.2%), with **96.1% leniency** (CI 89.1% to 97.6%): it passes almost every reply that should fail. The reason is
the rubric: no question asks whether the refund followed policy. Next week's rubric gets one.
*The reviewer here is simulated*: `scripts/simulated_reviewer.py` fills the sheet from the order database's ground
truth, standing in for a person reading each conversation.

Lesson from building this: the first rubric named a question `no_extra_promises` while asking "Does the reply promise
anything extra?" (pass: no). The judge sometimes answered the name instead of the question and flipped between yes
and no on the same reply; a second run with `--repeats 2` showed it. `agentstats grade` now warns about negated names
like that.

**4. Production failures become eval cases.** Last week's 20 production conversations (new situations: discount
requests, angry customers, unknown order numbers, two orders in one message) have no ground truth, only the judge and
the rules. The judge flagged both two-order requests ("Refund both A100 and A101"): nothing like them is in the eval
set. `agentstats promote find` proposes them, tagged `source:production`; once approved, `promote accept` adds them
and reports the eval set's provenance (40 curated, 2 from production; every label from a person).
*The approval is simulated* by `scripts/approve_all.py`.

**5. New model: SWITCH to Claude Haiku 5.5**, at 0.46x the cost per attempt (CI 0.45x to 0.46x), no worse on any
intent. **Read this one with care:** both models passed all 120 attempts, so the interval for the difference is
+0.0 to +0.0, which overstates the certainty. 120 passes don't prove a zero failure rate, so "shown no more than 3
points worse" is stronger than this data supports. The eval set is too easy to tell these two models apart (the coding
agent's 36/36 has the same problem); the promoted two-order cases and harder refunds are the fix. (agentstats doesn't
yet widen the interval when neither arm fails; that's a known limitation.)

## Files

| Path | What |
|---|---|
| `agents/support.py` | the support agent (OpenAI-compatible, standard library): rows, OTLP traces, judge items |
| `agents/coding_task.py` | the coding agent as an Inspect AI task with a hidden-test scorer |
| `config/agentstats.toml` | Acme's tiered gate policy (smoke, nightly, release) |
| `config/rules.json`, `config/rubric.json` | behaviour rule, judge rubric |
| `config/support_cases.jsonl`, `config/production_cases.jsonl`, `config/cases.jsonl` | eval cases, a week of production messages, the eval set's case list with provenance tags |
| `data/support/*` | recorded runs: `main` (current prompt), `pr-v2` (the PR), `haiku-5-5`, `unavailable-model`, `production-week` |
| `data/grade/*` | recorded judge runs (2 repeats each) |
| `data/coding/` | the Inspect log |
| `scripts/` | small helpers: the smoke filter, and the two simulated human steps |

To record fresh runs (needs `TOGETHER_API_KEY`; about $0.40):

```bash
python agents/support.py --prompt v1 --trials 3 --out data/support/main
python agents/support.py --prompt v2 --trials 3 --out data/support/pr-v2
agentstats grade data/support/pr-v2/items.jsonl --rubric config/rubric.json --model deepseek-ai/DeepSeek-V4.1-Flash \
    --base-url https://api.together.xyz/v1 --api-key-env TOGETHER_API_KEY --repeats 2 \
    --fields customer,order_record,agent_reply --out data/grade/pr-v2-labels.jsonl --rows data/grade/pr-v2-rows.jsonl
inspect eval agents/coding_task.py --model together/deepseek-ai/DeepSeek-V4.1-Flash --epochs 3 --log-format json
```
