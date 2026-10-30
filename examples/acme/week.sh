#!/usr/bin/env bash
# A week at Acme, replayed from recorded runs: no API keys, no cost. Run from this directory:
#   pip install agentstats && bash week.sh
# Each step is one row of the weekly table in README.md. Outputs go to out/ (recreated on each run).
set -uo pipefail
cd "$(dirname "$0")"
OUT=out
mkdir -p "$OUT"
A=agentstats
step() { printf '\n\033[1m===== %s\033[0m\n' "$1"; }

step "1. Every PR: smoke gate (16 sampled cases, 1 trial) on PR #42, the 'simplified' prompt v2"
$A sample config/cases.jsonl --policy config/agentstats.toml --tier smoke > "$OUT/smoke_cases.txt"
python3 scripts/pick.py "$OUT/smoke_cases.txt" data/support/main/rows.jsonl "$OUT/smoke_main.jsonl"
python3 scripts/pick.py "$OUT/smoke_cases.txt" data/support/pr-v2/rows.jsonl "$OUT/smoke_v2.jsonl"
$A gate --policy config/agentstats.toml --tier smoke --candidate "$OUT/smoke_v2.jsonl" --baseline "$OUT/smoke_main.jsonl"
echo "exit code: $? (1 = FAIL: the PR is blocked)"

step "2. Nightly: full gate (40 cases x 3 trials), v2 vs main"
$A gate --policy config/agentstats.toml --tier nightly --candidate data/support/pr-v2/rows.jsonl \
    --baseline data/support/main/rows.jsonl
echo "exit code: $?"

step "2b. Nightly: behaviour rules on last week's production traces (rules-only: no labels needed)"
$A convert-traces data/support/production-week/traces.json --format otlp --rules config/rules.json --arm production \
    --out "$OUT/production_rows.jsonl"
$A summary "$OUT/production_rows.jsonl"

step "2c. Nightly: what an unavailable model looks like (every call failed): UNHEALTHY, not 0%"
$A gate --policy config/agentstats.toml --tier nightly --candidate data/support/unavailable-model/rows.jsonl \
    --baseline data/support/main/rows.jsonl | head -4

step "3. Monday: sampled human review of the LLM judge (rubric judge on v2's replies)"
rm -rf "$OUT/review-wk1"
$A review sample data/grade/pr-v2-rows.jsonl --n 30 --stratify-by intent --out "$OUT/review-wk1"
python3 scripts/simulated_reviewer.py "$OUT/review-wk1" data/support/pr-v2/rows.jsonl
$A review import "$OUT/review-wk1/review.csv" --key "$OUT/review-wk1/key.jsonl" --rater orderdb \
    --out "$OUT/review-wk1/labels.jsonl"

step "4. Monday: last week's production failures become candidate eval cases"
$A promote find data/grade/production-rows.jsonl --source production --label-from human \
    --existing config/cases.jsonl --found 2026-11-09 --out "$OUT/candidates.jsonl"
python3 scripts/approve_all.py "$OUT/candidates.jsonl"          # stands in for a person approving them
cp config/cases.jsonl "$OUT/cases.jsonl"
$A promote accept "$OUT/candidates.jsonl" --into "$OUT/cases.jsonl"

step "5. New model out: Claude Haiku 5.5 instead of DeepSeek-V4.1-Flash"
rm -rf "$OUT/switch-haiku"
$A switch --current data/support/main/rows.jsonl --candidate data/support/haiku-5-5/rows.jsonl --margin 0.03 \
    --provenance independent --slice-by intent --name support-haiku-5-5 --out "$OUT/switch-haiku" | head -12

step "6. The coding agent: Inspect AI logs read directly"
$A summary data/coding/*.json | head -8
