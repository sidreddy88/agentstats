"""Starter files written by `agentstats init-ci`: a policy and three GitHub Actions workflows (smoke, nightly,
release). Each workflow has one step to replace: running your agent and writing results as JSONL or Harbor jobs."""

POLICY = '''# agentstats policy: evals in CI, in tiers. Thresholds are yours to set; these are starting points.
design = "curated"            # curated (hand-picked cases) | sampled (random from traffic) | clustered

[smoke]                       # every pull request: minutes, a stratified sample + rule checks
sample = 20
stratify_by = "feature"       # cases tagged feature:<name>
always_include = ["critical"]
runs = 1
require.rules = 1.0
require.min_pass_rate = 0.5
require.max_infra = 0.10

[nightly]                     # every night: the whole eval set, several runs per case
runs = 3
require.no_regression = 0.03
require.max_infra = 0.05

[release]                     # before a release: the whole set against the production baseline
runs = 5
require.no_regression = 0.02
require.max_cost_ratio = 1.10
require.max_p95_latency_ratio = 1.20
require.min_pass_hat_k = 0.70
require.max_infra = 0.05
'''

_SETUP = '''    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install agentstats
'''

SMOKE = '''name: agent-evals-smoke
on:
  pull_request:

jobs:
  smoke:
''' + _SETUP + '''      - name: Pick the smoke sample
        run: agentstats sample evals/cases.jsonl --policy agentstats.toml --tier smoke > smoke_cases.txt
      - name: Run the agent on the sample  # REPLACE: write one JSONL row per attempt (see the agentstats README)
        run: python run_evals.py --cases smoke_cases.txt --runs 1 --out results/smoke.jsonl
      - name: Gate
        run: agentstats gate --policy agentstats.toml --tier smoke --candidate results/smoke.jsonl
'''

NIGHTLY = '''name: agent-evals-nightly
on:
  schedule:
    - cron: "0 7 * * *"
  workflow_dispatch:

jobs:
  nightly:
''' + _SETUP + '''      # REPLACE: one file per run, or one file with a "run" field
      - name: Run the agent on the full set
        run: python run_evals.py --runs 3 --out results/nightly.jsonl
      - name: Gate against the baseline
        run: >
          agentstats gate --policy agentstats.toml --tier nightly
          --candidate results/nightly.jsonl --baseline evals/baseline.jsonl
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: nightly-results
          path: results/
'''

RELEASE = '''name: agent-evals-release
on:
  workflow_dispatch:
  push:
    tags: ["v*"]

jobs:
  release:
''' + _SETUP + '''      - name: Run the agent on the full set  # REPLACE
        run: python run_evals.py --runs 5 --out results/release.jsonl
      - name: Gate against the production baseline
        run: >
          agentstats gate --policy agentstats.toml --tier release
          --candidate results/release.jsonl --baseline evals/baseline.jsonl
      - uses: actions/upload-artifact@v4
        # A passing release run is the next baseline: commit it as evals/baseline.jsonl after you ship.
        with:
          name: release-results-candidate-baseline
          path: results/release.jsonl
'''

FILES = {
    "agentstats.toml": POLICY,
    ".github/workflows/agent-evals-smoke.yml": SMOKE,
    ".github/workflows/agent-evals-nightly.yml": NIGHTLY,
    ".github/workflows/agent-evals-release.yml": RELEASE,
}
