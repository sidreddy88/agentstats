"""
Set up evals for an agent system in one step, from its traces.

    agentstats init --days 14                         # Langfuse (keys in env): find the agents, suggest rules, ask
    agentstats init --otlp traces.json                # an OpenTelemetry export instead
    agentstats init --days 14 --yes                   # accept every suggested rule without asking

It finds the agents in the traces (Langfuse observations of type AGENT, or OTel spans with `gen_ai.agent.name` /
operation `invoke_agent`), runs `suggest` on each agent's runs and asks which rules to keep, then writes:
  agentstats/agents/<agent>.json            one monitoring config per agent (accepted rules; empty rules otherwise:
                                            run counts, errors and no-answers are still monitored)
  .github/workflows/agentstats-monitor.yml  weekly `agentstats monitor`, filed as a GitHub issue
  agentstats.toml + agent-evals-{smoke,nightly,release}.yml   the CI gate starters (as `init-ci`)
Existing files are left alone unless --force. Production traces are ungraded, so every rule it proposes is
monitor-only until an eval set ties it to outcomes; the printed next steps say how.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from agentstats import monitor, suggest
from agentstats.ci_templates import FILES as CI_FILES

CONFIG_DIR = "agentstats/agents"
MONITOR_WORKFLOW = ".github/workflows/agentstats-monitor.yml"


def monitor_workflow(version: str, *, otlp: bool, base_url: str) -> str:
    if otlp:
        fetch = '''      # REPLACE: write an OTLP JSON export of the last 7 days of traces to traces.json
      - name: Export last week's traces
        run: python export_traces.py --days 7 --out traces.json
      - name: Monitor
        run: agentstats monitor --config-dir agentstats/agents --otlp traces.json --out monitor
'''
    else:
        fetch = f'''      - name: Monitor
        env:
          LANGFUSE_PUBLIC_KEY: ${{{{ secrets.LANGFUSE_PUBLIC_KEY }}}}
          LANGFUSE_SECRET_KEY: ${{{{ secrets.LANGFUSE_SECRET_KEY }}}}
          LANGFUSE_BASE_URL: {base_url}
        run: agentstats monitor --config-dir agentstats/agents --days 7 --out monitor
'''
    return f'''name: agentstats-monitor
# Weekly monitoring of every agent configured in agentstats/agents/, from production traces, filed as an issue.
# Written by `agentstats init`. Add --judge --judge-model <model> to run each agent's rubric.
on:
  schedule:
    - cron: "0 14 * * 1"          # Mondays, 14:00 UTC
  workflow_dispatch:

permissions:
  contents: read
  issues: write

jobs:
  monitor:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "agentstats=={version}"
{fetch}      - uses: actions/upload-artifact@v4
        with:
          name: agentstats-monitor
          path: monitor/
          retention-days: 90
      - name: Open the weekly issue
        env:
          GH_TOKEN: ${{{{ github.token }}}}
        run: |
          printf '\\n\\nArtifacts: %s/%s/actions/runs/%s\\n' \\
            "$GITHUB_SERVER_URL" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID" >> monitor/report.md
          gh label create agentstats-monitor --color 5319e7 --description "agentstats monitoring" --force
          gh issue create --title "agentstats monitor, week ending $(date -u +%F)" \\
            --body-file monitor/report.md --label agentstats-monitor
'''


def ask(items: list[dict], input_fn=input) -> list[int]:
    """Ask about each rule item; the ids accepted."""
    accept = []
    for it in items:
        r = it["rule"]
        spec = f"called {r['called']}" if "called" in r else f"{r['before'][0]} before {r['before'][1]}"
        if input_fn(f"[{it['id']}] {it['agent']}: {it['use']} rule {r['name']} ({spec}). {it['why']}. "
                    "Accept? [y/N] ").strip().lower().startswith("y"):
            accept.append(it["id"])
    return accept


def run(root: str | Path, source, *, accept_all: bool = False, interactive: bool = True, force: bool = False,
        otlp: bool = False, base_url: str = "https://cloud.langfuse.com", observation_type: str = "AGENT",
        input_fn=input, log=print) -> dict:
    """Discover agents, suggest and accept rules, write configs and workflows. Returns what was done."""
    import agentstats

    root = Path(root)
    config_dir = root / CONFIG_DIR
    names = source.agent_names(observation_type)
    if not names:
        hint = ("no spans with gen_ai.agent.name or operation invoke_agent" if otlp else
                f"no Langfuse observations of type {observation_type} in the window (try --observation-type SPAN, "
                "or a longer --days)")
        raise ValueError(f"no agents found: {hint}")
    log("Agents found: " + ", ".join(f"{n} ({c} runs)" for n, c in names.most_common()))
    traj = []
    for name in sorted(names):
        traj += monitor.trajectories({"agent": name}, source.traces({"agent": name,
                                                                      "observation_type": observation_type}))
    proposal = suggest.propose(traj, config_dir)
    log("")
    log(suggest.render(proposal))
    rules = [it for it in proposal["items"] if it["kind"] == "rule"]
    if accept_all:
        accepted = [it["id"] for it in rules]
    elif interactive:
        accepted = ask(rules, input_fn)
    else:
        accepted = []
    written = suggest.apply(proposal, config_dir, accepted) if accepted else []
    configs = []
    for name in sorted(names):
        existing = next((p for p in config_dir.glob("*.json") if json.loads(p.read_text()).get("agent") == name), None)
        if existing is None:
            config_dir.mkdir(parents=True, exist_ok=True)
            existing = config_dir / f"{suggest._key(name)}.json"
            existing.write_text(json.dumps({"agent": name, "rules": []}, indent=1) + "\n")
            written.append(str(existing))
        if observation_type != "AGENT":
            cfg = json.loads(existing.read_text())
            if cfg.get("observation_type") != observation_type:
                cfg["observation_type"] = observation_type
                existing.write_text(json.dumps(cfg, indent=1) + "\n")
        configs.append(str(existing))
    files = {MONITOR_WORKFLOW: monitor_workflow(agentstats.__version__, otlp=otlp, base_url=base_url), **CI_FILES}
    wrote_files, skipped = [], []
    for rel, body in files.items():
        p = root / rel
        if p.exists() and not force:
            skipped.append(rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        wrote_files.append(rel)
    (config_dir.parent / "proposal.json").write_text(json.dumps(proposal, indent=1) + "\n")
    return {"agents": dict(names), "accepted": len(accepted), "proposed": len(rules), "configs": configs,
            "written": sorted(set(written)) + wrote_files, "skipped": skipped,
            "advice": [f"{it['agent']}: {it['text']}" for it in proposal["items"]
                       if it["kind"] == "advice" and not it["text"].startswith("no monitoring config yet")]}


def next_steps(result: dict, otlp: bool) -> str:
    lines = [f"Accepted {result['accepted']} of {result['proposed']} suggested rules; "
             f"{len(result['configs'])} agent configs in {CONFIG_DIR}/."]
    if result["written"]:
        lines += ["Wrote:"] + [f"  {p}" for p in result["written"]]
    if result["skipped"]:
        lines += ["Left alone (exists; --force to overwrite):"] + [f"  {p}" for p in result["skipped"]]
    if result["advice"]:
        lines += ["Advice:"] + [f"  {a}" for a in result["advice"]]
    lines += ["", "Next steps:",
              "  1. Try it now: agentstats monitor --config-dir agentstats/agents --days 7 --out monitor"
              + (" --otlp traces.json" if otlp else "") + "  (then read monitor/report.md)",
              "  2. " + ("Replace the export step in .github/workflows/agentstats-monitor.yml" if otlp else
                         "Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY as repository secrets") +
              ", commit agentstats/ and .github/, and the weekly issue starts next Monday.",
              "  3. For a judge: add a rubric (agentstats grade --help) and \"rubric\" to an agent's config, then "
              "--judge --judge-model <model> in the workflow; review its sample by hand before trusting it.",
              "  4. For CI gates: build an eval set (start from the weekly candidates: agentstats promote accept), "
              "then fill the REPLACE steps in the agent-evals-*.yml workflows."]
    return "\n".join(lines)


def window(days: int) -> tuple[dt.datetime, dt.datetime]:
    end = dt.datetime.now(dt.UTC).replace(microsecond=0)
    return end - dt.timedelta(days=days), end
