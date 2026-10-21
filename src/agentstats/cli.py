"""agentstats command line: summary, compare, judge, check, convert."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentstats import adapters, analysis, checks, judge, records, report
from agentstats.analysis import DESIGNS

DESIGN_HELP = ("how the eval set was built: curated (hand-picked, the default), sampled (random from real "
               "traffic), clustered (sampled, in groups that fail together; needs --cluster-by)")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--design", choices=DESIGNS, default="curated", help=DESIGN_HELP)
    p.add_argument("--cluster-by", help="tag prefix that names each case's group, e.g. 'feature' for feature:search")
    p.add_argument("--conf", type=float, default=0.95, help="confidence level (default 0.95)")
    p.add_argument("--json", action="store_true", help="print JSON instead of a report")


def _load_many(paths: list[str], arm: str | None) -> list[records.Attempt]:
    """Several paths for one arm are separate runs (e.g. three Harbor jobs)."""
    out = []
    for i, p in enumerate(paths, 1):
        atts = records.load(p, arm=arm, run=i if len(paths) > 1 else None)
        if len(paths) > 1 and Path(p).is_file():
            for a in atts:
                a.run = i
        out += atts
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentstats", description="Statistics for agent evals.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help="pass rate with the right CI, reliability, efficiency, infra failures")
    s.add_argument("results", nargs="+", help="JSONL file(s) or Harbor job dir(s); several = several runs")
    _common(s)

    c = sub.add_parser("compare", help="is B different from A?")
    c.add_argument("--a", nargs="+", required=True, help="baseline results (JSONL or Harbor job dirs)")
    c.add_argument("--b", nargs="+", required=True, help="candidate results")
    c.add_argument("--a-name", default="A")
    c.add_argument("--b-name", default="B")
    c.add_argument("--min-effect", type=float, help="ship only if the CI's lower bound beats this (e.g. 0.02)")
    _common(c)

    j = sub.add_parser("judge", help="judge vs human labels, human agreement, judge consistency")
    j.add_argument("labels", help="JSONL: item, rater (human:<name> | judge), verdict, [run], [split]")
    j.add_argument("--json", action="store_true")

    k = sub.add_parser("check", help="run deterministic trajectory rules over a Harbor job; writes JSONL")
    k.add_argument("job", help="Harbor job directory")
    k.add_argument("--rules", required=True, help="rules JSON (see agentstats.checks)")
    k.add_argument("--out", required=True, help="output JSONL with checks filled in")

    v = sub.add_parser("convert", help="Harbor job dir(s) to agentstats JSONL")
    v.add_argument("jobs", nargs="+")
    v.add_argument("--arm")
    v.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "summary":
        res = analysis.summarize(_load_many(args.results, None), args.design, args.cluster_by, args.conf)
        print(json.dumps(res, indent=2, default=str) if args.json else report.summary(res))
    elif args.cmd == "compare":
        res = analysis.compare(_load_many(args.a, "A"), _load_many(args.b, "B"), args.design, args.cluster_by,
                               args.conf, args.min_effect)
        print(json.dumps(res, indent=2, default=str) if args.json else report.compare(res, args.a_name, args.b_name))
    elif args.cmd == "judge":
        res = judge.analyze(judge.load(args.labels))
        print(json.dumps(res, indent=2, default=str) if args.json else report.judge(res))
    elif args.cmd == "check":
        rules = checks.load_rules(args.rules)
        out = []
        for t in sorted(p for p in Path(args.job).iterdir() if p.is_dir() and (p / "result.json").exists()):
            a = adapters.harbor_trial(t)
            traj = t / "agent" / "trajectory.json"
            if traj.exists():
                a.checks = checks.run_checks(rules, adapters.tool_calls(traj))
            out.append(a)
        records.write_jsonl(out, args.out)
        print(f"wrote {len(out)} attempts to {args.out}")
    elif args.cmd == "convert":
        records.write_jsonl(_load_many(args.jobs, args.arm), args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
