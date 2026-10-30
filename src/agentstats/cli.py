"""agentstats command line: summary, compare, judge, check, convert, sample, gate, size, init-ci,
preregister, switch."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agentstats import adapters, analysis, checks, gate, judge, policy, records, report, sample, sizing, switch_cli
from agentstats.analysis import DESIGNS
from agentstats.ci_templates import FILES as CI_FILES

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


def _review(args) -> int:
    from agentstats import review
    if args.review_cmd == "sample":
        atts = _load_many(args.rows, None)
        sheet, key, summary = review.sample(atts, args.n, stratify_by=args.stratify_by, seed=args.seed,
                                            link=args.link)
        sheet_path, key_path = review.write(args.out, sheet, key, summary)
        print(f"wrote {sheet_path} ({len(sheet)} items: {summary['sampled']}) and {key_path}. "
              "Give the reviewer only review.csv.")
        return 0
    labels, report = review.import_labels(args.sheet, args.key, rater=args.rater)
    out = Path(args.out)
    with out.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in labels)
    print(json.dumps(report, indent=2) if args.json else review.render(report))
    print(f"appended {len(labels)} label rows to {out}; run `agentstats judge {out}` for the full history")
    return 0


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

    v = sub.add_parser("convert", help="Harbor job dir(s) or Inspect AI logs to agentstats JSONL")
    v.add_argument("jobs", nargs="+",
                   help="Harbor job directories, or Inspect logs (.eval / .json) with --format inspect")
    v.add_argument("--format", choices=["harbor", "inspect"], default="harbor")
    v.add_argument("--scorer", help="Inspect: which scorer gives pass/fail (default: the log's headline scorer)")
    v.add_argument("--pass-threshold", type=float, default=1.0, help="Inspect: numeric scores pass at or above this")
    v.add_argument("--errors", choices=["infra", "fail"], default="infra",
                   help="Inspect: a sample that errored is an infrastructure failure (default) or a failed attempt")
    v.add_argument("--rules", help="Inspect: rules JSON; fills each row's checks from the sample's tool calls")
    v.add_argument("--arm", help="arm name (Inspect default: the log's model)")
    v.add_argument("--out", required=True)

    tr = sub.add_parser("convert-traces", help="OpenTelemetry (OTLP JSON) or Langfuse traces to agentstats JSONL")
    tr.add_argument("traces", nargs="+", help="OTLP JSON export(s), or Langfuse trace JSON / JSONL")
    tr.add_argument("--format", choices=["otlp", "langfuse"], required=True)
    tr.add_argument("--rules", help="rules JSON: fills each row's checks from the trace's tool calls")
    tr.add_argument("--verdicts", help="JSONL with trace_id, passed (and optional case, run, infra_error); "
                    "without it the rows are rules-only (passed = null)")
    tr.add_argument("--case-key", default="agentstats.case", help="OTLP span attribute naming the case")
    tr.add_argument("--arm", default="default")
    tr.add_argument("--out", required=True)

    sm = sub.add_parser("sample", help="deterministic stratified sample of cases for a smoke tier")
    sm.add_argument("cases", help="JSONL with a 'case' and optional 'tags' per row (cases or attempt rows)")
    sm.add_argument("--policy", help="take n, stratify_by, always_include, seed from this policy's tier")
    sm.add_argument("--tier", default="smoke")
    sm.add_argument("--n", type=int)
    sm.add_argument("--stratify-by")
    sm.add_argument("--always-include", nargs="*", default=None)
    sm.add_argument("--seed", type=int)
    sm.add_argument("--json", action="store_true")

    g = sub.add_parser("gate", help="check a tier's requirements; exit 0 pass, 1 fail, 2 inconclusive, 3 unhealthy")
    g.add_argument("--policy", required=True)
    g.add_argument("--tier", required=True)
    g.add_argument("--candidate", nargs="+", required=True, help="results (several paths = several runs)")
    g.add_argument("--baseline", nargs="+", help="baseline results, for comparison requirements")
    g.add_argument("--conf", type=float, default=0.95)
    g.add_argument("--json", action="store_true")

    sz = sub.add_parser("size", help="simulate a gate on your baseline: pass/fail rates for a range of true drops")
    sz.add_argument("--baseline", nargs="+", required=True, help="baseline results (several trials per case)")
    sz.add_argument("--trials", type=int, default=2, help="candidate trials per case (what each PR would run)")
    sz.add_argument("--rule", choices=sizing.RULES, default="drop",
                    help="drop = no_significant_drop (PR tier); noninferiority = no_regression (release tier)")
    sz.add_argument("--margin", type=float, default=0.0)
    sz.add_argument("--conf", type=float, default=0.90)
    sz.add_argument("--sims", type=int, default=1000)
    sz.add_argument("--seed", type=int, default=0)
    sz.add_argument("--json", action="store_true")

    ic = sub.add_parser("init-ci", help="write a starter policy and smoke/nightly/release GitHub workflows")
    ic.add_argument("--dir", default=".")
    ic.add_argument("--force", action="store_true")

    rv = sub.add_parser("review", help="sampled human review of graded attempts: is the grader right?")
    rsub = rv.add_subparsers(dest="review_cmd", required=True)
    rs = rsub.add_parser("sample", help="pick attempts for a person to review (blind sheet + key)")
    rs.add_argument("rows", nargs="+", help="attempt rows: JSONL, Inspect logs or Harbor jobs (one per run)")
    rs.add_argument("--n", type=int, default=40, help="how many to review (default 40)")
    rs.add_argument("--stratify-by", help="tag prefix to spread the sample over, e.g. intent for intent:<name>")
    rs.add_argument("--seed", type=int, default=0)
    rs.add_argument("--link", help="link template for each item, e.g. 'https://traces/{case}?run={run}' ({arm} too)")
    rs.add_argument("--out", required=True, help="a new directory: review.csv (for the reviewer), key.jsonl")
    ri = rsub.add_parser("import", help="read a filled review sheet: label rows for `judge`, plus agreement")
    ri.add_argument("sheet", help="the filled review.csv")
    ri.add_argument("--key", required=True, help="key.jsonl written next to the sheet")
    ri.add_argument("--rater", default="human", help="the reviewer's name (labels get rater human:<name>)")
    ri.add_argument("--out", required=True, help="label rows (JSONL) for `agentstats judge`; appended if it exists")
    ri.add_argument("--json", action="store_true")
    switch_cli.add_parsers(sub)
    args = ap.parse_args(argv)
    if args.cmd in ("preregister", "switch"):
        return switch_cli.run(args)
    if args.cmd == "review":
        return _review(args)
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
    elif args.cmd == "sample":
        tier = policy.load(args.policy).tiers[args.tier] if args.policy else None
        n = args.n or (tier.sample if tier else None)
        if not n:
            ap.error("sample needs --n or a policy tier with 'sample'")
        res = sample.stratified(sample.load_cases(args.cases), n,
                                args.stratify_by or (tier.stratify_by if tier else None),
                                args.always_include if args.always_include is not None
                                else (tier.always_include if tier else []),
                                args.seed if args.seed is not None else (tier.seed if tier else 0))
        for note in res["notes"]:
            print(f"note: {note}", file=sys.stderr)
        print(json.dumps(res, indent=2) if args.json else "\n".join(res["cases"]))
    elif args.cmd == "gate":
        pol = policy.load(args.policy)
        if args.tier not in pol.tiers:
            ap.error(f"tier {args.tier!r} not in the policy (have: {', '.join(pol.tiers)})")
        res = gate.evaluate(pol.tiers[args.tier], pol, _load_many(args.candidate, "candidate"),
                            _load_many(args.baseline, "baseline") if args.baseline else None, args.conf)
        text = gate.render(res)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(text)
        print(json.dumps(res, indent=2, default=str) if args.json else text)
        return res["exit_code"]
    elif args.cmd == "size":
        res = sizing.simulate(_load_many(args.baseline, "baseline"), args.trials, args.rule, args.margin, args.conf,
                              sims=args.sims, seed=args.seed)
        print(json.dumps(res, indent=2) if args.json else sizing.render(res))
    elif args.cmd == "init-ci":
        root = Path(args.dir)
        clash = [p for p in CI_FILES if (root / p).exists()]
        if clash and not args.force:
            ap.error(f"already exists: {', '.join(clash)} (use --force to overwrite)")
        for p, body in CI_FILES.items():
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            (root / p).write_text(body)
            print(f"wrote {root / p}")
    elif args.cmd == "convert-traces":
        from agentstats import traces
        from agentstats.checks import load_rules
        atts = traces.convert(args.traces, args.format, rules=load_rules(args.rules) if args.rules else None,
                              verdicts=traces.load_verdicts(args.verdicts) if args.verdicts else None,
                              arm=args.arm, case_key=args.case_key)
        records.write_jsonl(atts, args.out)
        graded = sum(1 for a in atts if a.passed is not None)
        print(f"wrote {args.out}: {len(atts)} traces, {graded} with a verdict"
              + ("" if graded else " (rules-only)"))
    elif args.cmd == "convert" and args.format == "inspect":
        from agentstats import inspect_logs
        rules = checks.load_rules(args.rules) if args.rules else None
        atts = [a for p in args.jobs
                for a in inspect_logs.load(p, scorer=args.scorer, pass_threshold=args.pass_threshold,
                                           errors=args.errors, rules=rules, arm=args.arm)]
        records.write_jsonl(atts, args.out)
        print(f"wrote {args.out}: {len(atts)} attempts")
    elif args.cmd == "convert":
        records.write_jsonl(_load_many(args.jobs, args.arm), args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
