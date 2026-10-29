"""
    agentstats preregister --name triage-haiku-5-5 --margin 0.03 --provenance current-model \\
        [--input labels=app/evals/triage_heldout.jsonl] [--slice-by template] --out decisions/triage-haiku-5-5
    (commit the registration, then run both arms)

    agentstats switch --prereg decisions/triage-haiku-5-5/prereg.json \\
        --current current.jsonl --candidate cheap.jsonl [--adapted cheap-adapted.jsonl] --out decisions/triage-haiku-5-5

    agentstats switch --current current.jsonl --candidate cheap.jsonl --margin 0.03 --provenance independent \\
        --name my-switch --out decisions/my-switch           # without a registration: recorded as not pre-registered

Exit codes for switch: 0 SWITCH, 1 KEEP, 2 NOT YET, 3 UNHEALTHY.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agentstats import prereg, records
from agentstats import switch as decide
from agentstats import switch_record as record

RULE_DEFAULTS = {"max_cost_ratio": 1.0, "conf": 0.95, "design": "curated"}


def _rule_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--margin", type=float, help="acceptable loss, e.g. 0.03 for 3 points")
    p.add_argument("--max-cost-ratio", type=float)
    p.add_argument("--conf", type=float)
    p.add_argument("--design", choices=["curated", "sampled", "clustered"])
    p.add_argument("--provenance", choices=decide.PROVENANCE,
                   help="who made the eval set: independent of both models, or the current model")
    p.add_argument("--slice-by", help="tag prefix to compare per slice, e.g. repo for repo:<name> tags")
    p.add_argument("--strict-slices", action="store_true", help="the decision can't be better than its worst slice")


def _rule(args) -> decide.Rule:
    if args.margin is None:
        raise SystemExit("--margin is required (or use --prereg)")
    return decide.Rule(margin=args.margin, **{k: getattr(args, k) if getattr(args, k) is not None else v
                                              for k, v in RULE_DEFAULTS.items()})


def _from_prereg(args) -> tuple[prereg.Registration, list[str]]:
    """The registered rule; flags that contradict it are an error, not an override."""
    reg = prereg.load(args.prereg)
    given = {"margin": args.margin, "max_cost_ratio": args.max_cost_ratio, "conf": args.conf, "design": args.design}
    conflicts = [f"--{k.replace('_', '-')} {v} (registered {getattr(reg.rule, k)})"
                 for k, v in given.items() if v is not None and v != getattr(reg.rule, k)]
    if args.provenance and args.provenance != reg.provenance:
        conflicts.append(f"--provenance {args.provenance} (registered {reg.provenance})")
    if args.slice_by and args.slice_by != reg.slice_by:
        conflicts.append(f"--slice-by {args.slice_by} (registered {reg.slice_by})")
    if args.strict_slices and not reg.strict_slices:
        conflicts.append("--strict-slices (registered without)")
    return reg, conflicts


def add_parsers(sub) -> None:
    """`preregister` and `switch` subcommands, added to agentstats' own parser."""


    p = sub.add_parser("preregister", help="write the switch rule down before any result exists")
    _rule_args(p)
    p.add_argument("--name", required=True)
    p.add_argument("--input", action="append", default=[], metavar="NAME=PATH",
                   help="a file the decision depends on (e.g. labels); its hash must match at decision time")
    p.add_argument("--note", default="", help="the plan: arms, cases, trials, how they'll be run")
    p.add_argument("--out", type=Path, required=True, help="directory for prereg.json and prereg.md")

    d = sub.add_parser("switch", help="switch to the cheaper setup? SWITCH / KEEP / NOT YET, with a decision record")
    d.add_argument("--prereg", type=Path, help="prereg.json from `agentstats preregister`: the rule comes from it")
    d.add_argument("--current", type=Path, required=True, help="agentstats rows for the current setup")
    d.add_argument("--candidate", type=Path, required=True, help="rows for the cheaper model, as-is")
    d.add_argument("--adapted", type=Path, help="rows for the cheaper model with an adapted harness (held-out cases)")
    _rule_args(d)
    d.add_argument("--preregistered", action="store_true",
                   help="(without --prereg) you assert the margin was set before any result; recorded as asserted")
    d.add_argument("--name")
    d.add_argument("--note", default="")
    d.add_argument("--out", type=Path, required=True, help="directory for decision.json and decision.md")


def run(args) -> int:

    if args.cmd == "preregister":
        if not args.provenance:
            raise SystemExit("--provenance is required: say who made the eval set")
        inputs = {}
        for item in args.input:
            name, _, path = item.partition("=")
            if not path:
                raise SystemExit(f"--input wants NAME=PATH, got {item!r}")
            inputs[name] = Path(path)
        try:
            path = prereg.create(args.out, args.name, _rule(args), args.provenance, slice_by=args.slice_by,
                                 strict_slices=args.strict_slices, inputs=inputs, note=args.note)
        except (FileExistsError, ValueError) as exc:
            print(exc)
            return 1
        print(f"wrote {path} and {path.with_suffix('.md')}. Commit them before running anything.")
        return 0

    results = [args.current, args.candidate] + ([args.adapted] if args.adapted else [])
    prereg_info = None
    if args.prereg:
        reg, conflicts = _from_prereg(args)
        if conflicts:
            print("these flags contradict the registration (the rule comes from the file): " + "; ".join(conflicts))
            return 1
        rule, name, provenance = reg.rule, args.name or reg.name, reg.provenance
        slice_by, strict = reg.slice_by, reg.strict_slices
        ok, reasons, info = prereg.verify(args.prereg, results)
        preregistered = ok
        prereg_info = {"file": str(args.prereg), "verified": ok, "reasons": reasons, **info}
    else:
        if not args.name:
            raise SystemExit("--name is required (or use --prereg)")
        rule, name = _rule(args), args.name
        provenance, slice_by, strict = args.provenance or "unknown", args.slice_by, args.strict_slices
        preregistered = args.preregistered
        if preregistered:
            prereg_info = {"file": None, "verified": False, "reasons": ["asserted with --preregistered, not checked"]}

    current = records.load(args.current, arm="current")
    candidate = records.load(args.candidate, arm="candidate")
    adapted = records.load(args.adapted, arm="adapted") if args.adapted else None
    result = decide.decide(current, candidate, adapted, rule, slice_by=slice_by, strict_slices=strict,
                           provenance=provenance, preregistered=preregistered)
    if prereg_info and not prereg_info["verified"] and args.prereg:
        result.warnings.append("pre-registration not verified: " + "; ".join(prereg_info["reasons"]))
    inputs = {"current": args.current, "candidate": args.candidate}
    if args.adapted:
        inputs["adapted"] = args.adapted
    r = record.build(result, rule, inputs, name=name, provenance=provenance, preregistered=preregistered,
                     slice_by=slice_by, strict_slices=strict, note=args.note, prereg=prereg_info)
    _, md = record.write(args.out, r)
    print(md.read_text())
    return decide.EXIT[result.decision]

