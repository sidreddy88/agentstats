"""
Pre-registration: write the switch rule down before any result exists, and check later that it was.

    agentstats preregister --name triage-haiku-5-5 --margin 0.03 --provenance current-model \\
        --input labels=app/evals/triage_heldout.jsonl --out decisions/triage-haiku-5-5
    git add decisions/triage-haiku-5-5 && git commit -m "pre-register" && git push
    ... run both arms ...
    agentstats switch --prereg decisions/triage-haiku-5-5/prereg.json --current ... --candidate ... --out ...

Why: the margin, confidence level and slices are easy to pick after seeing the numbers, and an answer chosen that way
isn't evidence of anything. A decision only counts as pre-registered when it can be shown, so `decide --prereg` checks:

1. prereg.json is committed in git, and unchanged since that commit;
2. that commit is older than every result file's last modification (a good-faith check: file times can be changed,
   but this catches the honest mistake of registering after running);
3. every declared input (e.g. the labels) still has the hash it had when registered.

If any check fails the decision is still made with the registered rule, but recorded as NOT pre-registered, with the
reasons. The rule itself always comes from the file; conflicting command-line flags are an error.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import agentstats
from agentstats.switch import PROVENANCE, Rule


@dataclass
class Registration:
    name: str
    rule: Rule
    provenance: str
    slice_by: str | None = None
    strict_slices: bool = False
    inputs: dict[str, dict] = field(default_factory=dict)      # name -> {"path", "sha256"}
    note: str = ""
    created_at: str = ""
    agentstats_version: str = ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(out_dir: Path, name: str, rule: Rule, provenance: str, *, slice_by: str | None = None,
           strict_slices: bool = False, inputs: dict[str, Path] | None = None, note: str = "") -> Path:
    """Write prereg.json and prereg.md. Refuses to overwrite an existing registration."""
    if provenance not in PROVENANCE:
        raise ValueError(f"provenance must be one of {PROVENANCE}")
    out_dir = Path(out_dir)
    if (out_dir / "prereg.json").exists():
        raise FileExistsError(f"{out_dir / 'prereg.json'} exists: a registration is never rewritten")
    reg = Registration(name, rule, provenance, slice_by, strict_slices,
                       {k: {"path": str(p), "sha256": _sha256(p)} for k, p in (inputs or {}).items()}, note,
                       datetime.now(UTC).isoformat(timespec="seconds"), agentstats.__version__)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(reg)
    (out_dir / "prereg.json").write_text(json.dumps(data, indent=1) + "\n")
    (out_dir / "prereg.md").write_text(render(data))
    return out_dir / "prereg.json"


def load(path: Path) -> Registration:
    d = json.loads(Path(path).read_text())
    return Registration(d["name"], Rule(**d["rule"]), d["provenance"], d.get("slice_by"), d.get("strict_slices", False),
                        d.get("inputs") or {}, d.get("note", ""), d.get("created_at", ""),
                        d.get("agentstats_version") or d.get("stepdown_version", ""))  # stepdown-era files


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def verify(path: Path, results: list[Path]) -> tuple[bool, list[str], dict]:
    """Was the registration committed before the results, and are the declared inputs unchanged?"""
    path = Path(path).resolve()
    reasons: list[str] = []
    info: dict = {}
    d = path.parent
    top = _git(d, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return False, ["the registration isn't in a git repository, so when it was written can't be shown"], info
    root = Path(top.stdout.strip())
    rel = str(path.relative_to(root.resolve()))
    d = root                                   # git pathspecs below are relative to the repository root
    log = _git(d, "log", "-1", "--format=%H %ct", "--", rel)
    if not log.stdout.strip():
        return False, ["the registration was never committed"], info
    commit, ts = log.stdout.split()
    info.update(commit=commit, committed_at=datetime.fromtimestamp(int(ts), UTC).isoformat(timespec="seconds"))
    if _git(d, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        reasons.append("the registration has uncommitted changes")
    first = _git(d, "log", "--diff-filter=A", "--format=%ct", "--", rel).stdout.split()
    registered_at = int(first[-1]) if first else int(ts)
    if len(_git(d, "log", "--format=%H", "--", rel).stdout.split()) > 1:
        reasons.append("the registration was changed after it was first committed")
    later = [str(p) for p in results if Path(p).stat().st_mtime < registered_at]
    if later:
        reasons.append(f"result file(s) older than the registration: {', '.join(later)}")
    reg = load(path)
    for name, inp in reg.inputs.items():
        p = Path(inp["path"])                      # as given at registration: relative to where it ran,
        if not p.exists():                         # else to the repository root
            p = root / inp["path"]
        if not p.exists():
            reasons.append(f"declared input {name} ({inp['path']}) is missing")
        elif _sha256(p) != inp["sha256"]:
            reasons.append(f"declared input {name} ({inp['path']}) changed since registration")
    info["registered_at"] = datetime.fromtimestamp(registered_at, UTC).isoformat(timespec="seconds")
    return not reasons, reasons, info


def render(d: dict) -> str:
    r = d["rule"]
    lines = [f"# Pre-registration: {d['name']}", "",
             (f"Written {d['created_at']} with agentstats {d['agentstats_version']}, before any result. Commit this "
              "file before running anything; `agentstats switch --prereg` checks that it was."), "",
             "## Rule", "",
             (f"- **SWITCH** if the lower {100 * r['conf']:g}% bound of (candidate - current) is at least "
              f"-{100 * r['margin']:g} points and the cost ratio's upper bound is at most {r['max_cost_ratio']:g}x;"),
             f"- **KEEP** if the upper bound is below -{100 * r['margin']:g} points, or it costs more;",
             "- **NOT YET** otherwise, reported as is.", "",
             (f"Eval set design: {r['design']}. Infrastructure failures capped at {100 * r['max_infra']:g}%. "
              f"Eval-set provenance: {d['provenance']}. Slices: "
              + (f"by `{d['slice_by']}`, {'enforced' if d['strict_slices'] else 'reported, not enforced'}."
                 if d["slice_by"] else "none.")), ""]
    if d["inputs"]:
        lines += ["## Declared inputs (must be unchanged at decision time)", ""]
        lines += [f"- {k}: `{v['path']}` (sha256 {v['sha256'][:16]})" for k, v in d["inputs"].items()]
        lines.append("")
    if d.get("note"):
        lines += ["## Plan", "", d["note"], ""]
    return "\n".join(lines)
