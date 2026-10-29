"""agentstats preregister and switch --prereg: the rule comes from a committed file, and the check is real."""
from __future__ import annotations

import json
import os
import subprocess
import time

from test_switch import ALL_PASS, rows

from agentstats import cli


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "labels.jsonl").write_text('{"case": "c1"}\n')
    return repo


def _register(repo, *extra):
    return cli.main(["preregister", "--name", "demo", "--margin", "0.03", "--provenance", "independent",
                     "--input", f"labels={repo / 'labels.jsonl'}", "--out", str(repo / "d"), *extra])


def _commit(repo):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "register")


def _results(repo, cheap_cost=0.02):
    time.sleep(1.1)                                   # results strictly after the registration commit (1 s clock)
    cur, cand = repo / "cur.jsonl", repo / "cand.jsonl"
    cur.write_text("".join(json.dumps(r) + "\n" for r in rows("cur", ALL_PASS, 0.10)))
    cand.write_text("".join(json.dumps(r) + "\n" for r in rows("cand", ALL_PASS, cheap_cost)))
    return cur, cand


def _decide(repo, cur, cand, *extra):
    code = cli.main(["switch", "--prereg", str(repo / "d" / "prereg.json"), "--current", str(cur),
                     "--candidate", str(cand), "--out", str(repo / "out"), *extra])
    out = repo / "out" / "decision.json"
    rec = json.loads(out.read_text()) if out.exists() else None
    return code, rec


def test_preregister_writes_the_rule_and_never_overwrites(tmp_path):
    repo = _repo(tmp_path)
    assert _register(repo, "--slice-by", "repo") == 0
    reg = json.loads((repo / "d" / "prereg.json").read_text())
    assert reg["rule"]["margin"] == 0.03 and reg["slice_by"] == "repo" and len(reg["inputs"]["labels"]["sha256"]) == 64
    assert "**SWITCH** if the lower 95% bound" in (repo / "d" / "prereg.md").read_text()
    assert _register(repo) == 1                       # a registration is never rewritten


def test_committed_before_the_results_is_verified_and_the_rule_comes_from_the_file(tmp_path):
    repo = _repo(tmp_path)
    _register(repo)
    _commit(repo)
    code, rec = _decide(repo, *_results(repo))
    assert code == 0 and rec["decision"] == "SWITCH"
    assert rec["margin_declared_before_results"] is True and rec["preregistration"]["verified"] is True
    assert rec["rule"]["margin"] == 0.03 and rec["name"] == "demo"
    assert "verified: committed" in (repo / "out" / "decision.md").read_text()


def test_uncommitted_registration_is_not_preregistered(tmp_path):
    repo = _repo(tmp_path)
    _register(repo)
    _, rec = _decide(repo, *_results(repo))
    assert rec["margin_declared_before_results"] is False
    assert "never committed" in " ".join(rec["preregistration"]["reasons"])
    assert any("not verified" in w for w in rec["warnings"])


def test_results_from_before_the_registration_are_caught(tmp_path):
    repo = _repo(tmp_path)
    cur, cand = _results(repo)
    old = time.time() - 3600
    os.utime(cur, (old, old))
    os.utime(cand, (old, old))
    _register(repo)
    _commit(repo)
    _, rec = _decide(repo, cur, cand)
    assert rec["margin_declared_before_results"] is False
    assert "older than the registration" in " ".join(rec["preregistration"]["reasons"])


def test_a_changed_declared_input_is_caught(tmp_path):
    repo = _repo(tmp_path)
    _register(repo)
    _commit(repo)
    (repo / "labels.jsonl").write_text('{"case": "c1", "relabelled": true}\n')
    _, rec = _decide(repo, *_results(repo))
    assert "changed since registration" in " ".join(rec["preregistration"]["reasons"])


def test_flags_that_contradict_the_registration_are_an_error(tmp_path, capsys):
    repo = _repo(tmp_path)
    _register(repo)
    _commit(repo)
    code, rec = _decide(repo, *_results(repo), "--margin", "0.10")
    assert code == 1 and rec is None and "contradict the registration" in capsys.readouterr().out
