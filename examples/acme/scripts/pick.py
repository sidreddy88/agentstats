"""Keep trial 1 of the sampled smoke cases: what a PR's smoke job would have run.  pick.py CASES ROWS OUT"""
import json
import sys

cases_file, rows_file, out = sys.argv[1:4]
keep = set(open(cases_file).read().split())
rows = [json.loads(x) for x in open(rows_file) if x.strip()]
open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows if r["case"] in keep and r["run"] == 1))
