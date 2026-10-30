"""Fill a review sheet as a person would, using the order database's ground truth (the support agent's own rows) in
their place. In real use a person reads each conversation and writes pass or fail.  simulated_reviewer.py DIR ROWS"""
import csv
import json
import sys

review_dir, truth_rows = sys.argv[1:3]
truth = {(r["case"], r["run"]): r["passed"] for r in map(json.loads, open(truth_rows))}
key = {k["item"]: k for k in map(json.loads, open(f"{review_dir}/key.jsonl"))}
rows = list(csv.DictReader(open(f"{review_dir}/review.csv")))
for r in rows:
    k = key[r["item"]]
    r["human_verdict"] = "pass" if truth[(k["case"], int(k["run"]))] else "fail"
    r["notes"] = "simulated reviewer: order database"
with open(f"{review_dir}/review.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"filled {len(rows)} verdicts (simulated reviewer)")
