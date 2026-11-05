"""Rescore saved runs offline (no GPU) with the current scorer's reply check.

    python scripts/rescore.py data/tasks_human.jsonl untrained=<run dir> fine-tuned=<run dir>

Each run dir holds the scores.jsonl written by evaluate(). Only reply_ok is
recomputed from the saved final replies; the other scores are kept as saved."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naija_agent.evaluate import load_tasks  # noqa: E402
from naija_agent.scoring import aggregate, compare, reply_ok  # noqa: E402

tasks = {t.id: t for t in load_tasks(sys.argv[1])}
metrics = {}
for arg in sys.argv[2:]:
    name, run_dir = arg.split("=", 1)
    rows = [json.loads(line) for line in open(Path(run_dir) / "scores.jsonl", encoding="utf-8")]
    run_tasks = [tasks[r["task_id"]] for r in rows]
    for t, r in zip(run_tasks, rows):
        r["reply_ok"] = reply_ok(t, r["final"])
        if not r["reply_ok"]:
            print(f"{name:<12} reply check failed  {t.id:<24} {(r['final'] or '')[:90]!r}")
    metrics[name] = aggregate(run_tasks, rows)
print()
print(compare(metrics, title=f"{Path(sys.argv[1]).name}, rescored offline"))
