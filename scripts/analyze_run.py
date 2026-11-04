"""Summarize a run folder: what failed, where, and how.

    python scripts/analyze_run.py runs/sft-v1 [--baseline runs/baseline] [--examples 3]

Reads scores.jsonl and trajectories.jsonl (as written by evaluate()).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(run: Path) -> tuple[list[dict], dict[str, dict]]:
    scores = [json.loads(line) for line in (run / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line]
    trajs = {}
    tp = run / "trajectories.jsonl"
    if tp.exists():
        for line in tp.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                trajs[r["metadata"]["task_id"]] = r
    return scores, trajs


def calls_of(traj: dict) -> list[str]:
    out = []
    for step in traj.get("steps", []):
        for c in (step.get("action") or {}).get("tool_calls") or []:
            out.append(f"{c.get('name')}({json.dumps(c.get('arguments'), ensure_ascii=False)[:120]})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    scores, trajs = load(Path(a.run))
    n = len(scores)
    ok = sum(s["success"] for s in scores)
    print(f"{a.run}: {ok:.0f}/{n} succeeded ({ok / n:.1%})")

    by_cat, by_lang = defaultdict(list), defaultdict(list)
    for s in scores:
        by_cat[s["category"]].append(s["success"])
        by_lang[s["lang"]].append(s["success"])
    print("\nby language: " + ", ".join(f"{k} {sum(v) / len(v):.0%}" for k, v in sorted(by_lang.items())))
    print("by category: " + ", ".join(f"{k} {sum(v) / len(v):.0%}" for k, v in sorted(by_cat.items())))

    fails = [s for s in scores if not s["success"]]
    reasons = Counter(tuple(sorted(k for k, v in s["checks"].items() if not v)) for s in fails)
    print(f"\n{len(fails)} failures by failed checks:")
    for r, c in reasons.most_common():
        print(f"  {c:4d}  {', '.join(r)}")
    print("\nparse errors in", sum(s["parse_errors"] > 0 for s in scores), "tasks;",
          "unexpected payments in", sum(s["unexpected_payments"] > 0 for s in scores), "tasks")

    if a.baseline:
        base = {s["task_id"]: s for s in load(Path(a.baseline))[0]}
        fixed = sum(1 for s in scores if s["success"] and not base.get(s["task_id"], {}).get("success"))
        broke = sum(1 for s in scores if not s["success"] and base.get(s["task_id"], {}).get("success"))
        print(f"\nvs baseline: fixed {fixed}, broke {broke}")

    print("\nexamples of failures per category:")
    seen = Counter()
    for s in fails:
        if seen[s["category"]] >= a.examples:
            continue
        seen[s["category"]] += 1
        failed = [k for k, v in s["checks"].items() if not v]
        t = trajs.get(s["task_id"], {})
        print(f"\n[{s['category']} / {s['lang']}] {s['prompt']}\n   failed: {failed}")
        for c in calls_of(t):
            print(f"   call: {c}")
        print(f"   final: {str(s.get('final'))[:220]}")


if __name__ == "__main__":
    main()
