"""Evaluate a policy on the held-out test tasks.

    python scripts/eval.py --policy gold --out runs/gold                # CPU sanity check
    python scripts/eval.py --policy hf --out runs/baseline              # untrained Tiny Aya
    python scripts/eval.py --policy hf --adapter <hf-repo-or-dir> --out runs/sft
    python scripts/eval.py --policy gold --out runs/gold --auditkit     # also score with AuditKIT

Writes to --out: trajectories.jsonl (AgentTune format, readable by AuditKIT),
scores.jsonl, metrics.json, report.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naija_agent.evaluate import evaluate, load_tasks  # noqa: E402
from naija_agent.runner import batched, gold_policy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["gold", "hf"], required=True)
    ap.add_argument("--tasks", default="data/tasks_test.jsonl")
    ap.add_argument("--model", default=None, help="HF model id (default: CohereLabs/tiny-aya-global)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter (HF repo id or local dir)")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (quick checks)")
    ap.add_argument("--auditkit", action="store_true", help="also score the run with AuditKIT")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = load_tasks(args.tasks, args.limit)
    if args.policy == "gold":
        policy = batched(gold_policy)
    else:
        from naija_agent.hf_policy import DEFAULT_MODEL, HFPolicy
        policy = HFPolicy(args.model or DEFAULT_MODEL, adapter=args.adapter,
                          load_in_4bit=args.load_in_4bit, max_new_tokens=args.max_new_tokens)
    evaluate(tasks, policy, args.out, batch_size=args.batch_size, config=vars(args))
    if args.auditkit:
        from naija_agent.auditkit_bridge import auditkit_eval
        print("\n== AuditKIT ==")
        print(auditkit_eval(tasks, args.out).summary())


if __name__ == "__main__":
    main()
