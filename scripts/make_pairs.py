"""Sample the model at every step of fresh tasks and keep its mistakes as DPO pairs.

    OYA_PROMPT=compact python scripts/make_pairs.py --adapter outputs/sft-v2 --out data/pairs-0.jsonl --shard 0/2

Tasks use the phrasings kept apart for DPO (lang.DPO_REQUESTS, Pidgin and Yoruba) with fresh names, numbers
and amounts; never the development set's templates or the native-speaker test requests. On the SFT training
phrasings v2 makes almost no mistakes, so they give no pairs. Run one shard per GPU and concatenate the outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naija_agent.pairs import make_pairs, positions  # noqa: E402
from naija_agent.tasks import generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="dpo", help="template pool: dpo (default) or train")
    ap.add_argument("--per-cell", type=int, default=20, help="tasks per (category, language)")
    ap.add_argument("--seed", type=int, default=2027, help="not 2026, which built the SFT data")
    ap.add_argument("--samples", type=int, default=4, help="samples per step")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16, help="conversations per generate() call")
    ap.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th task from k")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from naija_agent.hf_policy import HFPolicy

    k, n = map(int, args.shard.split("/"))
    tasks = generate(args.split, args.per_cell, args.seed)[k::n]
    steps = [(t, prefix, gold) for t in tasks for prefix, gold in positions(t)]
    policy = HFPolicy(args.model, adapter=args.adapter, load_in_4bit=args.load_in_4bit, max_new_tokens=160,
                      temperature=args.temperature)
    per_call = max(1, args.batch // args.samples)
    pairs, t0, shown = [], time.time(), 0
    for i in range(0, len(steps), per_call):
        chunk = steps[i:i + per_call]
        outs = policy([prefix for _, prefix, _ in chunk for _ in range(args.samples)])
        for j, (task, prefix, gold) in enumerate(chunk):
            pairs += make_pairs(task, prefix, gold, outs[j * args.samples:(j + 1) * args.samples])
        done = min(i + per_call, len(steps))
        if done - shown >= len(steps) / 10 or done == len(steps):
            shown = done
            print(f"[shard {args.shard}] {done}/{len(steps)} steps, {len(pairs)} pairs ({time.time() - t0:.0f}s)",
                  flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"shard": args.shard, "tasks": len(tasks), "steps": len(steps), "pairs": len(pairs),
                      "by_category": Counter(p["category"] for p in pairs),
                      "by_lang": Counter(p["lang"] for p in pairs)}))


if __name__ == "__main__":
    main()
