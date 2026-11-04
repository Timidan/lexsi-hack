"""Generate the task sets and the supervised fine-tuning data.

    python scripts/build_data.py                 # defaults below
    python scripts/build_data.py --train-per-cell 60 --test-per-cell 8

Writes to data/:
  tasks_train.jsonl, tasks_test.jsonl  tasks (prompt, world, expected outcome, gold plan)
  sft_train.jsonl, sft_val.jsonl       {"messages": [...]} reference chats for SFT
  stats.json                           counts by category and language
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naija_agent import generate, gold_conversation  # noqa: E402
from naija_agent.prompting import prompt_style  # noqa: E402


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-cell", type=int, default=40, help="tasks per (category, language) for training")
    ap.add_argument("--val-per-cell", type=int, default=2)
    ap.add_argument("--test-per-cell", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = generate("train", args.train_per_cell, args.seed)
    val = generate("train", args.val_per_cell, args.seed + 1)  # same templates, new values: for SFT loss only
    test = generate("test", args.test_per_cell, args.seed)

    write_jsonl(out / "tasks_train.jsonl", (t.to_dict() for t in train))
    write_jsonl(out / "tasks_test.jsonl", (t.to_dict() for t in test))
    for name, tasks in (("sft_train", train), ("sft_val", val)):
        write_jsonl(out / f"{name}.jsonl", (
            {"id": t.id, "category": t.category, "lang": t.lang, "messages": gold_conversation(t)} for t in tasks))

    stats = {
        "prompt_style": prompt_style(),
        "train": len(train), "val": len(val), "test": len(test),
        "train_by_category": Counter(t.category for t in train),
        "train_by_lang": Counter(t.lang for t in train),
        "test_by_category": Counter(t.category for t in test),
        "test_by_lang": Counter(t.lang for t in test),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if isinstance(v, (int, str))}))


if __name__ == "__main__":
    main()
