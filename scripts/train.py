"""Train the LoRA adapter (one GPU, or several with torchrun).

    python scripts/train.py --out outputs/sft-v2                       # one GPU
    torchrun --standalone --nproc_per_node=2 scripts/train.py --out outputs/sft-v2   # two GPUs

Set OYA_PROMPT=compact to train with the compact prompt (must match the data
built by scripts/build_data.py and the prompt used at evaluation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naija_agent.sft import train_lora  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--train", default="data/sft_train.jsonl")
    ap.add_argument("--val", default="data/sft_val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--save-every", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--time-budget-h", type=float, default=None)
    a = ap.parse_args()
    train_lora(a.model, a.train, a.out, val_path=a.val or None, max_len=a.max_len, epochs=a.epochs, lr=a.lr,
               grad_accum=a.grad_accum, lora_r=a.lora_r, lora_alpha=a.lora_alpha, save_every=a.save_every,
               seed=a.seed, load_in_4bit=a.load_in_4bit,
               time_budget_s=a.time_budget_h * 3600 if a.time_budget_h else None)


if __name__ == "__main__":
    main()
