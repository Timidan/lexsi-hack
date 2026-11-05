"""DPO on the model's own mistakes with Lexsi Labs' AgentTune (pairs from scripts/make_pairs.py).

    OYA_PROMPT=compact python scripts/train_dpo.py --adapter outputs/sft-v2 --pairs data/dpo_pairs.jsonl --out outputs/dpo-v3
    torchrun --standalone --nproc_per_node=2 scripts/train_dpo.py ...     # two GPUs

Continues training the SFT adapter. TRL keeps a frozen copy of it as the DPO reference, so the result is
again one LoRA adapter on Tiny Aya, loaded exactly like the SFT one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", required=True, help="the SFT adapter to start from")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=1.0, help="weight of the plain loss on chosen turns")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=-1, help="for a quick test")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    a = ap.parse_args()

    from naija_agent.compat import remove_old_torchao

    remove_old_torchao()
    import torch
    from agenttune.core.backend_factory import create_agentic_trainer
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    if a.dtype == "auto":
        a.dtype = ("bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16") if cuda else "fp32"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

    rows = [json.loads(line) for line in open(a.pairs, encoding="utf-8") if line.strip()]
    data = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])
    tok = AutoTokenizer.from_pretrained(a.model)
    device = {"": int(os.environ.get("LOCAL_RANK", 0))} if cuda else None
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)

    trainer = create_agentic_trainer(
        "dpo", backend="trl", model=model, processing_class=tok, train_dataset=data, output_dir=a.out,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=5, beta=a.beta, loss_type=["sigmoid", "sft"], loss_weights=[1.0, a.sft_weight],
        per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum, max_length=a.max_len,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=a.dtype == "fp16", bf16=a.dtype == "bf16", logging_steps=5, save_strategy="no", report_to="none",
        ddp_find_unused_parameters=False)
    stats = trainer.train()
    if int(os.environ.get("RANK", 0)) == 0:
        print(json.dumps({k: stats.get(k) for k in ("train_size", "final_loss", "total_steps", "metrics")},
                         default=str))


if __name__ == "__main__":
    main()
