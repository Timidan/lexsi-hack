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
    ap.add_argument("--load-in-4bit", action="store_true", help="frozen base in 4-bit (QLoRA); needs a GPU")
    a = ap.parse_args()

    from naija_agent.compat import remove_old_torchao

    remove_old_torchao()
    import torch
    import trl.trainer.dpo_trainer as trl_dpo
    from agenttune.core.backend_factory import create_agentic_trainer

    # TRL logs the mean token entropy by copying the full logits (2 sequences x ~1.5k tokens x Tiny Aya's 262k
    # vocabulary, about 3 GB); on a 16 GB T4 that copy alone ran out of memory. It is only a metric: log 0.
    trl_dpo.entropy_from_logits = lambda logits, chunk_size=128: logits.new_zeros(logits.shape[:-1])
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda = torch.cuda.is_available()
    if a.dtype == "auto":
        a.dtype = ("bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16") if cuda else "fp32"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

    tok = AutoTokenizer.from_pretrained(a.model)
    # from a file, not from_list: precomputing the reference scores caches them next to the dataset's own file
    data = load_dataset("json", data_files=a.pairs, split="train").select_columns(["prompt", "chosen", "rejected"])
    n = len(data)

    def fits(r) -> bool:  # TRL would cut a longer pair from the end, where the judged turn is: drop it instead
        return all(len(tok.apply_chat_template(r["prompt"] + r[s], tokenize=True, return_dict=True)["input_ids"])
                   <= a.max_len for s in ("chosen", "rejected"))

    data = data.filter(fits)
    print(f"{len(data)} of {n} pairs fit in {a.max_len} tokens", flush=True)
    device = {"": int(os.environ.get("LOCAL_RANK", 0))} if cuda else None
    quant = None
    if a.load_in_4bit:  # QLoRA: the frozen base in 4-bit leaves a 16 GB T4 room for the 262k-vocabulary logits
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map=device, quantization_config=quant)
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)

    trainer = create_agentic_trainer(
        "dpo", backend="trl", model=model, processing_class=tok, train_dataset=data, output_dir=a.out,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=5, beta=a.beta, loss_type=["sigmoid", "sft"], loss_weights=[1.0, a.sft_weight],
        per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum, max_length=a.max_len,
        # the reference (frozen v2) scores every pair once before training, instead of inside every step,
        # where its full logits sat in GPU memory next to the trained model's
        precompute_ref_log_probs=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=a.dtype == "fp16", bf16=a.dtype == "bf16", logging_steps=5, save_strategy="no", report_to="none",
        ddp_find_unused_parameters=False)
    if a.load_in_4bit and a.dtype == "fp16":
        # TRL casts QLoRA adapter weights to bf16; fp16 training (T4) needs them in fp32 for its gradient scaler
        setup = trainer.setup_trainer

        def setup_then_fp32() -> None:
            setup()
            for p in trainer.trainer.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()

        trainer.setup_trainer = setup_then_fp32
    stats = trainer.train()
    if int(os.environ.get("RANK", 0)) == 0:
        print(json.dumps({k: stats.get(k) for k in ("train_size", "final_loss", "total_steps", "metrics")},
                         default=str))


if __name__ == "__main__":
    main()
