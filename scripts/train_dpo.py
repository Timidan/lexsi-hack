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


def judged_turn_trainer():
    """TRL's DPOTrainer, scoring only the tokens from the first judged one on.

    TRL scores every token of the chat: for Tiny Aya that is ~1.5k tokens x a 262k-word vocabulary, in fp32,
    with gradients, which a 16 GB T4 cannot hold even with a 4-bit base. Chosen and rejected share the prompt and
    the judged turn is at the end, so cutting the chat to that window leaves TRL's loss unchanged."""
    import torch
    from trl import DPOTrainer
    from trl.trainer.dpo_trainer import selective_log_softmax, use_adapter

    def window(inputs):
        start = max(int(inputs["completion_mask"].float().argmax(dim=1).min()), 1)  # first judged token
        return start, inputs["input_ids"].shape[1] - start + 1                      # logits from start - 1

    class JudgedTurnDPOTrainer(DPOTrainer):
        def _compute_loss(self, model, inputs, return_outputs):
            assert self.precompute_ref_logps, "the reference must be scored up front (precompute_ref_log_probs)"
            start, keep = window(inputs)
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        logits_to_keep=keep, use_cache=False)
            cut = {k: v[:, start - 1:] if k in ("input_ids", "attention_mask", "completion_mask") else v
                   for k, v in inputs.items()}
            return super()._compute_loss(lambda **_: out, cut, return_outputs)

        def compute_ref_log_probs(self, inputs):
            start, keep = window(inputs)
            peft = self.accelerator.unwrap_model(self.model)
            with torch.no_grad(), use_adapter(peft, adapter_name="ref" if "ref" in peft.peft_config else None):
                logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                    logits_to_keep=keep, use_cache=False).logits[:, :-1]
            logps = selective_log_softmax(logits, inputs["input_ids"][:, start:]) * inputs["completion_mask"][:, start:]
            return logps.sum(dim=1).chunk(2, dim=0)

    return JudgedTurnDPOTrainer


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
    import trl
    from agenttune.core.backend_factory import create_agentic_trainer
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trl.DPOTrainer = judged_turn_trainer()   # AgentTune's TRL backend builds `trl.DPOTrainer`
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
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)

    trainer = create_agentic_trainer(
        "dpo", backend="trl", model=model, processing_class=tok, train_dataset=data, output_dir=a.out,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=5, beta=a.beta, loss_type=["sigmoid", "sft"], loss_weights=[1.0, a.sft_weight],
        per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum, max_length=a.max_len,
        precompute_ref_log_probs=True,   # the frozen v2 scores every pair once, before training
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=a.dtype == "fp16", bf16=a.dtype == "bf16", logging_steps=5, save_strategy="no", report_to="none",
        ddp_find_unused_parameters=False)
    stats = trainer.train()
    if int(os.environ.get("RANK", 0)) == 0:
        print(json.dumps({k: stats.get(k) for k in ("train_size", "final_loss", "total_steps", "metrics")},
                         default=str))


if __name__ == "__main__":
    main()
