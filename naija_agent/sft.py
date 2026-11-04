"""Supervised fine-tuning (LoRA) of Tiny Aya on the reference conversations.

Loss is computed only on the assistant's own turns (tool calls and replies,
including the end-of-turn markers, so the model learns when to stop). System
prompt, user messages and tool results are context only.

Written for a single free T4 (16 GB): fp16 LoRA with gradient checkpointing,
batch size 1 with gradient accumulation, and logits computed only at the
positions that are trained (Tiny Aya's 256k vocabulary would otherwise need
several GB just for the logits of a 2-3k token chat).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path


# ─── tokenisation with assistant-only labels (no torch needed) ────────────────

def assistant_spans(tok, messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Render the chat and return (text, [(start, end)]) character spans of each
    assistant turn, including the template's end-of-turn markers."""
    text = tok.apply_chat_template(messages, tokenize=False)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        upto = tok.apply_chat_template(messages[: i + 1], tokenize=False)
        if not (upto.startswith(prefix) and text.startswith(upto)):
            raise ValueError("chat template is not prefix-stable; cannot locate assistant turns")
        spans.append((len(prefix), len(upto)))
    return text, spans


def build_example(tok, messages: list[dict], max_len: int | None = None) -> dict | None:
    """{"input_ids", "labels"} with labels = -100 outside assistant turns.
    Returns None if longer than max_len tokens."""
    text, spans = assistant_spans(tok, messages)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if max_len and len(ids) > max_len:
        return None
    labels = [-100] * len(ids)
    for k, (a, b) in enumerate(offsets):
        if b > a and any(s <= a and b <= e for s, e in spans):
            labels[k] = ids[k]
    return {"input_ids": ids, "labels": labels}


def load_examples(tok, path: str | Path, max_len: int | None = None) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    out, dropped = [], 0
    for r in rows:
        ex = build_example(tok, r["messages"], max_len)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return out, dropped


# ─── training (torch) ─────────────────────────────────────────────────────────

def _assistant_loss(model, input_ids, labels):
    """Mean cross-entropy over assistant tokens, computing logits only where needed."""
    import torch.nn.functional as F

    tgt = labels[1:]
    pos = (tgt != -100).nonzero(as_tuple=True)[0]          # position p predicts token p+1
    out = model(input_ids=input_ids[None], logits_to_keep=pos, use_cache=False)
    logits = out.logits[0]
    if logits.shape[0] != pos.numel():                     # fallback: full logits came back
        logits = logits[pos]
    return F.cross_entropy(logits.float(), tgt[pos])


def train_lora(model_id: str, train_path: str, out_dir: str, *, val_path: str | None = None,
               max_len: int = 4096, epochs: float = 1.0, lr: float = 2e-4, grad_accum: int = 16,
               lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05, warmup_steps: int = 8,
               log_every: int = 5, save_every: int = 40, seed: int = 0, load_in_4bit: bool = False,
               time_budget_s: float | None = None) -> dict:
    """Train a LoRA adapter and save it to out_dir. Returns the training log.

    Runs on one GPU, or on several when launched with torchrun (one process per
    GPU): each process trains on its own share of every batch, the LoRA
    gradients are summed across GPUs before each optimizer step, and only
    process 0 prints, saves and publishes. Results match single-GPU training
    with the same effective batch size (grad_accum)."""
    from .compat import remove_old_torchao

    remove_old_torchao()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from .prompting import prompt_style
    from .publish import Publisher

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        import torch.distributed as dist
        from datetime import timedelta

        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        if grad_accum % world:
            raise ValueError(f"grad_accum ({grad_accum}) must be divisible by the number of GPUs ({world})")
    main = rank == 0

    def say(*args, **kw):
        if main:
            print(*args, **kw, flush=True)

    random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pub = Publisher(out.name) if main else None
    progress_path = out / "progress.json"

    def report(force: bool = False, **status) -> None:
        if not main:
            return
        progress_path.write_text(json.dumps({"phase": "training", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                             "gpus": world, **status}, indent=2))
        files = {"progress.json": progress_path}
        if (out / "train_log.json").exists():
            files["train_log.json"] = out / "train_log.json"
        pub.publish(files, f"{out.name}: training progress", force=force)

    tok = AutoTokenizer.from_pretrained(model_id)
    train, dropped = load_examples(tok, train_path, max_len)
    val = load_examples(tok, val_path, max_len)[0] if val_path else []
    lens = sorted(len(e["input_ids"]) for e in train)
    n_lab = sum(sum(t != -100 for t in e["labels"]) for e in train)
    say(f"train examples: {len(train)} (dropped {dropped} > {max_len} tokens) | "
        f"tokens median {lens[len(lens) // 2]}, max {lens[-1]} | trained tokens {n_lab:,} "
        f"({n_lab / sum(lens):.1%} of all) | prompt style: {prompt_style()} | GPUs: {world}")

    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if bf16 else torch.float16
    kwargs = {"dtype": dtype, "device_map": {"": local}}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    say("attention implementation:", getattr(model.config, "_attn_implementation", "?"))
    if load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    if main:
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:                       # LoRA weights in fp32 so the fp16 grad scaler can work
        p.data = p.data.float()
    if distributed:                        # start every GPU from identical LoRA weights
        for p in params:
            dist.broadcast(p.data, src=0)

    total_steps = max(1, math.ceil(len(train) * epochs / grad_accum))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup_steps) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    dev = torch.device(f"cuda:{local}")

    def to_t(ex):
        return (torch.tensor(ex["input_ids"], device=dev), torch.tensor(ex["labels"], device=dev))

    def all_sum(values: list[float]) -> list[float]:
        if not distributed:
            return values
        t = torch.tensor(values, device=dev, dtype=torch.float64)
        dist.all_reduce(t)
        return t.tolist()

    def val_loss():
        if not val:
            return None
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            losses = [_assistant_loss(model, *to_t(ex)).item() for ex in val[rank::world]]
        model.train()
        total, count = all_sum([sum(losses), len(losses)])
        return total / max(count, 1)

    # sanity check before spending GPU time: the loss must be finite
    model.train()
    with torch.autocast("cuda", dtype=dtype):
        first = _assistant_loss(model, *to_t(train[0])).item()
    if not math.isfinite(first):
        raise RuntimeError(f"initial loss is {first}; try load_in_4bit=True or a bf16 GPU")
    say(f"initial loss {first:.3f} | steps {total_steps} | effective batch {grad_accum} "
        f"({grad_accum // world} per GPU x {world}) | dtype {dtype}")
    report(force=True, step=0, total_steps=total_steps, initial_loss=first, train_examples=len(train))
    if main:
        say("live progress -> GitHub 'results' branch" if pub.enabled else "live progress publishing off (no GITHUB_TOKEN)")

    config = {"model_id": model_id, "max_len": max_len, "epochs": epochs, "lr": lr, "grad_accum": grad_accum,
              "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
              "warmup_steps": warmup_steps, "seed": seed, "load_in_4bit": load_in_4bit, "dtype": str(dtype),
              "train_examples": len(train), "total_steps": total_steps, "gpus": world,
              "prompt_style": prompt_style()}
    log = {"config": config, "steps": [], "val": []}
    log["val"].append({"step": 0, "loss": val_loss()})

    # same shuffled order on every GPU (same seed); each GPU takes every world-th example
    order: list[int] = []
    while len(order) < total_steps * grad_accum:
        idx = list(range(len(train)))
        random.shuffle(idx)
        order += idx
    order = order[: total_steps * grad_accum][rank::world]
    accum_here = grad_accum // world

    t0, tokens, run_loss = time.time(), 0, 0.0
    opt.zero_grad(set_to_none=True)
    for i, ex_idx in enumerate(order):
        ids, labels = to_t(train[ex_idx])
        with torch.autocast("cuda", dtype=dtype):
            loss = _assistant_loss(model, ids, labels) / grad_accum
        scaler.scale(loss).backward()
        run_loss += loss.item()
        tokens += ids.numel()
        if (i + 1) % accum_here:
            continue
        step = (i + 1) // accum_here
        if distributed:                    # sum the (scaled) LoRA gradients across GPUs
            grads = [p.grad for p in params if p.grad is not None]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            offset = 0
            for g in grads:
                n = g.numel()
                g.copy_(flat[offset:offset + n].view_as(g))
                offset += n
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        elapsed = time.time() - t0
        step_loss, all_tokens = all_sum([run_loss, float(tokens)])
        run_loss = 0.0
        log["steps"].append({"step": step, "loss": step_loss, "grad_norm": gnorm, "lr": sched.get_last_lr()[0],
                             "elapsed_s": round(elapsed, 1)})
        if step % log_every == 0 or step == total_steps:
            eta = elapsed / step * (total_steps - step)
            say(f"step {step}/{total_steps} loss {step_loss:.4f} grad_norm {gnorm:.2f} "
                f"lr {sched.get_last_lr()[0]:.2e} | {all_tokens / elapsed:.0f} tok/s | eta {eta / 60:.0f} min")
            report(step=step, total_steps=total_steps, loss=step_loss, grad_norm=gnorm,
                   tokens_per_s=round(all_tokens / elapsed), eta_min=round(eta / 60))
        if step % save_every == 0 or step == total_steps:
            if main:
                model.save_pretrained(out)
            log["val"].append({"step": step, "loss": val_loss()})
            say(f"  saved adapter at step {step} | val loss {log['val'][-1]['loss']}")
            if main:
                (out / "train_log.json").write_text(json.dumps(log, indent=2))
        # every GPU must agree on stopping, or the next gradient sum would hang
        over = 1.0 if (time_budget_s and elapsed > time_budget_s) else 0.0
        if all_sum([over])[0] > 0:
            say(f"time budget reached at step {step}; stopping early")
            if main:
                model.save_pretrained(out)
            log["stopped_early_at"] = step
            break

    log["seconds"] = round(time.time() - t0, 1)
    if main:
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        tok.save_pretrained(out)
    report(force=True, step=log["steps"][-1]["step"] if log["steps"] else 0, total_steps=total_steps,
           done=True, minutes=round(log["seconds"] / 60), val=log["val"])
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return log
