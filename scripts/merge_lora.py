"""Merge a LoRA adapter into the base model's weights, with no PEFT/transformers
version dependence (only torch + safetensors).

    python scripts/merge_lora.py --base CohereLabs/tiny-aya-global --adapter outputs/sft-v2 --out merged/

For every adapted linear layer: W' = W + (B @ A) * scale, scale = lora_alpha / r
(or lora_alpha / sqrt(r) with rsLoRA). Config and tokenizer files are copied
unchanged from the base model, so tools pinned to older library versions
(like llama.cpp's GGUF converter) read them exactly as they read the original.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

ADAPTER_PREFIX = "base_model.model."


def lora_pairs(adapter_tensors: dict) -> dict[str, tuple]:
    """{base weight key: (A, B)} from PEFT adapter keys like
    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"""
    pairs: dict[str, dict] = {}
    for key, t in adapter_tensors.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            continue
        module = key.split(".lora_")[0]
        if module.startswith(ADAPTER_PREFIX):
            module = module[len(ADAPTER_PREFIX):]
        side = "A" if ".lora_A." in key else "B"
        pairs.setdefault(module + ".weight", {})[side] = t
    missing = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if missing:
        raise ValueError(f"incomplete LoRA pairs for {missing[:3]}")
    return {k: (v["A"], v["B"]) for k, v in pairs.items()}


def merge(base_dir: Path, adapter_dir: Path, out_dir: Path) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    if cfg.get("fan_in_fan_out"):
        raise ValueError("fan_in_fan_out adapters are not supported")
    pairs = lora_pairs(load_file(str(adapter_dir / "adapter_model.safetensors")))

    out_dir.mkdir(parents=True, exist_ok=True)
    merged, shards = set(), sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {base_dir}")
    for shard in shards:
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                w = f.get_tensor(key)
                if key in pairs:
                    a, b = pairs[key]
                    delta = (b.float() @ a.float()) * scale
                    if delta.shape != w.shape:
                        raise ValueError(f"{key}: LoRA delta {tuple(delta.shape)} != weight {tuple(w.shape)}")
                    w = (w.float() + delta).to(w.dtype)
                    merged.add(key)
                tensors[key] = w.contiguous()
        save_file(tensors, str(out_dir / shard.name), metadata=metadata or {"format": "pt"})
    unused = set(pairs) - merged
    if unused:
        raise ValueError(f"{len(unused)} adapter weights matched no base weight, e.g. {sorted(unused)[:3]}")
    # config, tokenizer, chat template, shard index: copied unchanged from the base model
    for p in base_dir.iterdir():
        if p.is_file() and p.suffix != ".safetensors" and not p.name.startswith("."):
            shutil.copy(p, out_dir / p.name)
    return {"merged_weights": len(merged), "scale": scale, "r": r, "lora_alpha": alpha}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model folder or HF repo id")
    ap.add_argument("--adapter", required=True, help="LoRA adapter folder or HF repo id")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def local(path_or_repo: str) -> Path:
        p = Path(path_or_repo)
        if p.exists():
            return p
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(path_or_repo))

    info = merge(local(a.base), local(a.adapter), Path(a.out))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
