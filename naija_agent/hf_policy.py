"""Batched Hugging Face policy for Tiny Aya (base model or base + LoRA adapter).

Needs torch, transformers (and peft for adapters, bitsandbytes for 4-bit).
Works on a free Kaggle/Colab T4: fp16 for inference, or 4-bit if memory is tight.
"""

from __future__ import annotations

DEFAULT_MODEL = "CohereLabs/tiny-aya-global"


class HFPolicy:
    def __init__(self, model_id: str = DEFAULT_MODEL, adapter: str | None = None,
                 load_in_4bit: bool = False, max_new_tokens: int = 256, device: str | None = None,
                 temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # T4 has no fast bf16, so use fp16 there; bf16 on Ampere and newer (L4, A100).
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
        self.dtype = torch.bfloat16 if bf16_ok else torch.float16
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # one whole model per GPU (default: the first GPU); "auto" would split it across GPUs and run slower
        device_map = {"": device} if device else ({"": 0} if torch.cuda.is_available() else "cpu")
        kwargs = {"device_map": device_map, "dtype": self.dtype}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from .compat import remove_old_torchao

            remove_old_torchao()
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            if not load_in_4bit:
                self.model = self.model.merge_and_unload()   # same outputs, faster generation
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # 0 = greedy (evaluation); above 0 samples, used to find the model's likely mistakes (scripts/make_pairs.py)
        self.sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}

    def render(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, batch: list[list[dict]], tasks=None) -> list[str]:
        """Generate one reply per conversation. If a batch does not fit in GPU
        memory, it is split in half and retried (down to single conversations)."""
        import torch

        try:
            return self._generate(batch)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            mid = len(batch) // 2
            print(f"GPU out of memory with {len(batch)} conversations; retrying as {mid} + {len(batch) - mid}",
                  flush=True)
            return self(batch[:mid]) + self(batch[mid:])

    def _generate(self, batch: list[list[dict]]) -> list[str]:
        import torch

        texts = [self.render(m) for m in batch]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, **self.sampling,
                                      pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]
