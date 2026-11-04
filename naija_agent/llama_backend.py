"""Run the quantized (GGUF) agent on CPU with llama-cpp-python.

Used by the Hugging Face Space and by notebook 03's smoke test, so both
exercise the same code path.
"""

from __future__ import annotations

import os
from datetime import datetime

STOP = ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"]


class LlamaChat:
    def __init__(self, model_path: str | None = None, repo_id: str | None = None, filename: str | None = None,
                 n_ctx: int = 4096, n_threads: int | None = None, max_tokens: int = 256):
        from llama_cpp import Llama

        kw = {"n_ctx": n_ctx, "n_threads": n_threads or os.cpu_count() or 2, "verbose": False}
        if model_path:
            self.llm = Llama(model_path=model_path, **kw)
        else:
            self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename or "*.gguf", **kw)
        self.max_tokens = max_tokens
        self._template = None
        self.mode = "chat_completion"

    def _render(self, messages: list[dict]) -> str:
        """Render with the chat template stored in the GGUF (fallback path).
        BOS is left to the tokenizer, which adds it."""
        if self._template is None:
            import json

            import jinja2.ext
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            # same Jinja setup as transformers' apply_chat_template (Tiny Aya's template uses {% break %})
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[jinja2.ext.loopcontrols, jinja2.ext.do])

            def raise_exception(msg):
                raise ValueError(msg)

            env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
                x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
            env.globals.update(raise_exception=raise_exception, strftime_now=lambda f: datetime.now().strftime(f))
            self._template = env.from_string(self.llm.metadata["tokenizer.chat_template"])
        return self._template.render(messages=messages, add_generation_prompt=True, bos_token="", eos_token="")

    def reply(self, messages: list[dict]) -> str:
        if self.mode == "chat_completion":
            try:
                out = self.llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=self.max_tokens)
                return (out["choices"][0]["message"]["content"] or "").strip()
            except Exception as e:  # template features llama-cpp-python's formatter lacks
                print(f"chat completion failed ({e}); rendering the chat template directly from now on")
                self.mode = "template"
        out = self.llm.create_completion(self._render(messages), max_tokens=self.max_tokens, temperature=0.0,
                                         stop=STOP)
        return out["choices"][0]["text"].strip()

    def __call__(self, batch: list[list[dict]], tasks=None) -> list[str]:
        """Batch-policy interface (naija_agent.runner), one conversation at a time."""
        return [self.reply(m) for m in batch]
