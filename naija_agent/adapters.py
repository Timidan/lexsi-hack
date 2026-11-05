"""Locate a trained LoRA adapter: a local folder, a Kaggle input (the output of an
earlier notebook run attached with "Add Input"), or a Hugging Face repo."""

from __future__ import annotations

import glob
import os


def find_adapter(run_name: str, hf_repo: str | None = None, token: str | None = None) -> str:
    """Return a local folder holding adapter_config.json for `run_name`.

    Search order: outputs/<run_name> in this notebook, then any attached Kaggle
    input containing .../outputs/<run_name>/, then the Hugging Face repo."""
    local = os.path.join("outputs", run_name)
    if os.path.exists(os.path.join(local, "adapter_config.json")):
        return local
    hits = sorted(glob.glob(f"/kaggle/input/**/outputs/{run_name}/adapter_config.json", recursive=True))
    if hits:
        return os.path.dirname(hits[-1])
    if hf_repo:
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_repo, token=token)
    raise FileNotFoundError(
        f"No adapter for {run_name!r}. Attach the earlier training notebook's output (Kaggle: Add Input -> "
        f"Your Work -> the 02_train_and_eval notebook) or upload the adapter to Hugging Face.")
