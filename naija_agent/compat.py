"""Environment checks for free notebook GPUs (Kaggle, Colab)."""

from __future__ import annotations

TORCHAO_MIN = "0.16.0"   # recent PEFT raises ImportError if an older torchao is installed


def check_torchao() -> None:
    """Kaggle images ship torchao 0.10, and recent PEFT refuses to load LoRA next to a torchao older than 0.16.
    We do not use torchao; the notebooks uninstall it in their setup cell. Library code only checks, and stops
    with a clear message instead of PEFT's ImportError. Call before the first PEFT import."""
    import importlib.metadata as md

    try:
        version = md.version("torchao")
    except md.PackageNotFoundError:
        return
    from packaging.version import Version

    if Version(version) < Version(TORCHAO_MIN):
        raise RuntimeError(f"torchao {version} is installed, and PEFT cannot load a LoRA adapter next to a torchao "
                           f"older than {TORCHAO_MIN}. Oya does not use torchao: run `pip uninstall -y torchao`, "
                           f"then restart Python.")
