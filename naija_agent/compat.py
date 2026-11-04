"""Environment fixes for free notebook GPUs (Kaggle, Colab)."""

from __future__ import annotations

import importlib
import subprocess
import sys

TORCHAO_MIN = "0.16.0"   # recent PEFT raises ImportError if an older torchao is installed


def remove_old_torchao() -> None:
    """Kaggle images ship torchao 0.10, and recent PEFT refuses to load LoRA next
    to a torchao older than 0.16. We do not use torchao, so uninstall it.
    Call before the first PEFT import."""
    import importlib.metadata as md

    try:
        version = md.version("torchao")
    except md.PackageNotFoundError:
        return
    from packaging.version import Version

    if Version(version) >= Version(TORCHAO_MIN):
        return
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    importlib.invalidate_caches()
    try:
        md.version("torchao")
        print(f"warning: could not remove torchao {version}; PEFT may fail to load")
    except md.PackageNotFoundError:
        print(f"removed torchao {version} (older than {TORCHAO_MIN}, incompatible with PEFT, not used here)")
