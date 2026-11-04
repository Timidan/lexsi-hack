"""Publish run progress and results to the repo's `results` branch.

Kaggle and Colab logs are not reachable from outside, so runs push small
files (progress.json, train_log.json, metrics, trajectories) to GitHub as they
go. On by default only inside Kaggle or Colab notebooks, when a GITHUB_TOKEN
with write access is available there. Anywhere else (a laptop, CI, a dev
container) it is off unless OYA_PUBLISH=1; OYA_PUBLISH=0 turns it off
everywhere. Model weights are never pushed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO = os.environ.get("OYA_REPO", "Timidan/lexsi-hack")
BRANCH = "results"


def _in_notebook_gpu_service() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return True
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    return False


def publishing_wanted() -> bool:
    flag = os.environ.get("OYA_PUBLISH")
    if flag == "0":
        return False
    return flag == "1" or _in_notebook_gpu_service()


def _secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        pass
    return os.environ.get(name)


class Publisher:
    """Copies files into runs/<name>/ on the results branch and pushes, at most
    once per ``min_interval`` seconds unless ``force=True``. Never raises."""

    def __init__(self, name: str, token: str | None = None, min_interval: float = 600):
        self.name = name
        self.min_interval = min_interval
        self.last = 0.0
        self.token = (token or _secret("GITHUB_TOKEN")) if publishing_wanted() else None
        self.dir: str | None = None
        self.error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.token) and self.error is None

    def _git(self, *args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=cwd or self.dir, capture_output=True, text=True, timeout=120)

    def _scrub(self, text: str) -> str:
        return text.replace(self.token or "\0", "***")

    def _ensure_checkout(self) -> None:
        if self.dir:
            return
        self.dir = tempfile.mkdtemp(prefix="oya-results-")
        url = os.environ.get("OYA_REMOTE_URL") or f"https://x-access-token:{self.token}@github.com/{REPO}.git"
        if self._git("clone", "-q", "--depth", "1", "-b", BRANCH, url, self.dir, cwd=".").returncode != 0:
            self._git("init", "-q")
            self._git("checkout", "-q", "-b", BRANCH)
            self._git("remote", "add", "origin", url)

    def publish(self, files: dict[str, str | Path], message: str, force: bool = False) -> bool:
        """files: {name inside runs/<run>/: local path (file or folder)}."""
        if not self.enabled or (not force and time.time() - self.last < self.min_interval):
            return False
        try:
            self._ensure_checkout()
            target = Path(self.dir) / "runs" / self.name
            target.mkdir(parents=True, exist_ok=True)
            for dst, src in files.items():
                src = Path(src)
                if src.is_dir():
                    shutil.copytree(src, target / dst, dirs_exist_ok=True)
                elif src.exists():
                    shutil.copy(src, target / dst)
            self._git("add", "-A")
            ident = ["-c", "user.name=oya-runner", "-c", "user.email=oya-runner@users.noreply.github.com"]
            if self._git(*ident, "commit", "-q", "-m", message).returncode != 0:
                return False  # nothing changed
            for _ in range(2):
                r = self._git("push", "-q", "origin", BRANCH)
                if r.returncode == 0:
                    self.last = time.time()
                    return True
                self._git(*ident, "pull", "-q", "--rebase", "origin", BRANCH)
            print("results push failed:", self._scrub(r.stderr.strip())[:300])
            if "403" in r.stderr or "denied" in r.stderr.lower():
                self.error = "no write access"
            return False
        except Exception as e:  # never break a training run because of publishing
            print("results push error:", self._scrub(str(e))[:300])
            return False
