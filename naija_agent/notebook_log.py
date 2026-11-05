"""Mirror a notebook's output to GitHub so runs can be followed from outside Kaggle/Colab.

    from naija_agent.notebook_log import start_live_log
    start_live_log("sft-v2")

Copies everything printed in the notebook (including subprocess output relayed
through print, and tracebacks of failing cells) to runs/<name>/notebook_log.txt,
and pushes it to the repo's `results` branch every few minutes, at once when a
cell fails, and at the end of every cell after the first push. Uses the same
Publisher as the runs, so it is only active inside Kaggle/Colab with a
GITHUB_TOKEN that has write access.
"""

from __future__ import annotations

import atexit
import sys
import threading
import time
import traceback
from pathlib import Path

from .publish import Publisher

_flush = None


def flush_live_log() -> None:
    """Push the log now (call at the end of a notebook)."""
    if _flush is not None:
        _flush()


class _Tee:
    def __init__(self, stream, path: Path, lock: threading.Lock):
        self._stream, self._path, self._lock = stream, path, lock

    def write(self, text):
        n = self._stream.write(text)
        if text:
            with self._lock, open(self._path, "a", encoding="utf-8") as f:
                f.write(text)
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_live_log(name: str, every_s: float = 300) -> Publisher | None:
    # kept outside runs/<name>/ so publishing a run folder never includes (and races on) the log
    out = Path("logs") / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "notebook_log.txt"
    path.write_text(f"=== notebook log for {name}, started {time.strftime('%Y-%m-%d %H:%M:%S')} UTC ===\n",
                    encoding="utf-8")
    pub = Publisher(name, min_interval=60)
    if not pub.enabled:
        print("live notebook log: off (needs GITHUB_TOKEN with write access, inside Kaggle/Colab)")
        return None
    lock = threading.Lock()
    sys.stdout = _Tee(sys.stdout, path, lock)
    sys.stderr = _Tee(sys.stderr, path, lock)

    push_lock = threading.Lock()   # the timer thread and the cell hook share one git checkout

    def push(force: bool = False, why: str = "notebook log") -> None:
        with push_lock:
            pub.publish({"notebook_log.txt": path}, f"{name}: {why}", force=force)

    def loop() -> None:
        while True:
            time.sleep(every_s)
            push(force=True)

    threading.Thread(target=loop, daemon=True).start()
    global _flush
    _flush = lambda: push(force=True, why="notebook finished")  # noqa: E731
    atexit.register(_flush)
    try:
        ip = get_ipython()  # noqa: F821 (defined inside IPython/Jupyter)

        def after_cell(result) -> None:
            err = result.error_in_exec or result.error_before_exec
            if err is not None:
                with lock, open(path, "a", encoding="utf-8") as f:
                    f.write("\n=== CELL FAILED ===\n" + "".join(
                        traceback.format_exception(type(err), err, err.__traceback__)))
                push(force=True, why="cell failed")

        ip.events.register("post_run_cell", after_cell)
    except NameError:
        pass
    push(force=True, why="notebook started")
    print(f"live notebook log -> GitHub 'results' branch: runs/{name}/notebook_log.txt")
    return pub
