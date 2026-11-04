"""Evaluate a batch policy on a task file and write the run folder."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .runner import BatchPolicy, run_episodes, to_agenttune_record
from .scoring import aggregate, format_report, score_episode
from .tasks import Task


def load_tasks(path: str | Path, limit: int | None = None) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        tasks = [Task(**json.loads(line)) for line in f if line.strip()]
    return tasks[:limit] if limit else tasks


def evaluate(tasks: list[Task], policy: BatchPolicy, out_dir: str | Path, batch_size: int = 8,
             config: dict | None = None, verbose: bool = True) -> dict:
    """Run, score and save: trajectories.jsonl (AgentTune format), scores.jsonl,
    metrics.json, report.txt. Returns the metrics dict."""
    t0, last = time.time(), [0]

    def progress(done, total):
        if verbose and (done - last[0] >= max(1, total // 20) or done == total):
            last[0] = done
            print(f"{done}/{total} tasks done ({time.time() - t0:.0f}s)", flush=True)

    episodes = run_episodes(tasks, policy, batch_size=batch_size, progress=progress)
    scores = [score_episode(t, e) for t, e in zip(tasks, episodes)]
    metrics = aggregate(tasks, scores)
    metrics["config"] = {**(config or {}), "n_tasks": len(tasks), "seconds": round(time.time() - t0, 1)}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as f:
        for t, e, s in zip(tasks, episodes, scores):
            f.write(json.dumps(to_agenttune_record(t, e, s), ensure_ascii=False, default=str) + "\n")
    with open(out / "scores.jsonl", "w", encoding="utf-8") as f:
        for t, e, s in zip(tasks, episodes, scores):
            f.write(json.dumps({"task_id": t.id, "category": t.category, "lang": t.lang, "prompt": t.prompt,
                                "final": e.final, **s}, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = format_report(metrics, title=f"== {out} ==")
    (out / "report.txt").write_text(report + "\n")
    if verbose:
        print(report)
    return metrics
