"""Evaluate a batch policy on a task file and write the run folder."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .prompting import prompt_style
from .runner import BatchPolicy, run_episodes, to_agenttune_record
from .scoring import aggregate, format_report, score_episode
from .tasks import Task


def load_tasks(path: str | Path, limit: int | None = None) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        tasks = [Task(**json.loads(line)) for line in f if line.strip()]
    return tasks[:limit] if limit else tasks


def evaluate(tasks: list[Task], policy: BatchPolicy, out_dir: str | Path, batch_size: int = 8,
             config: dict | None = None, verbose: bool = True, publish: bool = True) -> dict:
    """Run, score and save: trajectories.jsonl (AgentTune format), scores.jsonl,
    metrics.json, report.txt. Returns the metrics dict. With publish=True and a
    GITHUB_TOKEN available, progress and results also go to the results branch."""
    from .publish import Publisher

    t0, last = time.time(), [0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pub = Publisher(out.name) if publish else None
    progress_path = out / "progress.json"

    def progress(done, total):
        if verbose and (done - last[0] >= max(1, total // 20) or done == total):
            last[0] = done
            print(f"{done}/{total} tasks done ({time.time() - t0:.0f}s)", flush=True)
            if pub:
                progress_path.write_text(json.dumps({"phase": "evaluation", "done": done, "total": total,
                                                     "elapsed_s": round(time.time() - t0),
                                                     "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
                pub.publish({"progress.json": progress_path}, f"{out.name}: evaluation progress")

    episodes = run_episodes(tasks, policy, batch_size=batch_size, progress=progress)
    scores = [score_episode(t, e) for t, e in zip(tasks, episodes)]
    metrics = aggregate(tasks, scores)
    metrics["config"] = {**(config or {}), "prompt_style": prompt_style(), "n_tasks": len(tasks),
                         "seconds": round(time.time() - t0, 1)}

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
    if pub and pub.publish({".": out}, f"{out.name}: evaluation results", force=True):
        print(f"results pushed to the 'results' branch under runs/{out.name}/")
    return metrics


def evaluate_parallel(tasks: list[Task], jobs: list[dict], batch_size: int = 8) -> dict[str, dict]:
    """Evaluate several models on the same tasks, one model per GPU at the same time.

    jobs: [{"name", "make_policy": callable(device) -> BatchPolicy, "out_dir", "config"}].
    Models are loaded one after another (loading in parallel threads is not safe),
    then evaluated in parallel threads. With fewer GPUs than jobs, everything runs
    one job at a time on the first GPU. Returns {name: metrics}."""
    import gc
    import threading

    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except ImportError:
        torch, n_gpu = None, 0

    def free() -> None:
        gc.collect()
        if torch is not None and n_gpu:
            torch.cuda.empty_cache()

    results: dict[str, dict] = {}
    if n_gpu >= len(jobs) > 1:
        print(f"evaluating {[j['name'] for j in jobs]} in parallel, one GPU each", flush=True)
        policies = [j["make_policy"](f"cuda:{i}") for i, j in enumerate(jobs)]
        errors: dict[str, BaseException] = {}

        def run(i: int) -> None:
            j = jobs[i]
            try:
                results[j["name"]] = evaluate(tasks, policies[i], j["out_dir"], batch_size=batch_size,
                                              config=j.get("config"), verbose=(i == 0))
            except BaseException as e:  # reported after all jobs finish
                errors[j["name"]] = e

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        del policies
        free()
        for j in jobs[1:]:
            if j["name"] in results:
                print((Path(j["out_dir"]) / "report.txt").read_text(), flush=True)
        if errors:
            raise RuntimeError(f"evaluation failed for {list(errors)}: {errors}")
        return results
    for j in jobs:
        policy = j["make_policy"]("cuda:0" if n_gpu else None)
        results[j["name"]] = evaluate(tasks, policy, j["out_dir"], batch_size=batch_size, config=j.get("config"))
        del policy
        free()
    return results
