"""Run one task with any policy and record the episode.

A policy is ``policy(messages, task) -> str``: given the chat so far (system,
user, assistant, ...) it returns the next assistant message. The Hugging Face
policy ignores ``task``; the gold policy uses it to replay the reference plan.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from .prompting import ParseError, format_call_turn, format_result_turn, parse_tool_calls, system_message
from .tasks import Task
from .world import World, validate_args, SCHEMA_BY_NAME

Policy = Callable[[list[dict], Task], str]
MAX_TURNS = 6


@dataclass
class CallRecord:
    turn: int
    name: str | None
    arguments: object
    valid: bool          # parsed, known tool, arguments pass the schema
    result: dict


@dataclass
class Episode:
    task_id: str
    messages: list[dict]
    calls: list[CallRecord] = field(default_factory=list)
    final: str | None = None
    ledger: list[dict] = field(default_factory=list)
    turns: int = 0
    parse_errors: int = 0


def initial_messages(prompt: str) -> list[dict]:
    return [{"role": "system", "content": system_message()}, {"role": "user", "content": prompt}]


class _Live:
    """An episode in progress."""

    def __init__(self, task: Task):
        self.task = task
        self.world = World.from_dict(task.world)
        self.ep = Episode(task_id=task.id, messages=initial_messages(task.prompt))
        self.done = False

    def step(self, reply: str, max_turns: int) -> None:
        ep, turn = self.ep, self.ep.turns
        ep.turns += 1
        ep.messages.append({"role": "assistant", "content": reply})
        try:
            calls = parse_tool_calls(reply)
        except ParseError as e:
            ep.parse_errors += 1
            res = {"status": "failed", "error": f"could not parse tool call ({e}). Reply with only a JSON list."}
            ep.calls.append(CallRecord(turn, None, reply, False, res))
            ep.messages.append({"role": "user", "content": format_result_turn([("unknown", res)])})
            calls = []
        else:
            if calls is None:
                ep.final = reply.strip()
                self._finish()
                return
            results = []
            for c in calls:
                valid = (c["name"] in SCHEMA_BY_NAME and isinstance(c["arguments"], dict)
                         and validate_args(c["name"], c["arguments"]) is None)
                res = self.world.execute(c["name"], c["arguments"])
                ep.calls.append(CallRecord(turn, c["name"], c["arguments"], valid, res))
                results.append((c["name"], res))
            ep.messages.append({"role": "user", "content": format_result_turn(results)})
        if ep.turns >= max_turns:
            self._finish()

    def _finish(self) -> None:
        self.ep.ledger = self.world.ledger
        self.done = True


def run_episode(task: Task, policy: Policy, max_turns: int = MAX_TURNS) -> Episode:
    live = _Live(task)
    while not live.done:
        live.step(policy(live.ep.messages, task), max_turns)
    return live.ep


BatchPolicy = Callable[[list[list[dict]], list[Task]], list[str]]


def run_episodes(tasks: list[Task], batch_policy: BatchPolicy, batch_size: int = 8,
                 max_turns: int = MAX_TURNS, progress: Callable[[int, int], None] | None = None) -> list[Episode]:
    """Run many tasks, sending up to ``batch_size`` live conversations to the
    model at once (one generate() call per round). Returns episodes in task order."""
    pending = [_Live(t) for t in tasks]
    queue, active, finished = list(pending), [], 0
    while queue or active:
        while queue and len(active) < batch_size:
            active.append(queue.pop(0))
        replies = batch_policy([lv.ep.messages for lv in active], [lv.task for lv in active])
        for lv, reply in zip(active, replies):
            lv.step(reply, max_turns)
        finished += sum(lv.done for lv in active)
        active = [lv for lv in active if not lv.done]
        if progress:
            progress(finished, len(pending))
    return [lv.ep for lv in pending]


def batched(policy: Policy) -> BatchPolicy:
    """Wrap a one-at-a-time policy for run_episodes."""
    return lambda batch, tasks: [policy(m, t) for m, t in zip(batch, tasks)]


def gold_policy(messages: list[dict], task: Task) -> str:
    """Replays the task's reference plan, then gives the reference reply."""
    done = sum(1 for m in messages if m["role"] == "assistant")
    if done < len(task.gold_plan):
        return format_call_turn(task.gold_plan[done])
    return task.gold_final


def gold_conversation(task: Task) -> list[dict]:
    """The full reference chat for supervised fine-tuning (system, user, assistant, ...)."""
    return run_episode(task, gold_policy).messages


# ─── AgentTune trajectory record (read by AuditKIT's episodes_from_agenttune) ──

def to_agenttune_record(task: Task, ep: Episode, score: dict | None = None) -> dict:
    """Same shape as agenttune.agentic.trajectory.dataset.Trajectory.to_dict()."""
    steps = []
    by_turn: dict[int, list[CallRecord]] = {}
    for c in ep.calls:
        by_turn.setdefault(c.turn, []).append(c)
    assistant_msgs = [m["content"] for m in ep.messages if m["role"] == "assistant"]
    for t, text in enumerate(assistant_msgs):
        calls = by_turn.get(t, [])
        if calls:
            action = {"tool_calls": [{"id": f"call_{t}_{i}", "name": c.name, "arguments": c.arguments}
                                     for i, c in enumerate(calls)]}
            observation = json.dumps([c.result for c in calls], ensure_ascii=False)
        else:
            action, observation = {}, text
        steps.append({"step_number": t, "state": "", "action": action, "observation": observation,
                      "thought": text, "reward": None})
    return {
        "task": task.prompt,
        "steps": steps,
        "reward": (score or {}).get("success", 0.0),
        "trajectory_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{task.id}-{len(ep.messages)}")),
        "final_response": ep.final or "",
        "metadata": {"task_id": task.id, "category": task.category, "lang": task.lang,
                     "template_id": task.template_id, "ledger": ep.ledger, "score": score or {}},
    }
