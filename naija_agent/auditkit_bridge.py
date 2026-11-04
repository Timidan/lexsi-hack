"""Score a run folder with Lexsi Labs' AuditKIT (pip install auditkit).

Our own scorer (scoring.py) is the primary metric. AuditKIT gives an
independent second opinion with its standard agent metrics: a verified
final-state outcome plus tool-call F1 and validity against the gold route.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tasks import Task
from .world import TOOL_NAMES


def _ledger_key(ledger: list[dict]) -> str:
    return json.dumps(sorted(json.dumps(e, sort_keys=True) for e in ledger))


def auditkit_eval(tasks: list[Task], run_dir: str | Path,
                  scorers: tuple[str, ...] = ("tool_call_f1", "tool_call_validity")):
    from auditkit.agent_eval import (AgentCase, AgentEvalRunner, AgentEvalSpec, FinalStateAssertion,
                                     episodes_from_agenttune)

    path = Path(run_dir) / "trajectories.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes = episodes_from_agenttune(str(path))
    by_id = {t.id: t for t in tasks}
    cases = []
    for ep, rec in zip(episodes, records, strict=True):
        task = by_id[rec["metadata"]["task_id"]]
        ep.case_id = task.id
        ep.final_state = {"ledger": _ledger_key(rec["metadata"]["ledger"])}
        cases.append(AgentCase(
            id=task.id, task=task.prompt, category=task.category, allowed_tools=list(TOOL_NAMES),
            reference_turns=[[{"name": c["name"], "arguments": c["arguments"]} for c in turn]
                             for turn in task.gold_plan],
            outcome=FinalStateAssertion("ledger", equals=_ledger_key(task.expected["ledger"])),
        ))
    spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes, scorers=list(scorers), trials=1)
    return AgentEvalRunner().run(spec)
