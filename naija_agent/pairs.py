"""Preference pairs for DPO from the model's own mistakes, one step at a time.

For every assistant turn of a task's reference chat, the model is sampled a few times given the chat so far
(the reference turns and the real tool results). Each sample is judged against the reference turn with the
scorer's own rules; a mistake becomes a pair: prompt = the chat so far, chosen = the reference turn,
rejected = the model's turn. A sample that only takes a different lookup route is not clearly wrong and is
skipped.
"""

from __future__ import annotations

import json

from .lang import REPLY_LANG
from .prompting import ParseError, parse_tool_calls
from .runner import gold_conversation
from .scoring import _looks_yoruba, reply_ok
from .tasks import Task
from .world import PAYMENT_TOOLS, SCHEMA_BY_NAME

LOOKUP_TOOLS = set(SCHEMA_BY_NAME) - PAYMENT_TOOLS


def _canon(calls: list[dict]) -> list[str]:
    return sorted(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls)


def is_mistake(task: Task, gold_turn: str, sample: str) -> bool | None:
    """True if `sample` is wrong where the reference wrote `gold_turn`, False if it matches,
    None if it calls different lookup tools (not clearly wrong, so no pair is made)."""
    try:
        got = parse_tool_calls(sample)
    except ParseError:
        return True                                   # a broken tool call
    want = parse_tool_calls(gold_turn)
    if want is None:                                  # the reference answers or asks here (always the last turn)
        if got is not None:
            return True                               # acted instead of asking or answering
        answers = task.expected.get("answer_any")
        return not reply_ok(task, sample) or bool(answers) and not any(v in sample for v in answers)
    if got is None:
        return True                                   # answered in text where a tool call was needed
    if _canon(got) == _canon(want):
        return False
    if all(c["name"] in LOOKUP_TOOLS for c in got):
        return None
    return True                                       # a wrong, extra or unknown payment call


def wrong_language(pair: dict) -> bool:
    """The rejected turn is a reply in the wrong language. v2 makes this mistake almost only one way (a Yoruba
    reply to a Pidgin request: 75 of the first 222 pairs, against 1), and training on those taught v3 to avoid
    Yoruba replies altogether, so these pairs are left out."""
    text = pair["rejected"][0]["content"]
    try:
        if parse_tool_calls(text) is not None:
            return False
    except ParseError:
        return False
    return _looks_yoruba(text) != (REPLY_LANG[pair["lang"]] == "yo")


def positions(task: Task) -> list[tuple[list[dict], str]]:
    """(chat so far, reference turn) for every assistant turn of the task's reference chat."""
    conv = gold_conversation(task)
    return [(conv[:i], m["content"]) for i, m in enumerate(conv) if m["role"] == "assistant"]


def make_pairs(task: Task, prefix: list[dict], gold_turn: str, samples: list[str], limit: int = 2) -> list[dict]:
    """Up to `limit` distinct mistakes among `samples`, as DPO pairs in TRL's conversational format."""
    out, seen = [], set()
    for s in (s.strip() for s in samples):
        if s in seen or is_mistake(task, gold_turn, s) is not True:
            continue
        seen.add(s)
        out.append({"task_id": task.id, "category": task.category, "lang": task.lang,
                    "turn": sum(m["role"] == "assistant" for m in prefix), "prompt": prefix,
                    "chosen": [{"role": "assistant", "content": gold_turn}],
                    "rejected": [{"role": "assistant", "content": s}]})
        if len(out) == limit:
            break
    return out
