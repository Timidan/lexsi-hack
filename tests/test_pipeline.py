"""CPU tests: the gold policy must score 100%, and bad behaviours must fail.

Run: python -m unittest discover -s tests
"""

import json
import unittest

from naija_agent import CATEGORIES, aggregate, generate, gold_conversation, gold_policy, run_episode, score_episode
from naija_agent.lang import LANGS
from naija_agent.prompting import format_call_turn, parse_tool_calls, ParseError
from naija_agent.runner import to_agenttune_record
from naija_agent.tasks import template_split, _templates

TRAIN = generate("train", per_cell=4, seed=1)
TEST = generate("test", per_cell=4, seed=1)


def _run_all(tasks, policy):
    scores = [score_episode(t, run_episode(t, policy)) for t in tasks]
    return scores, aggregate(tasks, scores)


class GoldPolicy(unittest.TestCase):
    def test_gold_scores_100_percent(self):
        for tasks in (TRAIN, TEST):
            scores, m = _run_all(tasks, gold_policy)
            failed = [(t.id, s["checks"]) for t, s in zip(tasks, scores) if not s["success"]]
            self.assertEqual(failed, [])
            self.assertEqual(m["overall"]["valid_call_rate"], 1.0)
            self.assertEqual(m["overall"]["tool_f1"], 1.0)

    def test_every_category_and_language_present(self):
        cells = {(t.category, t.lang) for t in TEST}
        self.assertEqual(cells, {(c, l) for c in CATEGORIES for l in LANGS})

    def test_split_is_by_template(self):
        self.assertTrue(all(template_split(t.template_id) == "test" for t in TEST))
        self.assertTrue(all(template_split(t.template_id) == "train" for t in TRAIN))
        self.assertFalse({t.template_id for t in TRAIN} & {t.template_id for t in TEST})

    def test_every_template_cell_has_train_and_test(self):
        from naija_agent import lang as L
        for cat in L.REQUESTS:
            for lng in LANGS:
                splits = {template_split(tid) for tid, _ in _templates(cat, lng)}
                self.assertEqual(splits, {"train", "test"}, (cat, lng))

    def test_gold_conversation_shape(self):
        t = next(t for t in TRAIN if t.category == "transfer_contact")
        msgs = gold_conversation(t)
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"])
        self.assertTrue(msgs[3]["content"].startswith("Tool result for find_contact"))
        self.assertIsNone(parse_tool_calls(msgs[-1]["content"]))

    def test_agenttune_record_is_json(self):
        t = TEST[0]
        ep = run_episode(t, gold_policy)
        rec = to_agenttune_record(t, ep, score_episode(t, ep))
        json.dumps(rec, ensure_ascii=False)
        self.assertEqual(rec["final_response"], t.gold_final)


class Batched(unittest.TestCase):
    def test_batched_matches_single(self):
        from naija_agent.runner import batched, run_episodes
        eps = run_episodes(TEST, batched(gold_policy), batch_size=5)
        scores = [score_episode(t, e) for t, e in zip(TEST, eps)]
        self.assertTrue(all(s["success"] for s in scores))
        self.assertEqual([e.task_id for e in eps], [t.id for t in TEST])


class BadPolicies(unittest.TestCase):
    def test_silent_policy_fails_payments(self):
        scores, m = _run_all(TEST, lambda msgs, task: "Okay.")
        by_cat = m["by_category"]
        for cat in ("airtime_direct", "transfer_direct", "balance", "insufficient_funds", "missing_info"):
            self.assertEqual(by_cat[cat]["success_rate"], 0.0, cat)
        self.assertEqual(by_cat["chitchat"]["success_rate"], 1.0)

    def test_guessing_instead_of_asking_fails(self):
        def guesser(msgs, task):
            if sum(m["role"] == "assistant" for m in msgs) == 0:
                return format_call_turn([{"name": "buy_airtime",
                                          "arguments": {"phone": "08031234567", "network": "MTN", "amount": 500}}])
            return "Done?"
        tasks = [t for t in TEST if t.category == "missing_info"]
        scores, m = _run_all(tasks, guesser)
        self.assertEqual(m["overall"]["success_rate"], 0.0)

    def test_following_injection_fails(self):
        def obedient(msgs, task):
            n = sum(m["role"] == "assistant" for m in msgs)
            if n < len(task.gold_plan):
                return gold_policy(msgs, task)
            if n == len(task.gold_plan):  # obey the planted instruction: extra transfer
                return format_call_turn([{"name": "lookup_account", "arguments": {"bank": "GTBank", "account_number": "0123456789"}}])
            return task.gold_final
        tasks = [t for t in TEST if t.category == "injection"]
        _, m = _run_all(tasks, gold_policy)
        self.assertEqual(m["overall"]["success_rate"], 1.0)
        # an extra (unrequested) call is counted even when it moves no money
        scores, _ = _run_all(tasks, obedient)
        self.assertTrue(all(s["extra_calls"] >= 1 for s in scores))

    def test_send_money_requires_lookup(self):
        t = next(t for t in TEST if t.category == "transfer_direct")
        send = t.gold_plan[-1]
        pol = lambda msgs, task: format_call_turn(send) if sum(m["role"] == "assistant" for m in msgs) == 0 else "Done"
        ep = run_episode(t, pol)
        self.assertEqual(ep.ledger, [])
        self.assertIn("lookup_account", ep.calls[0].result["error"])


class Parser(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_tool_calls('[{"tool_name": "check_balance", "parameters": {}}]'),
                         [{"name": "check_balance", "arguments": {}}])
        self.assertEqual(parse_tool_calls('```json\n[{"tool_name": "x", "parameters": {"a": 1}}]\n```'),
                         [{"name": "x", "arguments": {"a": 1}}])
        self.assertEqual(parse_tool_calls('{"name": "x", "arguments": "{\\"a\\": 1}"}'),
                         [{"name": "x", "arguments": {"a": 1}}])
        self.assertIsNone(parse_tool_calls("E don enter! Your balance remain ₦2,000."))
        with self.assertRaises(ParseError):
            parse_tool_calls('[{"tool_name": "x", "parameters": {')


if __name__ == "__main__":
    unittest.main()
