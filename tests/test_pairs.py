"""DPO pairs: a wrong step becomes a pair, the reference step does not, a different lookup is skipped."""

import unittest

from naija_agent.pairs import is_mistake, make_pairs, positions
from naija_agent.prompting import format_call_turn
from naija_agent.tasks import generate


class Pairs(unittest.TestCase):
    def test_mistakes_become_pairs(self):
        tasks = generate("train", 1, seed=5)
        ask = next(t for t in tasks if t.category == "missing_info")
        prefix, gold = positions(ask)[0]
        guess = format_call_turn([{"name": "check_balance", "arguments": {}}])
        self.assertFalse(is_mistake(ask, gold, gold))
        self.assertTrue(is_mistake(ask, gold, guess))           # must ask, not act
        pairs = make_pairs(ask, prefix, gold, [gold, guess, guess])
        self.assertEqual([p["rejected"][0]["content"] for p in pairs], [guess])

        contact = next(t for t in tasks if t.category == "airtime_contact")
        _, pay = positions(contact)[1]
        lookup = format_call_turn([{"name": "check_balance", "arguments": {}}])
        self.assertIsNone(is_mistake(contact, pay, lookup))     # another route, not clearly wrong


if __name__ == "__main__":
    unittest.main()
