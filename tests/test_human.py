"""Human-written test set: parsing, task building and scoring (gold must pass, a silent agent must not)."""

import unittest

from naija_agent.human import SCENARIOS, build_task, parse_requests
from naija_agent.runner import gold_policy, run_episode
from naija_agent.scoring import reply_ok, score_episode

SAMPLE = "\n".join(
    [f"{k}Y: yoruba request {k}" for k in SCENARIOS] + [f"{k}P: pidgin request {k}" for k in SCENARIOS]
    + ["# a comment", "6Y:", "99Y: unknown scenario"])


class HumanSet(unittest.TestCase):
    def test_parse(self):
        rows = parse_requests(SAMPLE)
        self.assertEqual(len(rows), 2 * len(SCENARIOS))
        self.assertIn((7, "pcm", "pidgin request 7"), rows)

    def test_gold_passes_and_silence_fails(self):
        for k in SCENARIOS:
            for variant in (("", "injection") if k in (6, 7) else ("",)):
                t = build_task(k, "yo", f"request {k}", variant)
                self.assertTrue(score_episode(t, run_episode(t, gold_policy))["success"], t.id)
                silent = score_episode(t, run_episode(t, lambda m, task: "Okay."))["success"]
                self.assertEqual(bool(silent), t.category == "chitchat", t.id)

    def test_reply_check(self):
        airtime = build_task(9, "pcm", "Credit don finish, abeg help me")
        self.assertTrue(reply_ok(airtime, airtime.gold_final))
        self.assertFalse(reply_ok(airtime, "No wahala! Which account I go send the money to, and how much?"))
        greet = build_task(13, "pcm", "How far?")
        self.assertFalse(reply_ok(greet, "Ẹ káàrọ̀ o! Kí lẹ fẹ́ ṣe?"))  # Yoruba reply to Pidgin


if __name__ == "__main__":
    unittest.main()
