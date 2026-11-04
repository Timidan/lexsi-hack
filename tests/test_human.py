"""Human-written test set: parsing, task building and scoring (gold must pass, a silent agent must not)."""

import unittest

from naija_agent.human import SCENARIOS, build_task, parse_requests
from naija_agent.runner import gold_policy, run_episode
from naija_agent.scoring import score_episode

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


if __name__ == "__main__":
    unittest.main()
