"""LoRA key pairing used by scripts/merge_lora.py (no torch needed)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_lora import lora_pairs  # noqa: E402


class MergeKeys(unittest.TestCase):
    def test_pairs_map_to_base_weights(self):
        t = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": "A0",
             "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": "B0",
             "base_model.model.model.layers.3.mlp.down_proj.lora_A.weight": "A1",
             "base_model.model.model.layers.3.mlp.down_proj.lora_B.weight": "B1"}
        self.assertEqual(lora_pairs(t), {"model.layers.0.self_attn.q_proj.weight": ("A0", "B0"),
                                         "model.layers.3.mlp.down_proj.weight": ("A1", "B1")})

    def test_incomplete_pair_rejected(self):
        with self.assertRaises(ValueError):
            lora_pairs({"base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": "A"})


if __name__ == "__main__":
    unittest.main()
