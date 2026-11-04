"""Loss-mask test for SFT data (needs `transformers` and `tokenizers`, no torch).

Uses a small Cohere-style chat template written for this test, so it checks
our masking logic without downloading the real Tiny Aya tokenizer.
"""

import unittest

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
except ImportError:  # pragma: no cover
    PreTrainedTokenizerFast = None

from naija_agent import generate, gold_conversation
from naija_agent.sft import build_example

TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'system' %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ m['content'] }}<|END_OF_TURN_TOKEN|>"
    "{% elif m['role'] == 'user' %}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ m['content'] }}<|END_OF_TURN_TOKEN|>"
    "{% else %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{{ m['content'] }}"
    "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{% endif %}"
)
SPECIALS = ["<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>",
            "<|SYSTEM_TOKEN|>", "<|START_RESPONSE|>", "<|END_RESPONSE|>"]


def _tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<BOS_TOKEN>", eos_token="<|END_OF_TURN_TOKEN|>",
                                   pad_token="<PAD>", additional_special_tokens=SPECIALS, chat_template=TEMPLATE)


@unittest.skipIf(PreTrainedTokenizerFast is None, "transformers not installed")
class LossMask(unittest.TestCase):
    def test_only_assistant_turns_are_trained(self):
        tok = _tokenizer()
        task = next(t for t in generate("train", 2, seed=3) if t.category == "transfer_contact")
        msgs = gold_conversation(task)
        ex = build_example(tok, msgs)
        trained = tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
        expected = "".join(m["content"] + "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
                           for m in msgs if m["role"] == "assistant")
        self.assertEqual(trained, expected)
        self.assertNotIn("Tool result", trained)
        self.assertNotIn("You are Oya", trained)

    def test_max_len_drops_long_examples(self):
        tok = _tokenizer()
        msgs = gold_conversation(generate("train", 1, seed=4)[0])
        self.assertIsNone(build_example(tok, msgs, max_len=10))


if __name__ == "__main__":
    unittest.main()
