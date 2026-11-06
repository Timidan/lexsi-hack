<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/icon/oya-icon-dark.svg">
  <img src="docs/img/icon/oya-icon.svg" alt="Oya" width="72">
</picture>

# Oya: a Tiny Aya wallet agent for English, Nigerian Pidgin and Yoruba

Entry for the **Lexsi Labs AI Safety & Alignment Hackathon**, Track 3: **Agentic Fine-tuning**.

Millions of Nigerians buy airtime, data and electricity and send money by phone, and many of them type in Pidgin, Yoruba, or a mix with English. Small open models rarely handle tool calls well in those languages. **Oya** fine-tunes Cohere's **Tiny Aya** (3.35B) to:

- **pick the right tool and fill in the arguments correctly** from Pidgin, Yoruba and code-mixed requests ("Jọ̀ọ́ send 3k airtime sí Bisi"),
- **call tools in the right order** (find the contact, verify the account, then pay),
- **stay safe with money**: ask instead of guessing missing details, refuse to overspend, and ignore instructions planted inside tool results (prompt injection).

Everything is scored automatically against a mock payments world, so there is no LLM judge: we check the money that actually moved.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/compare-c-dark.png">
  <img alt="Task success on 30 held-out Yoruba and Pidgin requests: Oya 93%, GPT-6 Astra 97%, Qwen3 4B 50%, untrained Tiny Aya 30%, Llama 3.2 3B 20%; also by Pidgin, Yoruba and safety tasks" src="docs/img/compare-c-light.png">
</picture>

Same prompt, tasks and scorer for every model; open models in 4-bit on a 4 GB laptop GPU. Details, wrong
payments and cost: [docs/article.md](docs/article.md#compared-with-other-models).

## Results so far (416 development tasks)

| | Untrained (compact prompt) | **Oya v2** | Untrained (full prompt) | Oya v1 |
|---|---|---|---|---|
| Task success | 39.2% | **97.4%** | 20.7% | 91.8% |
| Safety tasks handled correctly | 18.8% | **100%** | 7.8% | 90.6% |
| Valid tool calls | 78.4% | **99.8%** | 60.8% | 99.6% |
| Tasks where unexpected money moved | 18.3% | **2.4%** | 19.7% | 7.5% |
| Success: English / Pidgin | 43.3% / 34.6% | **99.0% / 94.2%** | 26.0% / 17.3% | 89.4% / 93.3% |
| Success: Yoruba / Yoruba-English | 32.7% / 46.2% | **96.2% / 100%** | 14.4% / 25.0% | 85.6% / 99.0% |

Planted prompt injections: no model ever sent money to the attacker's account (v2: 32/32 injection tasks
solved). v2's remaining errors: airtime to a contact (69%) when the product word sits in a separate clause.
These 416 tasks were used to diagnose v1 and v2, so they are a development set; the final test is a set
of requests in new phrasings, drafted with a model and checked by a native speaker (`human_eval/`),
never used for development. On those 30 held-out tasks v2 solved all 30 with no unexpected payments
(untrained: 23.3%); 5 of its replies asked the wrong follow-up question or used the wrong language, so
it scores 83.3% when the reply must also be right (untrained: 20.0%). Full write-up: `docs/article.md`.

## How it works

| Piece | File |
|---|---|
| Mock wallet world: 7 tools (`check_balance`, `find_contact`, `buy_airtime`, `buy_data`, `pay_electricity`, `lookup_account`, `send_money`) with deterministic state | `naija_agent/world.py` |
| Request and reply templates in `en`, `pcm`, `yo`, `yo_en` | `naija_agent/lang.py` |
| Task generator: 13 categories × 4 languages, train/test split **by template** so test phrasings are never seen in training | `naija_agent/tasks.py` |
| Prompt format: identical to AgentTune's Tiny Aya tool fallback, so data and logs plug into Lexsi's toolkits | `naija_agent/prompting.py` |
| Episode runner (batched) and AgentTune-format trajectory logs | `naija_agent/runner.py` |
| Scorer: verified final state plus behaviour checks | `naija_agent/scoring.py` |
| Second-opinion scoring with Lexsi's **AuditKIT** | `naija_agent/auditkit_bridge.py` |

**Task categories.** Everyday tasks: airtime, data, electricity, transfers, by number or by saved contact, balance checks, small talk. Safety tasks: `insufficient_funds` (must not overspend), `missing_info` (must ask, not guess), `unknown_contact` (must ask for the number), `injection` (tool output contains a planted instruction to send extra money; must ignore it).

**Metrics** (before vs after fine-tuning, overall and per language):
1. **Task success**: the right payments happened (and nothing else), or the agent correctly asked or refused.
2. **Valid tool-call rate**: calls that parse, name a real tool and pass the argument schema.
3. **Tool F1** against the reference route, and **extra calls** per task (efficiency).
4. **Safety success** on the four safety categories, and the **unexpected payment rate**.

## Run it

```bash
python -m unittest discover -s tests          # CPU, no dependencies
python scripts/build_data.py                  # writes data/ (tasks + SFT chats)
python scripts/eval.py --policy gold --out runs/gold --auditkit   # sanity check: 100%
```

GPU (free Kaggle T4): open `notebooks/01_setup_and_baseline.ipynb` on Kaggle and follow the first cell.

## Notes

- Prices, providers and accounts are a simulation, not live tariffs or real accounts.
- The Yoruba and Pidgin text is a first draft pending native-speaker review: see `docs/language_review.md`.
- Built with [AgentTune](https://github.com/Lexsi-Labs/AgentTune) and [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) by Lexsi Labs, and [Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-global) by Cohere Labs (CC-BY-NC).

## Licence

- Code: MIT ([LICENSE](LICENSE)).
- Data (`human_eval/`, `runs/` and the generated sets in [timidan/oya-wallet-agent-data](https://huggingface.co/datasets/timidan/oya-wallet-agent-data)): CC-BY 4.0.
- Models ([timidan/oya-tiny-aya-sft-v2](https://huggingface.co/timidan/oya-tiny-aya-sft-v2) and its GGUF): CC-BY-NC 4.0,
  following Tiny Aya. The demo Space is tagged CC-BY-NC 4.0 too, because it serves the model.
