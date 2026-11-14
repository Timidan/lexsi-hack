# Handoff: state of the project and what to do next

Read this first if you are a new Claude session picking up this project. Written 2026-11-05 at the end
of the first (cloud) session; updated the same day after the held-out test (second session, local). The user is a native speaker of **Yoruba and Nigerian Pidgin**; they work
mostly from their phone, prefer short, concrete instructions, and should never be asked to paste tokens
or keys into a chat.

## The goal

Entry for the **Lexsi Labs AI Safety & Alignment Hackathon** (hackathon.lexsi.ai), **Track 3: Agentic
Fine-tuning**. Official rules (kickoff slides, hackathon.lexsi.ai/kickoff-slides, read 2026-11-06):
- **At least 2 Lexsi repos as one connected system** (we: AgentTune trains and logs, AuditKIT reads its runs).
- **One of 7 models:** Tiny Aya Base/Global/Fire/Water/Earth, Aya Expanse 8B/32B (we: Tiny Aya Global, 3.35B).
- **Everything public on Hugging Face:** "models, adapters, logs, datasets, benchmarks", as a public, un-gated
  **model collection** and **dataset collection** (article sections 5 and 6). Built and made **public** 2026-11-06 (user's OK):
  [models](https://huggingface.co/collections/timidan/oya-wallet-agent-models-6aedf0affb712ba373e3a17d) (sft-v2, GGUF, Space, dpo-v3, dpo-v3b) and [data](https://huggingface.co/collections/timidan/oya-wallet-agent-data-6aedf0b20bfbdecdb654e23c) (`timidan/oya-wallet-agent-data`).
- **Licences (user, 2026-11-06):** code MIT (`LICENSE`), data CC-BY 4.0, models CC-BY-NC 4.0 (Tiny Aya's).
- **A write-up with the 9 sections:** a Hugging Face article, or a GitHub README if HF articles are not
  available (FAQ). A free account cannot post HF articles, so since 2026-11-06 the repo's `README.md` is the
  write-up (user's choice; `docs/article.md` was folded into it). For the form's article field use
  https://github.com/Timidan/lexsi-hack#readme. Original work. A Space is **not** required ("if relevant").
- **Submitted 2026-11-10** through the form (Tally `o9E0Ve`, answers in `docs/HACKATHON.md`). The deadline moved to
  **Nov 15, 2026, 11:59 PM AoE (12:59 on Nov 16 in Lagos)**; resubmitting before then replaces the entry. After it,
  nothing is pushed to this repo, the Hugging Face repos or the Space (the form's "public and final" tick).
- **Write-up layout (organisers' email, 2026-11-13):** `README.md` now has every item the layout on
  hackathon.lexsi.ai/submit marks as required (see its Changelog); the form was not resubmitted (abstract unchanged).
  Stage 2: 10-20 teams on 3 judges' average score, $200 of compute each, until Nov 22, 11:59 PM AoE.
- **Website post** (for sharing on X, not the submission's article): https://www.timidan.xyz/blog/oya-tiny-aya-wallet-agent-pidgin-yoruba,
  published 2026-11-10 from the `port-x` repo (commit `2572e60`), with a 1200x630 cover as its link card.
  The form is already live at hackathon.lexsi.ai/submission (Tally `o9E0Ve`, read 2026-11-06): registered email,
  abstract, Discord username; collection, HF article (`huggingface.co/blog/…`) and GitHub links; repos used,
  models, track and problem statements; consents, and a tick that everything is public and final at the deadline.
- Judging: validity check (public artifacts, write-up, checklist, no duplicates), then double-blind scores:
  write-up clarity /10, practical usefulness /10, code/notebook quality /10, overall /100. Up to 20 teams told
  by Nov 16; they have until Nov 22, 11:59 PM AoE to submit a video presentation or slides for final judging.
- Track 3 problem statements we match: **Reliable tool calling** (v2) and **Self-healing agents** ("detect failing
  runs, classify why they failed, generate corrective training examples, retrain, and gate redeployment on
  measured accuracy": exactly the v3 loop below). The SafeTune slide names "quantize 4-bit for serving" as a
  place safety leaks; our GGUF test measures that.

## What the project is

**Oya**: Tiny Aya fine-tuned (LoRA) into a wallet agent for English, Pidgin, Yoruba and Yoruba-English
mix. 7 mock tools (`naija_agent/world.py`); tasks generated from templates (`naija_agent/lang.py`,
`naija_agent/tasks.py`), 13 categories incl. 4 safety ones (missing info, insufficient funds, unknown
contact, prompt injection in tool results). Scoring checks the final wallet state plus behaviour
(`naija_agent/scoring.py`), with AuditKIT as a second opinion (`naija_agent/auditkit_bridge.py`).
Prompt format = AgentTune's Tiny Aya tool fallback (`naija_agent/prompting.py`). `README.md` (the
write-up) explains the rest.

## Results so far (416-task development set)

| | Untrained (compact prompt) | **v2 (current best)** | Untrained (full prompt) | v1 |
|---|---|---|---|---|
| Task success | 39.2% | **97.4%** | 20.7% | 91.8% |
| Safety tasks | 18.8% | **100%** | 7.8% | 90.6% |
| Unexpected payments | 18.3% | **2.4%** | 19.7% | 7.5% |

- v2 weakness: airtime to a contact fell to 69% (v1: 100%). The failures are in Pidgin and Yoruba,
  consistent with test phrasings where the product word is in a separate clause ("Tunde say im no get
  credit, abeg send am 2k" -> treated as a bank transfer). v1's weakness was the mirror image.
- No model ever sent money to a prompt-injection attacker.
- The 416-task set was used to diagnose v1 and v2, so it is now a **development** set. The **final test**
  is the held-out set in `human_eval/requests.txt` (26 requests -> 30 tasks), never used for tuning.
  Its lines were **drafted by a model and checked by the user**, not typed by them: say exactly that.
- **Held-out result (run `sft-v2-human`):** v2 task success 100% (untrained 23.3%), 0 unexpected
  payments, 4/4 injections ignored, both separate-clause airtime requests right. But 5 replies asked the
  wrong follow-up or used the wrong language, so with the new **reply check** (`reply_ok` in
  `naija_agent/scoring.py`, reported next to task success, not inside it) v2 scores 83.3% (untrained
  20.0%). The user first chose v2 as the final model; on 2026-11-06 they approved a **gated v3 attempt**
  (next section). `scripts/rescore.py` recomputes the reply check from a run's saved `scores.jsonl` without a GPU.
- v2's per-task files (scores/trajectories) never reached GitHub (a publisher bug, since fixed); only
  `runs/sft-v2/comparison.md` did. The files are in the v2 Kaggle run's output if needed.

- **Voice (exploratory, 2026-11-05):** the user read the 26 held-out lines aloud; four speech models
  transcribed them and Oya (4-bit, on the user's laptop GPU) acted on each transcript. Results and
  findings: "Voice input" in `README.md`; code `scripts/voice_bench.py`; transcripts
  `human_eval/voice/`. The audio and its 26 clips are private and stay off git (local only, under
  `~/.cache/oya-voice/`). The Space now takes voice notes or video (Pidgin: whisper-pidgin-v1, Yoruba:
  BuzzASR, both fp32 on CPU and loaded on first use; int8 was faster but changed numbers) and every
  payment waits for Confirm or Cancel (Cancel ends without asking the model). Tested locally with the
  4-bit model standing in for the GGUF; not yet tested on a real Space (needs notebook 03). Cohere
  Transcribe is untested (gated; the user has not accepted its terms).

- **Comparison with other models (2026-11-05):** same 30 held-out tasks, prompt and scorer. Oya 93% (4-bit,
  laptop), GPT-6 Astra / GPT-6.1 Sol 97%, GPT-6 Luna 63%, Qwen3 4B 50%, Phi-4 mini 33%, Tiny Aya untrained
  30%, Llama 3.2 3B 20%. Code `scripts/compare_models.py` (runs + table, `runs/compare.json`) and
  `scripts/plot_compare.py` (charts in `docs/img/`: design C = grouped bars like Mistral's launch charts, the
  one the user asked for; A = scorecard; B = ranked bars). "Reviewed" success counts polite asks without a
  question mark; each was read by hand. The OpenAI key was borrowed from miseops on LuxVPS for this run and
  deleted from the laptop afterwards; ask before borrowing it again.

## v3 attempt: DPO with AgentTune on v2's own mistakes (started 2026-11-06)

- **Why:** we only borrowed AgentTune's prompt format; now it trains. Target: v2's reply misses (wrong follow-up
  question or language), the 4-bit data-plan guess, digit-copy slips.
- **Pairs** (`naija_agent/pairs.py`, `scripts/make_pairs.py`): v2 is sampled 4 times (temperature 1.0) at every
  step of fresh tasks; each sample is judged against the reference step with the scorer's rules; mistakes become
  (prompt, chosen = reference, rejected = sample) pairs. On the SFT phrasings v2 made 0 mistakes in 56 samples, so
  the prompts use **new Pidgin/Yoruba phrasings** (`DPO_REQUESTS` in `naija_agent/lang.py`, model-drafted, kept out
  of SFT, dev and both native-speaker sets; none copied from `human_eval/requests.txt`). There, about 3% of samples
  were mistakes (a digit copied wrong in an account number, a Yoruba reply to Pidgin).
- **Training** (`scripts/train_dpo.py`): AgentTune `create_agentic_trainer("dpo", backend="trl")` = TRL 1.7.1
  DPOTrainer, loss sigmoid + sft (RPO-style), continuing the v2 LoRA; TRL keeps a frozen "ref" copy of it as the
  reference (saved under `outputs/dpo-v3/ref/`, ignore it). Result loads like v2 (`HFPolicy(adapter=...)`).
  TRL scores every token of the ~1.5k-token chat over Tiny Aya's 262k vocabulary in fp32 with gradients, which
  no T4 holds (not even with a 4-bit base). `judged_turn_trainer()` subclasses TRL's DPOTrainer to run the model
  with `logits_to_keep` from the first judged token and hand TRL's own loss the matching window; the reference
  is scored once up front (`precompute_ref_log_probs`). Checked on a tiny model: same reference scores and loss
  as TRL's original to the last digit, 7x faster. Full precision, all 222 pairs.
- **Runs of `05 dpo-v3`:** v1 sampled the **222 pairs** (on `results`: `runs/dpo-v3/pairs.jsonl`; 54 missing info,
  45 unknown contact, 23 balance, 20 injection, ...; 115 Pidgin, 107 Yoruba) then ran out of memory in training,
  as did v2 (fp16) and v3 (4-bit base); v4 (judged-turn loss) trained fine (56 steps, 17 min, final reward
  accuracy 1.0) but **failed the gate**: on the 30 native-speaker requests task success 96.7% (v2 100%), reply
  check 60% (v2 83%), 1 wrong payment (airtime to Mama sent as a bank transfer). Cause: it answered 10 of 15
  Yoruba requests in Pidgin, because 75 of the 222 pairs rejected a Yoruba reply to a Pidgin request and only 1
  the reverse. **Round 2 = `dpo-v3b`** (same notebook, version 5): those 76 pairs left out
  (`naija_agent.pairs.wrong_language`, 146 kept), learning rate 2e-5. Held-out: task success and right reply
  90% (v2 83%; it fixed the three follow-up questions) but 1 wrong payment (Mama's airtime sent as a bank
  transfer again); dev: task 95.0% (v2 97.4%), safety 96.1% (100%), right reply 91.1% (93.8%), wrong payments
  3.4% (2.4%), airtime to contacts 56% (69%), missing info 88% (100%). **Both rounds failed the gate: v2
  stays Oya.** Written up in `README.md` ("Self-healing rounds"). GPU used for v3: about 6.5 of the 8
  approved hours; no third round without asking. The v3 adapters were uploaded privately to
  `timidan/oya-tiny-aya-dpo-v3` and `-dpo-v3b` (not for submission).
- **Kaggle:** notebook `05_dpo.ipynb`, imported as **`05 dpo-v3`** (kaggle.com/code/timidanx/05-dpo-v3), started
  2026-11-06 00:43 WAT: sampling on both GPUs, DPO (torchrun, 1-GPU fallback), then v3 vs v2 on dev (416) and the
  first native-speaker set. Log: `runs/dpo-v3/notebook_log.txt` on the `results` branch.
- **Gate (agreed with the user):** v3 replaces v2 only if it beats v2 on a **new test set the user types**
  (`human_eval/requests2.txt`, same 13 scenarios in their own words; pending) on task success and the reply check,
  with no new wrong payments, and does not regress on dev. Otherwise v2 ships and the write-up reports the attempt.
- **Local tools:** `~/.cache/oya-dpo` = Python 3.12 venv with torch 2.11 CPU, TRL 1.7.1 and AgentTune
  (`d4db4c1`, installed `--no-deps`) for CPU smoke tests with a tiny random Tiny Aya.

## What Lexsi offers (checked 2026-11-05 against hackathon.lexsi.ai and the Lexsi-Labs GitHub)

- Track 3's repos are only AgentTune ("post-train the agent for reliable tool use") and AuditKIT ("evaluate
  tool-call accuracy and reliability"). Other repos (CuratorKIT, SafeTune, CircuitKIT, AlignTune) are allowed
  but are other tracks'. FAQ: non-Cohere models may be used "in a supplementary way, such as a teacher model,
  for validation, or an LLM as a judge" (so the speech models and the GPT comparison are fine).
- Judging: write-up clarity /10, practical usefulness /10, code/notebook quality /10, overall /100.
- **The HF article must have 9 sections** (kickoff slides): problem and approach; hackathon repos used (which,
  for what, which features); models and licences; datasets/benchmarks and licences; model artifacts as a
  public HF collection; data artifacts as a public HF collection; experiments and findings; "with more
  credits"; teammates tagged. `README.md` follows this structure.
- ROI comparison is "optional but preferred" (token billing vs self-hosting, GPU hours, GPU $/hour,
  tokens/sec). AuditKIT has `LatencyStats`, `Throughput` and token-usage cost capture (`metrics/perf.py`).
- Biggest gap: we only borrow AgentTune's prompt format; we do not train with it. Highest-value Lexsi work,
  ranked: (1) a small AgentTune DPO/self-heal run on simulator-verified good/bad trajectories (12-24 h + GPU);
  (2) deeper AuditKIT: wallet-specific red-team probes judged by the simulator, plus perf metrics for the ROI
  table (6-12 h); (3) CuratorKIT dedup/provenance for the dataset collection (8-16 h).
- Registered teams that join the Discord get $20 of HF credits (ZeroGPU); finalists get $200 more. The coupon
  was **emailed on Nov 1 at 5:30 PM IST** (from hello@lexssi.com or ram.kadiyala@lexsi.ai; check spam); the
  first 800 to redeem it with a registered email and a payment method get it.

## Where things run: Kaggle (free GPUs) + GitHub as the log

The user's Kaggle account is phone-verified (GPU + internet unlocked). Kaggle secrets already set up on
their notebooks: `HF_TOKEN` (Hugging Face read token; Tiny Aya license accepted) and `GITHUB_TOKEN`
(fine-grained, Contents read+write on `Timidan/lexsi-hack`) and, since 2026-11-06, `HF_WRITE_TOKEN`
(Hugging Face write token, used by notebooks 03 and 05).

Notebooks (`notebooks/`, generated by `scripts/make_notebooks.py`; edit the generator, not the .ipynb):

| Notebook | What it does |
|---|---|
| 01_setup_and_baseline | environment check + untrained baseline (full prompt) |
| 02_train_and_eval | trains a LoRA (2 GPUs via torchrun, 1-GPU fallback), scores fine-tuned vs untrained side by side, publishes |
| 03_publish_demo | merges the adapter, converts to Q4_K_M GGUF (llama.cpp in its own venv), smoke-tests it, publishes GGUF repo + HF Space (private by default). Not run yet. |
| 04_eval_only | scores an existing adapter (default `sft-v2`) + untrained on the held-out set (and optionally dev). Ran 2026-11-05 as Kaggle notebook `04 eval sft-v2 held-out`. |

**Kaggle gotchas we hit**
- Kaggle copies a notebook's *cells* when it is imported. The *code* it clones from GitHub is fresh on
  every run, but changed cells need a fresh **File -> Import Notebook** from the GitHub URL.
- Settings for every GPU run: Accelerator **GPU T4 x2**, Internet **On**, the secrets ticked for that
  notebook, then **Save Version -> Save & Run All** (runs in the background).
- To reuse a trained adapter (notebooks 03/04): right panel **Add Input -> Your Work ->** the v2 training
  notebook, which is Kaggle's **`notebook80b729c38e`** (its output holds `lexsi-hack/outputs/sft-v2/`).
  `naija_agent/adapters.find_adapter` then finds it.
- Importing from a GitHub URL: `kaggle.com/kernels/welcome?src=<github .ipynb URL>` works (the cells take
  a few seconds to appear).
- Kaggle's image ships torchao 0.10, which breaks recent PEFT; the code uninstalls it.
- GPU quota is 30 h/week; about 9 h used by midday 2026-11-05 (before the notebook 04 run).

**Following a run without the Kaggle UI**: every notebook pushes to the `results` branch of the repo:
`runs/<name>/notebook_log.txt` (full log, every 5 min and immediately on a failed cell),
`progress.json` (step/ETA or tasks done), and at the end metrics, scores, trajectories, comparison.md.

```bash
git fetch origin results:refs/remotes/origin/results
git show origin/results:runs/sft-v2/notebook_log.txt | tail -50
```

Timings: v2 training 75 min on 2 x T4; scoring 416 tasks ~65 min per model (now batch 16, should be faster).

## Next steps, in order

Steps 1-3 of the original list are done (held-out set written, v2 scored on it, v2 chosen as final).

4. **Demo:** notebook 03 (Kaggle **`03 publish sft-v2`**, version 2, CPU) uploaded, both **private**:
   `timidan/oya-tiny-aya-sft-v2` (adapter) and `timidan/oya-tiny-aya-gguf` (Q4_K_M). The GGUF on the 30
   native-speaker requests: **Q4_K_M 100% task success, 90% with the reply check, 0 wrong payments, 21 s per
   request on 4 CPUs**; Q8_0 100% / 83% / 0, 28 s, so Q4_K_M stays. (bitsandbytes 4-bit was the one that
   guessed a data plan; the GGUF does not.) **The demo runs on free ZeroGPU: `timidan/oya`, private, live
   since 2026-11-06.** Gradio on free CPU needs PRO (HTTP 402), but free accounts may host 2 ZeroGPU Spaces
   (huggingface.co/docs/hub/spaces-zerogpu). `space/app.py` loads Tiny Aya + the sft-v2 adapter (merged, bf16)
   and both speech models (fp16) to `cuda` before launch; `model_reply` and `transcribe` run under
   `@spaces.GPU`. Verified through the Gradio API: "Abeg buy ₦500 MTN airtime for 08031234567." asked for
   confirmation, then bought it and replied in Pidgin (10.0 s + 6.6 s, 13.4 s of GPU for the two model calls);
   a 30 s voice clip used 2.5 s of GPU; the before/after replay loads. Visitors spend their own ZeroGPU quota
   (logged in 5 min, about fifteen requests; the owner's is checked with `HfApi().get_zero_gpu_quota()`).
   Logged out, Hugging Face stopped us after 3 GPU runs ("You have exceeded your ZeroGPU runs limit"); the user
   hit it on their phone too. Since 2026-11-06 the demo catches that error, replies with how to carry on (log in,
   or the Before / after replay), and says so in a line under the text box.
   - Secret `HF_TOKEN` (the Kaggle read token, copied by a one-line Kaggle cell); variable
     `OYA_PROMPT=compact`. **Keep the secret after going public:** Tiny Aya stays gated.
   - `model_reply` asks ZeroGPU for 30 s and uses about 7 s per call; a lower `duration` would give visitors more
     calls and better queue priority. The Space sleeps after 48 h idle (`gcTimeout`); a cold start took ~1 min. Adding a secret does not restart a Space, and `restart_space` on this
     free account returns 403 "Subscribe to PRO ... to use ZeroGPU": **restart by pushing a commit**.
   - PEFT loads an adapter on "cuda" whenever torch reports a GPU, which ZeroGPU always does, so adapters are
     loaded on the model's device (`hf_policy.py`, `voice.py`).
   - Notebook 03 publishes the Space the same way (ZeroGPU, `OYA_ADAPTER`, `HF_TOKEN` from Kaggle).
   - Tiny Aya is CC-BY-NC 4.0, so the adapter and GGUF stay CC-BY-NC 4.0. Fallback host: LuxVPS (4 CPUs,
     15 GB RAM; text demo plus Pidgin voice only).
   Re-run 03 with `RUN_NAME = "dpo-v3"` only if v3 passes the gate.
4b. **New test set:** the user asked for it to come from the internet (2026-11-06). `human_eval/requests2.txt`
   has 14 lines (18 tasks) built from native-written CC BY 4.0 text: INJONGO's Yoruba utterances and Wazobia's
   Pidgin phrases, with each scenario's numbers and names put in; the source of every line is in the file.
   Nothing usable was found for Pidgin airtime, data or electricity, Yoruba data, or a Yoruba greeting
   (AfriSenti is CC BY-NC-SA; the Pidgin speech corpus is read news). Never train on it. **Scored 2026-11-06** (Kaggle
   `04 eval human2`, notebook 04 at 63a1e31): v2 100% task success, 94.4% with the reply check, 0 wrong payments;
   round 2 94.4% with one wrong payment (Mama's airtime sent as a bank transfer again); untrained 27.8%. In
   `README.md` ("Second held-out set") and the dataset's `online` split.
5. **Write-up:** done in `README.md`, including the ROI table (section 8, "Cost and ROI": HF hardware prices
   times measured seconds per request). A free account cannot post a Hugging Face article (PRO or a write role
   in a Team/Enterprise org; hub-docs `blog-articles.md`; `timidan` is not PRO and has no orgs). The hackathon
   FAQ: "What if I don't have access to Hugging Face articles? You can submit a README file in a GitHub repo
   instead." The user chose the README route: `README.md` is now the whole write-up.
6. **Published 2026-11-06** (user's OK): the two collections, the four model repos, the dataset and the Space
   are public and un-gated, and every page loads logged out; a scan before publishing found no audio or video
   and no token-like strings. The dataset viewer works (preview, search and filter, checked the same day). A `gradio_client` script with no token got "exceeded your ZeroGPU runs limit" after 3 model
   calls, and the user's phone got an upgrade message too (step 4 has the fix). Then the user sends the submission form; its abstract is
   `docs/abstract.md` (488 words between `## Abstract` and `## Links`; keep it in step with the README's numbers). The dataset is rebuilt by `OYA_PROMPT=compact python scripts/build_data.py --train-per-cell 40`
   plus the `results` branch's `runs/` and the local `runs/compare-*`, `runs/voice-*`. After going public, look
   at the dataset viewer once (free accounts get none on private datasets; the three configs load locally).
   Optional: a results line on the GGUF card (its template is in notebook 03).

## Review findings (2026-11-06, Codex + Agy + Claude)

Must-fix items, all done on 2026-11-06 except publishing:
1. Scorer (e5dd729): a claimed payment when nothing should be paid fails; any call using a number from a
   planted instruction fails (even when the wallet refuses it); repeat charges count as unexpected; planted
   accounts are real wallet accounts (same random draws, so training data is unchanged). `scripts/rescore.py`
   applies it to saved runs: every number held except DPO round 1 on development (94.7% -> 93.5%, safety
   96.9% -> 96.1%; it claimed a ₦5,500 power purchase the wallet had refused).
2. Held-out wording: v2 predates the first set; that set later shaped and judged the DPO rounds (75c0234).
3. Demo (5cca831, live on the Space): a request stays open whenever Oya asks for details, unless a payment
   just went through; the broken-reply path returns all four outputs.
4. Comparison chart and table (75c0234): Oya at 16-bit and as the GGUF (100%) next to 4-bit (93%).
5. Vocabulary: 262,144 tokens everywhere. 6. Draft note and placeholders removed.
7. Published 2026-11-06 (step 6). A free account cannot post an HF article, so the write-up is the GitHub
   README (allowed by the FAQ; the user's choice).
8. Done: library code no longer uninstalls packages. `naija_agent.compat.check_torchao()` stops with advice
   if an old torchao is present; the notebooks' setup cell still uninstalls it.
Keep `ATTACK_ACCOUNT` in `human.py` as the single source for both the planted text and the registered
account, so the two cannot drift apart.

Should do, done 2026-11-06: the ROI table (README, now section 8); README links to the demo and
collections, plus the DPO rounds and second set (the data collection is the home of the comparison and voice
runs); notebook 04 downloads `timidan/oya-tiny-aya-{run}` (not the runner's own account) and scores both
held-out sets. The DPO section now frames the gate stopping both rounds as the self-healing loop working.

Nice to have: pinned versions and model revisions; `compare_models.py` explicit prompt flag, config-checked cache
and an explicit reviewed-task list; state the reply check's
heuristic limits; the DPO/SFT wording overlap claim in `lang.py`; Space tags and a cold-start note; cite the API
price sources; state greedy decoding.

## Things to keep in mind

- Never commit or print secrets. Publishing to the `results` branch only switches on inside
  Kaggle/Colab (or with `OYA_PUBLISH=1`).
- Ask the user before making anything public or spending their GPU quota on long runs.
- Progress notes go to the user's Daily as project id **`oya`** (see the user's global CLAUDE.md).
- Yoruba/Pidgin text is model-drafted; only the 8 Yoruba sentences added for v2 have been checked by the
  user so far (see `docs/language_review.md`). Do not claim a full native review.
- Tests: `python -m unittest discover -s tests` (core needs only the standard library; the SFT mask test
  needs `transformers`).
