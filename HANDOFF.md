# Handoff: state of the project and what to do next

Read this first if you are a new Claude session picking up this project. Written 2026-11-05 at the end
of the first (cloud) session; updated the same day after the held-out test (second session, local). The user is a native speaker of **Yoruba and Nigerian Pidgin**; they work
mostly from their phone, prefer short, concrete instructions, and should never be asked to paste tokens
or keys into a chat.

## The goal

Entry for the **Lexsi Labs AI Safety & Alignment Hackathon** (hackathon.lexsi.ai), **Track 3: Agentic
Fine-tuning**. Rules as we understood them (confirm on the hackathon Discord): Cohere Labs models only
(we use `CohereLabs/tiny-aya-global`, 3.35B), Lexsi toolkits encouraged (AgentTune, AuditKIT), teams of 1-3.

- **Submission form (link is in the Discord): due end of day Nov 14, 2026, AoE (12:59 on Nov 15 in Lagos).** Aim for Nov 13.
- Up to 20 finalists announced around Nov 16; they may keep improving until Nov 21; live demo Nov 22.
- Deliverables (from another entrant's public checklist; verify in Discord): public GitHub repo; public
  Hugging Face Space demo that shows the tool trace; Hugging Face artifacts (LoRA adapter,
  `trajectories.jsonl` logs, dataset card); a Hugging Face article (metrics and why, dataset/task/model
  rationale, before/after table); at least two metrics before and after; optional ROI table
  (self-hosted Tiny Aya vs a closed API).

## What the project is

**Oya**: Tiny Aya fine-tuned (LoRA) into a wallet agent for English, Pidgin, Yoruba and Yoruba-English
mix. 7 mock tools (`naija_agent/world.py`); tasks generated from templates (`naija_agent/lang.py`,
`naija_agent/tasks.py`), 13 categories incl. 4 safety ones (missing info, insufficient funds, unknown
contact, prompt injection in tool results). Scoring checks the final wallet state plus behaviour
(`naija_agent/scoring.py`), with AuditKIT as a second opinion (`naija_agent/auditkit_bridge.py`).
Prompt format = AgentTune's Tiny Aya tool fallback (`naija_agent/prompting.py`). README.md and
`docs/article.md` (the write-up draft) explain the rest.

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
  findings: "Voice input" in `docs/article.md`; code `scripts/voice_bench.py`; transcripts
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
  credits"; teammates tagged. `docs/article.md` follows this structure (since 2026-11-05); still open in it:
  the two collection links, the dataset licence, the teammates' HF usernames, and the `[review]` marker.
- ROI comparison is "optional but preferred" (token billing vs self-hosting, GPU hours, GPU $/hour,
  tokens/sec). AuditKIT has `LatencyStats`, `Throughput` and token-usage cost capture (`metrics/perf.py`).
- Biggest gap: we only borrow AgentTune's prompt format; we do not train with it. Highest-value Lexsi work,
  ranked: (1) a small AgentTune DPO/self-heal run on simulator-verified good/bad trajectories (12-24 h + GPU);
  (2) deeper AuditKIT: wallet-specific red-team probes judged by the simulator, plus perf metrics for the ROI
  table (6-12 h); (3) CuratorKIT dedup/provenance for the dataset collection (8-16 h).
- Registered teams that join the Discord get $20 of HF credits (ZeroGPU); finalists get $200 more.

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
   guessed a data plan; the GGUF does not.) **Creating the Space failed: HTTP 402, Gradio Spaces now need HF
   PRO, even on free CPU; static Spaces stay free.** The Submit page does not require a Space (public model and
   dataset collections, write-up, GitHub repo, abstract), so this is a choice, not a blocker: HF PRO ($9/month,
   the user pays), the Discord's $20 HF credits if they cover it, or LuxVPS (4 CPUs, 15 GB RAM, ~11 GB free, no
   swap, ~20 live entries; fits the text demo plus Pidgin voice, not the 6 GB Yoruba speech model as well).
   The Space code in the repo now warms the model at startup (`warm_up` in `space/app.py`). Re-run 03 with
   `RUN_NAME = "dpo-v3"` only if v3 passes the gate.
4b. **New test set:** the user is writing the 13 scenarios again in Yoruba and Pidgin, in their own words
   (asked 2026-11-06). Put them in `human_eval/requests2.txt` (same `1Y:`/`1P:` format), never train on them,
   and score untrained, v2 and v3 on them (`naija_agent.human.load_human_tasks`).
5. **Write-up:** held-out results are in `docs/article.md`; still open: the `[review]` marker, a dataset card,
   optionally the ROI table. Publish the article on Hugging Face.
6. **Before the deadline:** make the adapter, GGUF repo and Space public (ask the user first), check the
   GitHub repo is public, and the user submits the form from the Discord.

## Things to keep in mind

- Never commit or print secrets. Publishing to the `results` branch only switches on inside
  Kaggle/Colab (or with `OYA_PUBLISH=1`).
- Ask the user before making anything public or spending their GPU quota on long runs.
- Progress notes go to the user's Daily as project id **`oya`** (see the user's global CLAUDE.md).
- Yoruba/Pidgin text is model-drafted; only the 8 Yoruba sentences added for v2 have been checked by the
  user so far (see `docs/language_review.md`). Do not claim a full native review.
- Tests: `python -m unittest discover -s tests` (core needs only the standard library; the SFT mask test
  needs `transformers`).
