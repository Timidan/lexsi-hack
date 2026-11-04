"""Generate the Kaggle/Colab notebooks in notebooks/ (they share setup cells).

    python scripts/make_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells):
    out = []
    for kind, src in cells:
        lines = src.strip("\n").split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}}


SETUP_HELP = """
**One-time Kaggle setup**
1. Settings → **Accelerator: GPU T4 x2** (or P100) and **Internet: on** (needs a verified phone number).
2. Add-ons → **Secrets**: `HF_TOKEN` (Hugging Face *read* token; accept the Tiny Aya license first) and, while the repo is private, `GITHUB_TOKEN` (fine-grained, read-only on `Timidan/lexsi-hack`). Tick both for this notebook.
3. **Save Version → Save & Run All** runs it in the background; results appear in the **Output** tab.

On Colab: Runtime → T4 GPU, add the same names under 🔑 Secrets, then Run all (keep the tab open).
"""

SETUP_CELLS = [
    ("code", """
import subprocess, sys
print(sys.version)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
"""),
    ("code", """
# Secrets come from Kaggle (Add-ons -> Secrets), Colab (key icon) or environment variables.
def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            return os.environ.get(name)

HF_TOKEN = secret("HF_TOKEN")
GITHUB_TOKEN = secret("GITHUB_TOKEN")
assert HF_TOKEN, "Add an HF_TOKEN secret (and accept the Tiny Aya license on Hugging Face)"

url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{REPO}.git" if GITHUB_TOKEN else f"https://github.com/{REPO}.git"
subprocess.run(["rm", "-rf", "lexsi-hack"])
r = subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, url, "lexsi-hack"], capture_output=True, text=True)
print("clone ok" if r.returncode == 0 else "clone FAILED: " + r.stderr.replace(GITHUB_TOKEN or "<none>", "***"))
os.chdir("lexsi-hack")
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
"""),
    ("code", """
# Tiny Aya (cohere2 architecture) is verified with transformers 5.x (same floor AgentTune uses).
!pip install -q -U "transformers>=5.17,<6" accelerate peft bitsandbytes auditkit
"""),
    ("code", """
from huggingface_hub import login
login(token=HF_TOKEN)

# CPU tests + build the task sets (takes seconds)
!python -m unittest discover -s tests
!python scripts/build_data.py --train-per-cell {TRAIN_PER_CELL}
"""),
]

SAMPLES_CELL = ("code", """
samples = [
    "Abeg buy ₦500 MTN airtime for 08031234567.",          # Pidgin, direct
    "Fi 5000 ránṣẹ́ sí Tunde.",                              # Yoruba, needs find_contact first
    "Bá mi ra káàdì.",                                       # Yoruba, missing details: should ask
    "What's my balance?",                                    # English
    "Jọ̀ọ́ pay 3k for light, meter mi ni 45123456789, Ikeja Electric.",  # Yoruba-English mix
]
for s, out in zip(samples, policy([initial_messages(s) for s in samples])):
    print("USER:", s, "\\nMODEL:", out[:400], "\\n" + "-" * 60)
""")

FAILURES_CELL = ("code", """
# Failures to read
import json
rows = [json.loads(l) for l in open(f"{RUN_DIR}/scores.jsonl", encoding="utf-8")]
bad = [r for r in rows if not r["success"]]
print(f"{len(bad)} failures out of {len(rows)}")
for r in bad[:15]:
    failed = [k for k, v in r["checks"].items() if not v]
    print(f"[{r['category']} / {r['lang']}] {r['prompt']}\\n   failed: {failed}\\n   final: {str(r['final'])[:200]}\\n")
""")

AUDITKIT_CELL = ("code", """
# Independent second opinion with Lexsi Labs' AuditKIT
from naija_agent.auditkit_bridge import auditkit_eval
print(auditkit_eval(tasks, RUN_DIR).summary())
""")


baseline = nb([
    ("markdown", """
# 01 · Setup check + baseline (untrained Tiny Aya)

Loads Tiny Aya on a free T4, runs a few sample requests, then scores the **untrained** model on all 416 held-out test tasks. This is the "before" number.
""" + SETUP_HELP),
    ("code", """
# ---- settings ----
LIMIT = None          # e.g. 40 for a quick check; None = all 416 test tasks
BATCH_SIZE = 8        # lower to 4 if you hit CUDA out-of-memory
MODEL_ID = "CohereLabs/tiny-aya-global"
REPO = "Timidan/lexsi-hack"
BRANCH = "main"
TRAIN_PER_CELL = 40
RUN_DIR = "runs/baseline"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one GPU is enough for a 3.35B model
"""),
    *SETUP_CELLS,
    ("markdown", "## Load Tiny Aya and look at the prompt"),
    ("code", """
import sys, torch, transformers
sys.path.insert(0, ".")
print("torch", torch.__version__, "| transformers", transformers.__version__)

from naija_agent.hf_policy import HFPolicy
from naija_agent.runner import initial_messages

policy = HFPolicy(MODEL_ID, max_new_tokens=256)
print("dtype:", policy.dtype, "| GPU memory used: %.1f GB" % (torch.cuda.memory_allocated() / 1e9))

text = policy.render(initial_messages("Abeg buy ₦500 MTN airtime for 08031234567."))
print(len(policy.tok(text, add_special_tokens=False).input_ids), "prompt tokens")
print(text[:300], "\\n...\\n", text[-400:])
"""),
    SAMPLES_CELL,
    ("markdown", "## Baseline: untrained Tiny Aya on the held-out test set"),
    ("code", """
from naija_agent.evaluate import evaluate, load_tasks

tasks = load_tasks("data/tasks_test.jsonl", LIMIT)
metrics = evaluate(tasks, policy, RUN_DIR, batch_size=BATCH_SIZE,
                   config={"model": MODEL_ID, "adapter": None, "dtype": str(policy.dtype)})
"""),
    AUDITKIT_CELL,
    FAILURES_CELL,
])

train = nb([
    ("markdown", """
# 02 · Fine-tune Tiny Aya (LoRA) and re-evaluate

Trains a LoRA adapter on the reference conversations (assistant turns only), then scores the tuned model on the **same 416 held-out test tasks** as notebook 01, so the numbers compare directly.

Roughly 1.5 to 3 hours on one T4 for one epoch; the log prints an ETA. The adapter is saved every 40 steps, so a timeout still leaves a usable checkpoint.
""" + SETUP_HELP + """
Optional: a `HF_WRITE_TOKEN` secret (Hugging Face *write* token) uploads the adapter and logs to a private Hugging Face repo at the end.
"""),
    ("code", """
# ---- settings ----
RUN_NAME = "sft-v1"
TRAIN_PER_CELL = 40      # training chats per (category, language): 40 -> 2,080 chats
EPOCHS = 1.0
LR = 2e-4
GRAD_ACCUM = 16          # effective batch size
LORA_R, LORA_ALPHA = 16, 32
MAX_LEN = 4096
TIME_BUDGET_H = 8.5      # stop training early (and still evaluate) before Kaggle's 12 h limit
EVAL_LIMIT = None        # None = all 416 test tasks
BATCH_SIZE = 8
HUB_REPO = None          # e.g. "your-hf-username/oya-tiny-aya-lora" (needs HF_WRITE_TOKEN)

MODEL_ID = "CohereLabs/tiny-aya-global"
REPO = "Timidan/lexsi-hack"
BRANCH = "main"
ADAPTER_DIR = f"outputs/{RUN_NAME}"
RUN_DIR = f"runs/{RUN_NAME}"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""),
    *SETUP_CELLS,
    ("markdown", "## Train"),
    ("code", """
import sys, torch, transformers
sys.path.insert(0, ".")
print("torch", torch.__version__, "| transformers", transformers.__version__)
from naija_agent.sft import train_lora

log = train_lora(MODEL_ID, "data/sft_train.jsonl", ADAPTER_DIR, val_path="data/sft_val.jsonl",
                 max_len=MAX_LEN, epochs=EPOCHS, lr=LR, grad_accum=GRAD_ACCUM,
                 lora_r=LORA_R, lora_alpha=LORA_ALPHA, time_budget_s=TIME_BUDGET_H * 3600)
print(f"trained in {log['seconds'] / 60:.0f} min | val loss: {[v['loss'] for v in log['val']]}")
"""),
    ("code", """
import matplotlib.pyplot as plt
steps = [s["step"] for s in log["steps"]]
plt.figure(figsize=(7, 3.5))
plt.plot(steps, [s["loss"] for s in log["steps"]], label="train")
vals = [v for v in log["val"] if v["loss"] is not None]
plt.plot([v["step"] for v in vals], [v["loss"] for v in vals], "o-", label="val")
plt.xlabel("step"); plt.ylabel("loss (assistant tokens)"); plt.legend(); plt.title(RUN_NAME)
plt.tight_layout(); plt.savefig(f"{ADAPTER_DIR}/loss.png", dpi=150); plt.show()
"""),
    ("markdown", "## Evaluate the tuned model on the held-out test set"),
    ("code", """
import gc
gc.collect(); torch.cuda.empty_cache()

from naija_agent.hf_policy import HFPolicy
from naija_agent.runner import initial_messages
from naija_agent.evaluate import evaluate, load_tasks

policy = HFPolicy(MODEL_ID, adapter=ADAPTER_DIR, max_new_tokens=256)
tasks = load_tasks("data/tasks_test.jsonl", EVAL_LIMIT)
metrics = evaluate(tasks, policy, RUN_DIR, batch_size=BATCH_SIZE,
                   config={"model": MODEL_ID, "adapter": ADAPTER_DIR, "train": log["config"]})
"""),
    AUDITKIT_CELL,
    SAMPLES_CELL,
    FAILURES_CELL,
    ("code", """
# Optional: upload adapter + logs to a private Hugging Face repo
HF_WRITE_TOKEN = secret("HF_WRITE_TOKEN")
if HUB_REPO and HF_WRITE_TOKEN:
    import shutil
    from huggingface_hub import HfApi
    shutil.copytree(RUN_DIR, f"{ADAPTER_DIR}/eval", dirs_exist_ok=True)
    api = HfApi(token=HF_WRITE_TOKEN)
    api.create_repo(HUB_REPO, private=True, exist_ok=True)
    api.upload_folder(folder_path=ADAPTER_DIR, repo_id=HUB_REPO, commit_message=f"{RUN_NAME}")
    print("uploaded to https://huggingface.co/" + HUB_REPO)
else:
    print("Skipping upload. The adapter is in the Output tab under lexsi-hack/" + ADAPTER_DIR)
"""),
])

for name, book in (("01_setup_and_baseline.ipynb", baseline), ("02_train_and_eval.ipynb", train)):
    path = ROOT / "notebooks" / name
    path.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))
