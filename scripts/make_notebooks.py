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
2. Add-ons → **Secrets**: `HF_TOKEN` (Hugging Face *read* token; accept the Tiny Aya license first) Optional: `GITHUB_TOKEN` (fine-grained, **Contents: Read and write** on `Timidan/lexsi-hack`) so the notebook pushes its results to the `results` branch. Tick them for this notebook.
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
import os, subprocess, sys

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

# mirror this notebook's output (and any cell error) to the repo's `results` branch
sys.path.insert(0, ".")
from naija_agent.notebook_log import start_live_log
start_live_log(LOG_NAME)
"""),
    ("code", """
# Tiny Aya (cohere2 architecture) is verified with transformers 5.x (same floor AgentTune uses).
!pip install -q -U "transformers>=5.17,<6" accelerate peft bitsandbytes auditkit
# Kaggle ships torchao 0.10, which recent PEFT refuses to load next to; we don't use it
!pip uninstall -y -q torchao
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
LOG_NAME = "baseline-full"

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
    ("code", """
# push the end of this notebook's log to GitHub
from naija_agent.notebook_log import flush_live_log
flush_live_log()
"""),
])

train = nb([
    ("markdown", """
# 02 · Fine-tune Tiny Aya (LoRA) and compare before vs after

1. Builds the training chats and trains a LoRA adapter on the assistant turns only, using **both GPUs** when the notebook has two (falls back to one GPU automatically if that fails).
2. Scores the **fine-tuned** and the **untrained** model on the same 416 held-out test tasks, at the same time, one model per GPU, and prints a before/after table.
3. Publishes live progress and all results to the repo's `results` branch (needs a `GITHUB_TOKEN` secret with write access).

About 1 to 1.5 hours on Kaggle's T4 x2 with the compact prompt.
""" + SETUP_HELP + """
Optional: a `HF_WRITE_TOKEN` secret (Hugging Face *write* token) uploads the adapter to a private Hugging Face repo at the end.
"""),
    ("code", """
# ---- settings ----
RUN_NAME = "sft-v2"
PROMPT_STYLE = "compact"   # "compact" = 40% shorter prompt (faster); "full" = the prompt sft-v1 used
TRAIN_PER_CELL = 40        # training chats per (category, language): 40 -> 2,080 chats
EPOCHS = 1.0
LR = 2e-4
GRAD_ACCUM = 16            # effective batch size (split across GPUs)
LORA_R, LORA_ALPHA = 16, 32
MAX_LEN = 4096
TIME_BUDGET_H = 8.0        # stop training early (and still evaluate) well before Kaggle's 12 h limit
EVAL_LIMIT = None          # None = all 416 test tasks
EVAL_BASELINE = True       # also score the untrained model with the same prompt (before/after table)
BATCH_SIZE = 16            # conversations generated together; halves automatically if GPU memory runs out
HUB_REPO = "auto"          # upload adapter (private) to {hf user}/oya-tiny-aya-{RUN_NAME}; needs HF_WRITE_TOKEN

MODEL_ID = "CohereLabs/tiny-aya-global"
REPO = "Timidan/lexsi-hack"
BRANCH = "main"
ADAPTER_DIR = f"outputs/{RUN_NAME}"
RUN_DIR = f"runs/{RUN_NAME}"
BASE_RUN_DIR = f"runs/baseline-{PROMPT_STYLE}"
LOG_NAME = RUN_NAME

import os
os.environ["OYA_PROMPT"] = PROMPT_STYLE            # used by data building, training and evaluation
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""),
    *SETUP_CELLS,
    ("markdown", "## Train (both GPUs if available)"),
    ("code", """
import json, sys, torch, transformers
sys.path.insert(0, ".")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| GPUs:", torch.cuda.device_count())

from huggingface_hub import snapshot_download
snapshot_download(MODEL_ID)                     # download once, before the GPU workers start
if GITHUB_TOKEN:
    os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN   # lets the training workers publish live progress

def run(cmd):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()

NGPU = torch.cuda.device_count()
args = (f"--model {MODEL_ID} --out {ADAPTER_DIR} --epochs {EPOCHS} --lr {LR} --grad-accum {GRAD_ACCUM} "
        f"--lora-r {LORA_R} --lora-alpha {LORA_ALPHA} --max-len {MAX_LEN} --time-budget-h {TIME_BUDGET_H}")
rc = 1
if NGPU >= 2:
    rc = run(f"{sys.executable} -m torch.distributed.run --standalone --nproc_per_node={NGPU} scripts/train.py {args}")
    if rc != 0:
        print(f"\\n*** {NGPU}-GPU training failed (exit code {rc}); retrying on one GPU ***\\n", flush=True)
if rc != 0:
    rc = run(f"CUDA_VISIBLE_DEVICES=0 {sys.executable} scripts/train.py {args}")
assert rc == 0, "training failed: see the log above"

log = json.load(open(f"{ADAPTER_DIR}/train_log.json"))
print(f"trained in {log['seconds'] / 60:.0f} min on {log['config']['gpus']} GPU(s) | val loss:",
      [round(v["loss"], 4) for v in log["val"] if v["loss"] is not None])
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
    ("markdown", "## Score fine-tuned vs untrained on the same held-out test set"),
    ("code", """
import gc
gc.collect(); torch.cuda.empty_cache()
from naija_agent.evaluate import evaluate_parallel, load_tasks
from naija_agent.hf_policy import HFPolicy
from naija_agent.scoring import compare

tasks = load_tasks("data/tasks_test.jsonl", EVAL_LIMIT)
jobs = [{"name": "fine-tuned", "out_dir": RUN_DIR,
         "make_policy": lambda d: HFPolicy(MODEL_ID, adapter=ADAPTER_DIR, device=d),
         "config": {"model": MODEL_ID, "adapter": ADAPTER_DIR, "train": log["config"]}}]
if EVAL_BASELINE:
    jobs.append({"name": "untrained", "out_dir": BASE_RUN_DIR,
                 "make_policy": lambda d: HFPolicy(MODEL_ID, device=d),
                 "config": {"model": MODEL_ID, "adapter": None}})
metrics = evaluate_parallel(tasks, jobs, batch_size=BATCH_SIZE)

if EVAL_BASELINE:
    table = compare({"untrained": metrics["untrained"], "fine-tuned": metrics["fine-tuned"]},
                    title=f"{RUN_NAME}: before vs after ({len(tasks)} held-out test tasks, {PROMPT_STYLE} prompt)")
    open(f"{RUN_DIR}/comparison.md", "w").write(table)
    print(table)
"""),
    ("code", """
# Requests written by a native speaker (human_eval/requests.txt), if any: never used for training
if os.path.exists("data/tasks_human.jsonl"):
    human = load_tasks("data/tasks_human.jsonl")
    hjobs = [dict(j, out_dir=j["out_dir"] + "-human") for j in jobs]
    hmetrics = evaluate_parallel(human, hjobs, batch_size=BATCH_SIZE)
    if EVAL_BASELINE:
        htable = compare({"untrained": hmetrics["untrained"], "fine-tuned": hmetrics["fine-tuned"]},
                         title=f"{RUN_NAME}: {len(human)} held-out requests checked by a native speaker")
        open(f"{RUN_DIR}/comparison_human.md", "w").write(htable)
        print(htable)
else:
    print("no human-written requests yet (human_eval/requests.txt is empty)")
"""),
    ("code", """
# Independent second opinion with Lexsi Labs' AuditKIT
from naija_agent.auditkit_bridge import auditkit_eval
for d in [RUN_DIR] + ([BASE_RUN_DIR] if EVAL_BASELINE else []):
    print(f"== AuditKIT: {d} ==")
    print(auditkit_eval(tasks, d).summary(), "\\n")
"""),
    ("code", """
# The five sample requests, answered by the fine-tuned model
from naija_agent.runner import initial_messages
policy = HFPolicy(MODEL_ID, adapter=ADAPTER_DIR)
""" + SAMPLES_CELL[1]),
    FAILURES_CELL,
    ("code", """
# Upload the adapter (+ eval results) to a PRIVATE Hugging Face repo (made public at submission time).
# Notebook 03 takes it from there to build the demo.
HF_WRITE_TOKEN = secret("HF_WRITE_TOKEN")
if HUB_REPO and HF_WRITE_TOKEN:
    import shutil
    from huggingface_hub import HfApi
    api = HfApi(token=HF_WRITE_TOKEN)
    hub_repo = f"{api.whoami()['name']}/oya-tiny-aya-{RUN_NAME}" if HUB_REPO == "auto" else HUB_REPO
    shutil.copytree(RUN_DIR, f"{ADAPTER_DIR}/eval", dirs_exist_ok=True)
    card = "\\n".join([
        "---", f"base_model: {MODEL_ID}", "library_name: peft", "license: cc-by-nc-4.0",
        "tags: [lora, tool-calling, agents, yoruba, nigerian-pidgin, aya, lexsi-hackathon]", "---",
        f"# Oya LoRA adapter ({RUN_NAME})", "",
        f"LoRA adapter for {MODEL_ID}: a tool-calling wallet agent for English, Nigerian Pidgin and Yoruba",
        "(airtime, data, electricity, transfers), trained to ask when details are missing, refuse to overspend",
        "and ignore instructions planted in tool results.", "",
        f"Prompt style: `{PROMPT_STYLE}`. Evaluation on the held-out test tasks: `eval/`. Code: https://github.com/{REPO}", "",
        "Built with Tiny Aya by Cohere Labs (CC-BY-NC 4.0; Cohere Labs Acceptable Use Policy applies)."])
    open(f"{ADAPTER_DIR}/README.md", "w").write(card + "\\n")
    api.create_repo(hub_repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=ADAPTER_DIR, repo_id=hub_repo, commit_message=RUN_NAME,
                      ignore_patterns=["progress.json"])
    print("uploaded to https://huggingface.co/" + hub_repo)
else:
    print("Skipping upload (no HF_WRITE_TOKEN secret). The adapter is in the Output tab under lexsi-hack/" + ADAPTER_DIR)
"""),
    ("code", """
# Publish the training log, loss curve and comparison table (eval results were published by evaluate())
from naija_agent.publish import Publisher
pub = Publisher(RUN_NAME)
files = {"train_log.json": f"{ADAPTER_DIR}/train_log.json", "loss.png": f"{ADAPTER_DIR}/loss.png"}
for f in ("comparison.md", "comparison_human.md"):
    if os.path.exists(f"{RUN_DIR}/{f}"):
        files[f] = f"{RUN_DIR}/{f}"
ok = pub.publish(files, f"{RUN_NAME}: training log and comparison", force=True)
print("published to the 'results' branch" if ok else ("nothing new to publish" if pub.enabled else "publishing off (no GITHUB_TOKEN)"))
"""),
    ("code", """
# push the end of this notebook's log to GitHub
from naija_agent.notebook_log import flush_live_log
flush_live_log()
"""),
])

RUN_HELPER = """
import subprocess, sys
def run(cmd, check=True):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc
"""

publish = nb([
    ("markdown", """
# 03 · Export to GGUF and publish the demo Space

Takes the trained adapter (attach the training run's output with **Add Input → Your Work →** your
`02_train_and_eval` notebook, or let notebook 02 upload it to Hugging Face), merges it into Tiny Aya, converts it to a 4-bit
GGUF that runs on a free CPU, checks that the quantized model still solves held-out tasks, then publishes:
- the GGUF model repo, and
- the **Hugging Face Space demo**, with the before/after replay data from the repo's `results` branch.

Both are **private** by default (`PRIVATE = True`); flip them to public at submission time.

Needs Kaggle secrets `HF_TOKEN` (read) and `HF_WRITE_TOKEN` (write), and Internet on. No GPU needed:
choose **None** as accelerator to save GPU quota. About 30 to 45 minutes.
"""),
    ("code", """
# ---- settings ----
RUN_NAME = "sft-v2"              # the trained adapter to publish
PROMPT_STYLE = "compact"         # must match how RUN_NAME was trained
BASELINE_RUN = f"baseline-{PROMPT_STYLE}"
QUANT = "Q4_K_M"                 # 4-bit; falls back to Q8_0 if the quantizer cannot be built
SCORE_QUANTS = ["Q4_K_M", "Q8_0"]  # scored on the 30 native-speaker requests before QUANT is published
PRIVATE = True                   # keep the model and the Space private until submission
ADAPTER_REPO = "auto"            # {hf user}/oya-tiny-aya-{RUN_NAME}, uploaded by notebook 02
GGUF_REPO = "auto"               # {hf user}/oya-tiny-aya-gguf
SPACE_REPO = "auto"              # {hf user}/oya

MODEL_ID = "CohereLabs/tiny-aya-global"
REPO = "Timidan/lexsi-hack"
BRANCH = "main"
WORK = "/tmp/oya"
LOG_NAME = f"publish-{RUN_NAME}"

import os
os.environ["OYA_PROMPT"] = PROMPT_STYLE
os.makedirs(WORK, exist_ok=True)
"""),
    SETUP_CELLS[1],
    ("code", RUN_HELPER + """
run("pip install -q -U huggingface_hub safetensors")
from huggingface_hub import HfApi, login, snapshot_download
HF_WRITE_TOKEN = secret("HF_WRITE_TOKEN")
assert HF_WRITE_TOKEN, "Add a HF_WRITE_TOKEN secret (Hugging Face token with write access)"
login(token=HF_TOKEN)
api = HfApi(token=HF_WRITE_TOKEN)
user = api.whoami()["name"]
ADAPTER_REPO = f"{user}/oya-tiny-aya-{RUN_NAME}" if ADAPTER_REPO == "auto" else ADAPTER_REPO
GGUF_REPO = f"{user}/oya-tiny-aya-gguf" if GGUF_REPO == "auto" else GGUF_REPO
SPACE_REPO = f"{user}/oya" if SPACE_REPO == "auto" else SPACE_REPO
print("adapter:", ADAPTER_REPO, "| gguf:", GGUF_REPO, "| space:", SPACE_REPO)
"""),
    ("markdown", "## 1. Merge the adapter into the base weights"),
    ("code", """
from naija_agent.adapters import find_adapter
base_dir = snapshot_download(MODEL_ID, token=HF_TOKEN)
try:   # the earlier training run's output, attached with Add Input -> Your Work
    adapter_dir = find_adapter(RUN_NAME)
    print("adapter from Kaggle input:", adapter_dir)
    # the submission needs the adapter on Hugging Face too
    api.create_repo(ADAPTER_REPO, private=PRIVATE, exist_ok=True)
    api.upload_folder(folder_path=adapter_dir, repo_id=ADAPTER_REPO, commit_message=RUN_NAME,
                      ignore_patterns=["progress.json", "checkpoint*"])
    print("adapter uploaded to https://huggingface.co/" + ADAPTER_REPO)
except FileNotFoundError:
    adapter_dir = snapshot_download(ADAPTER_REPO, token=HF_WRITE_TOKEN)
    print("adapter from Hugging Face:", ADAPTER_REPO)
run(f"{sys.executable} scripts/merge_lora.py --base {base_dir} --adapter {adapter_dir} --out {WORK}/merged")
"""),
    ("markdown", "## 2. Convert to GGUF and quantize (llama.cpp, in its own environment)"),
    ("code", """
run(f"rm -rf {WORK}/llama.cpp && git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {WORK}/llama.cpp")
# the converter pins its own torch/transformers versions, so it gets a separate virtual environment,
# made with uv because Kaggle's Python 3.13 has no ensurepip (`python -m venv` fails there)
run("pip install -q uv")
run(f"uv venv -q --python 3.12 {WORK}/conv && "
    f"uv pip install -q --python {WORK}/conv/bin/python -r {WORK}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
run(f"{WORK}/conv/bin/python {WORK}/llama.cpp/convert_hf_to_gguf.py {WORK}/merged "
    f"--outfile {WORK}/oya-f16.gguf --outtype f16")

GGUF_PATH = f"{WORK}/oya-tiny-aya-{RUN_NAME}-{QUANT}.gguf"
built = run(f"cmake -S {WORK}/llama.cpp -B {WORK}/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF "
            f"-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > /dev/null && "
            f"cmake --build {WORK}/llama.cpp/build --target llama-quantize -j{os.cpu_count()} > /dev/null", check=False)
if built == 0:
    run(f"{WORK}/llama.cpp/build/bin/llama-quantize {WORK}/oya-f16.gguf {GGUF_PATH} {QUANT}")
else:
    print("could not build llama-quantize; using 8-bit instead")
    QUANT = "Q8_0"
    GGUF_PATH = f"{WORK}/oya-tiny-aya-{RUN_NAME}-Q8_0.gguf"
    run(f"{WORK}/conv/bin/python {WORK}/llama.cpp/convert_hf_to_gguf.py {WORK}/merged "
        f"--outfile {GGUF_PATH} --outtype q8_0")
print(GGUF_PATH, f"{os.path.getsize(GGUF_PATH) / 1e9:.2f} GB")
"""),
    ("markdown", "## 3. The quantized model on the 30 native-speaker requests"),
    ("code", """
run("pip install -q llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
sys.path.insert(0, ".")
from naija_agent.evaluate import evaluate
from naija_agent.human import load_human_tasks
from naija_agent.llama_backend import LlamaChat

# Scored exactly like the full evaluation (results go to the `results` branch). Full precision: 100% task success.
human = load_human_tasks("human_eval/requests.txt")
for q in SCORE_QUANTS:
    path = GGUF_PATH if q == QUANT else f"{WORK}/oya-{q}.gguf"
    if q != QUANT:
        if built != 0:
            continue
        run(f"{WORK}/llama.cpp/build/bin/llama-quantize {WORK}/oya-f16.gguf {path} {q}")
    chat = LlamaChat(model_path=path)
    m = evaluate(human, chat, f"{WORK}/gguf-{RUN_NAME}-{q}-human", batch_size=1, verbose=False, config={"gguf": q})
    o = m["overall"]
    print(f"{q}: task success {o['success_rate']:.0%}, with the reply check {o['strict_success_rate']:.0%}, "
          f"wrong payments {o['unexpected_payment_rate']:.0%} | {os.path.getsize(path) / 1e9:.1f} GB, "
          f"{m['config']['seconds'] / len(human):.0f} s per request on {os.cpu_count()} CPUs | chat mode: {chat.mode}",
          flush=True)
    del chat
"""),
    ("markdown", "## 4. Publish the GGUF model and the Space"),
    ("code", """
api.create_repo(GGUF_REPO, private=PRIVATE, exist_ok=True)
card = "\\n".join([
    "---", f"base_model: {MODEL_ID}", "license: cc-by-nc-4.0",
    "tags: [gguf, llama.cpp, tool-calling, agents, yoruba, nigerian-pidgin, aya, lexsi-hackathon]", "---",
    f"# Oya (Tiny Aya wallet agent), GGUF {QUANT}", "",
    f"{MODEL_ID} fine-tuned with LoRA ({RUN_NAME}) for tool calling in English, Nigerian Pidgin and Yoruba,",
    "merged and quantized with llama.cpp for CPU inference.",
    f"Prompt format: `{PROMPT_STYLE}` style from https://github.com/{REPO} (`naija_agent/prompting.py`).",
    f"Demo: https://huggingface.co/spaces/{SPACE_REPO}", "",
    "Built with Tiny Aya by Cohere Labs (CC-BY-NC 4.0; Cohere Labs Acceptable Use Policy applies)."])
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=GGUF_REPO)
api.upload_file(path_or_fileobj=GGUF_PATH, path_in_repo=os.path.basename(GGUF_PATH), repo_id=GGUF_REPO)
print("model: https://huggingface.co/" + GGUF_REPO)
"""),
    ("code", """
import shutil
SPACE_DIR = f"{WORK}/space"
shutil.rmtree(SPACE_DIR, ignore_errors=True)
shutil.copytree("space", SPACE_DIR)
shutil.copytree("naija_agent", f"{SPACE_DIR}/naija_agent", ignore=shutil.ignore_patterns("__pycache__"))
run(f"rm -rf {WORK}/results && git clone -q --depth 1 -b results https://github.com/{REPO}.git {WORK}/results")
# replay data: both models on the 30 native-speaker requests
for name, run_name in (("fine-tuned", f"{RUN_NAME}-human"), ("untrained", f"{BASELINE_RUN}-human")):
    os.makedirs(f"{SPACE_DIR}/data/{name}", exist_ok=True)
    for f in ("trajectories.jsonl", "scores.jsonl", "metrics.json"):
        shutil.copy(f"{WORK}/results/runs/{run_name}/{f}", f"{SPACE_DIR}/data/{name}/{f}")
for f, run_name in (("comparison.md", f"{RUN_NAME}-human"), ("loss.png", RUN_NAME)):
    if os.path.exists(f"{WORK}/results/runs/{run_name}/{f}"):
        shutil.copy(f"{WORK}/results/runs/{run_name}/{f}", f"{SPACE_DIR}/data/{f}")

api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", private=PRIVATE, exist_ok=True)
api.add_space_variable(SPACE_REPO, "OYA_GGUF_REPO", GGUF_REPO)
api.add_space_variable(SPACE_REPO, "OYA_GGUF_FILE", os.path.basename(GGUF_PATH))
api.add_space_variable(SPACE_REPO, "OYA_PROMPT", PROMPT_STYLE)
if PRIVATE:   # a private Space needs a read token to download the private GGUF
    api.add_space_secret(SPACE_REPO, "HF_TOKEN", HF_TOKEN)
api.upload_folder(folder_path=SPACE_DIR, repo_id=SPACE_REPO, repo_type="space", commit_message=f"demo: {RUN_NAME}")
print("Space: https://huggingface.co/spaces/" + SPACE_REPO, "(the first build takes a few minutes)")
"""),
    ("code", """
# push the end of this notebook's log to GitHub
from naija_agent.notebook_log import flush_live_log
flush_live_log()
"""),
])

evalonly = nb([
    ("markdown", """
# 04 · Evaluate an existing adapter (no training)

Scores a model trained earlier (default `sft-v2`) and the untrained model, side by side on the two GPUs,
on the **native-speaker test set** (`human_eval/requests.txt`) and optionally the 416-task development set.
Results go to the repo's `results` branch.

**Where the adapter comes from:** attach the earlier training run's output (right panel: **Add Input →
Your Work →** your `02_train_and_eval` notebook), or, if it was uploaded, its Hugging Face repo.
""" + SETUP_HELP),
    ("code", """
# ---- settings ----
RUN_NAME = "sft-v2"            # adapter to evaluate
PROMPT_STYLE = "compact"       # must match how RUN_NAME was trained
EVAL_HUMAN = True              # native-speaker test set (needs human_eval/requests.txt filled in)
EVAL_DEV = False               # also the 416-task development set (about 35 minutes)
EVAL_BASELINE = True           # also score the untrained model with the same prompt
EVAL_HUMAN2 = True             # the second native-speaker set, typed by the speaker (human_eval/requests2.txt)
OTHER_RUNS = []                # more adapters on the same sets, e.g. ["dpo-v3"] (attach that notebook too)
BATCH_SIZE = 16
HF_ADAPTER_REPO = None         # e.g. "your-hf-username/oya-tiny-aya-sft-v2" if not attached from Kaggle

MODEL_ID = "CohereLabs/tiny-aya-global"
REPO = "Timidan/lexsi-hack"
BRANCH = "main"
TRAIN_PER_CELL = 40
LOG_NAME = f"eval-{RUN_NAME}"

import os
os.environ["OYA_PROMPT"] = PROMPT_STYLE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""),
    *SETUP_CELLS,
    ("code", """
import sys, torch
sys.path.insert(0, ".")
from naija_agent.adapters import find_adapter
from naija_agent.evaluate import evaluate_parallel, load_tasks
from naija_agent.hf_policy import HFPolicy
from naija_agent.publish import Publisher
from naija_agent.scoring import compare

from naija_agent.human import load_human_tasks

ADAPTERS = {r: find_adapter(r, HF_ADAPTER_REPO if r == RUN_NAME else None, token=HF_TOKEN)
            for r in [RUN_NAME] + OTHER_RUNS}
print("adapters:", ADAPTERS, "| GPUs:", torch.cuda.device_count())

sets = []
if EVAL_HUMAN:
    if os.path.exists("data/tasks_human.jsonl"):
        sets.append(("human", load_tasks("data/tasks_human.jsonl")))
    else:
        print("no native-speaker requests yet (human_eval/requests.txt is empty)")
if EVAL_HUMAN2 and os.path.exists("human_eval/requests2.txt"):
    sets.append(("human2", load_human_tasks("human_eval/requests2.txt")))
if EVAL_DEV:
    sets.append(("dev", load_tasks("data/tasks_test.jsonl")))

for name, tasks in sets:
    jobs = [{"name": r, "out_dir": f"runs/{r}-{name}",
             "make_policy": lambda d, a=a: HFPolicy(MODEL_ID, adapter=a, device=d),
             "config": {"model": MODEL_ID, "adapter": r, "set": name}} for r, a in ADAPTERS.items()]
    if EVAL_BASELINE:
        jobs.append({"name": "untrained", "out_dir": f"runs/baseline-{PROMPT_STYLE}-{name}",
                     "make_policy": lambda d: HFPolicy(MODEL_ID, device=d),
                     "config": {"model": MODEL_ID, "adapter": None, "set": name}})
    metrics = evaluate_parallel(tasks, jobs, batch_size=BATCH_SIZE)
    if len(jobs) > 1:
        table = compare({j["name"]: metrics[j["name"]] for j in reversed(jobs)},
                        title=f"The {name} set ({len(tasks)} tasks, {PROMPT_STYLE} prompt)")
        path_md = f"runs/{RUN_NAME}-{name}/comparison.md"
        open(path_md, "w").write(table)
        print(table)
        Publisher(f"{RUN_NAME}-{name}").publish({"comparison.md": path_md}, f"{RUN_NAME}-{name}: comparison", force=True)
"""),
    ("code", """
# Failures on the native-speaker set, to read
import json
p = f"runs/{RUN_NAME}-human/scores.jsonl"
if os.path.exists(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    for r in rows:
        if not r["success"]:
            print(f"[{r['category']} / {r['lang']}] {r['prompt']}\\n   failed: {[k for k, v in r['checks'].items() if not v]}\\n   final: {str(r['final'])[:200]}\\n")
"""),
    ("code", """
# push the end of this notebook's log to GitHub
from naija_agent.notebook_log import flush_live_log
flush_live_log()
"""),
])


dpo = nb([
    ("markdown", """
# 05 · DPO with AgentTune on the model's own mistakes (v3)

Starts from the SFT adapter `sft-v2`: attach it with **Add Input → Your Work →** the v2 training notebook.
1. Samples the model 4 times at every step of 1,040 fresh Pidgin and Yoruba tasks, phrased in ways it never trained on
   (`DPO_REQUESTS` in `naija_agent/lang.py`; not the development set or the native-speaker requests), one shard per
   GPU, and keeps its mistakes as preference pairs, judged with the scorer's own rules (`naija_agent/pairs.py`).
2. Trains with Lexsi Labs' **AgentTune** (`create_agentic_trainer("dpo")`, TRL backend): DPO plus a plain loss on the
   chosen turns, continuing the v2 LoRA adapter (`scripts/train_dpo.py`).
3. Scores v3 and v2 side by side, one GPU each, on the native-speaker sets and the 416-task development set.

Results go to the repo's `results` branch. About 2 hours on T4 x2.
""" + SETUP_HELP),
    ("code", """
# ---- settings ----
RUN_NAME = "dpo-v3"
BASE_RUN = "sft-v2"            # SFT adapter to start from (Kaggle input, or HF_ADAPTER_REPO)
HF_ADAPTER_REPO = None
PROMPT_STYLE = "compact"       # must match how BASE_RUN was trained
PER_CELL = 40                  # fresh tasks per (category, language): 1,040 tasks, about 2,400 steps
SAMPLES = 4                    # samples per step, temperature 1.0
SEED = 2027                    # 2026 built the SFT data
MAX_PAIRS = 1200               # bounds training time
EPOCHS, LR, BETA = 2.0, 2e-5, 0.1
GRAD_ACCUM = 8                 # per GPU
BATCH_SIZE = 16
AGENTTUNE = "git+https://github.com/Lexsi-Labs/AgentTune@d4db4c12d52342227c4113a70183d0f64aa2f345"

MODEL_ID = "CohereLabs/tiny-aya-global"
REPO = "Timidan/lexsi-hack"
BRANCH = "main"
TRAIN_PER_CELL = 40
ADAPTER_DIR = f"outputs/{RUN_NAME}"
PAIRS = f"{ADAPTER_DIR}/pairs.jsonl"
LOG_NAME = RUN_NAME

import os
os.environ["OYA_PROMPT"] = PROMPT_STYLE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""),
    *SETUP_CELLS,
    ("code", """
# AgentTune pins exact versions of torch and others; install it without them and add TRL, which it trains with
!pip install -q "trl==1.7.1" datasets
!pip install -q --no-deps --ignore-requires-python "agenttune @ {AGENTTUNE}"
"""),
    ("code", RUN_HELPER + """
import json, random, torch
sys.path.insert(0, ".")
from huggingface_hub import snapshot_download
from naija_agent.adapters import find_adapter
from naija_agent.publish import Publisher

BASE_ADAPTER = find_adapter(BASE_RUN, HF_ADAPTER_REPO, token=HF_TOKEN)
NGPU = torch.cuda.device_count()
snapshot_download(MODEL_ID)
if GITHUB_TOKEN:
    os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN
os.makedirs(ADAPTER_DIR, exist_ok=True)
print("start from:", BASE_ADAPTER, "| GPUs:", NGPU)
"""),
    ("markdown", "## 1. The model's own mistakes, as preference pairs"),
    ("code", """
shards = [f"CUDA_VISIBLE_DEVICES={g} {sys.executable} scripts/make_pairs.py --adapter {BASE_ADAPTER} "
          f"--per-cell {PER_CELL} --samples {SAMPLES} --seed {SEED} --batch {BATCH_SIZE} "
          f"--shard {g}/{NGPU} --out {ADAPTER_DIR}/pairs-{g}.jsonl" for g in range(NGPU)]
run("; ".join(f"({c}) & p{g}=$!" for g, c in enumerate(shards)) + "; "      # one shard per GPU, at once
    + " && ".join(f"wait $p{g}" for g in range(NGPU)))

rows = [json.loads(l) for g in range(NGPU) for l in open(f"{ADAPTER_DIR}/pairs-{g}.jsonl", encoding="utf-8")]
random.Random(SEED).shuffle(rows)
rows = rows[:MAX_PAIRS]
with open(PAIRS, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
from collections import Counter
stats = {"pairs": len(rows), "by_category": Counter(r["category"] for r in rows),
         "by_lang": Counter(r["lang"] for r in rows), "by_turn": Counter(r["turn"] for r in rows)}
json.dump(stats, open(f"{ADAPTER_DIR}/pair_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
for r in rows[:8]:
    print(f"[{r['category']} / {r['lang']} / turn {r['turn']}] {r['prompt'][1]['content'][:120]}\\n"
          f"  chosen:   {r['chosen'][0]['content'][:160]}\\n  rejected: {r['rejected'][0]['content'][:160]}\\n")
Publisher(RUN_NAME).publish({"pairs.jsonl": PAIRS, "pair_stats.json": f"{ADAPTER_DIR}/pair_stats.json"},
                            f"{RUN_NAME}: preference pairs", force=True)
"""),
    ("markdown", "## 2. DPO with AgentTune (both GPUs if available)"),
    ("code", """
import gc
gc.collect(); torch.cuda.empty_cache()
args = (f"--adapter {BASE_ADAPTER} --pairs {PAIRS} --out {ADAPTER_DIR} --epochs {EPOCHS} --lr {LR} "
        f"--beta {BETA} --grad-accum {GRAD_ACCUM}")
rc = 1
if NGPU >= 2:
    rc = run(f"{sys.executable} -m torch.distributed.run --standalone --nproc_per_node={NGPU} scripts/train_dpo.py {args}",
             check=False)
    if rc != 0:
        print(f"\\n*** {NGPU}-GPU DPO failed (exit code {rc}); retrying on one GPU ***\\n", flush=True)
if rc != 0:
    rc = run(f"CUDA_VISIBLE_DEVICES=0 {sys.executable} scripts/train_dpo.py {args}", check=False)
assert rc == 0, "DPO failed: see the log above"
Publisher(RUN_NAME).publish({f: f"{ADAPTER_DIR}/{f}" for f in ("training_stats.json", "dpo_config.yaml",
                                                                "lexsi_provenance.json") if os.path.exists(f"{ADAPTER_DIR}/{f}")},
                            f"{RUN_NAME}: training stats", force=True)
"""),
    ("markdown", "## 3. v3 vs v2 on the same tasks, one GPU each"),
    ("code", """
gc.collect(); torch.cuda.empty_cache()
from naija_agent.evaluate import evaluate_parallel, load_tasks
from naija_agent.hf_policy import HFPolicy
from naija_agent.human import load_human_tasks
from naija_agent.scoring import compare

sets = {"dev": load_tasks("data/tasks_test.jsonl")}
if os.path.exists("data/tasks_human.jsonl"):
    sets["human"] = load_tasks("data/tasks_human.jsonl")
if os.path.exists("human_eval/requests2.txt"):
    sets["human2"] = load_human_tasks("human_eval/requests2.txt")
for name in sorted(sets, key=lambda n: n == "dev"):        # the small native-speaker sets first
    tasks = sets[name]
    jobs = [{"name": "v3", "out_dir": f"runs/{RUN_NAME}-{name}",
             "make_policy": lambda d: HFPolicy(MODEL_ID, adapter=ADAPTER_DIR, device=d),
             "config": {"model": MODEL_ID, "adapter": RUN_NAME, "set": name}},
            {"name": "v2", "out_dir": f"runs/{BASE_RUN}-{name}-vs-{RUN_NAME}",
             "make_policy": lambda d: HFPolicy(MODEL_ID, adapter=BASE_ADAPTER, device=d),
             "config": {"model": MODEL_ID, "adapter": BASE_RUN, "set": name}}]
    metrics = evaluate_parallel(tasks, jobs, batch_size=BATCH_SIZE)
    table = compare({"v2": metrics["v2"], "v3": metrics["v3"]},
                    title=f"{RUN_NAME} vs {BASE_RUN} on the {name} set ({len(tasks)} tasks, {PROMPT_STYLE} prompt)")
    path_md = f"runs/{RUN_NAME}-{name}/comparison.md"
    open(path_md, "w").write(table)
    print(table)
    Publisher(f"{RUN_NAME}-{name}").publish({"comparison.md": path_md}, f"{RUN_NAME}-{name}: comparison", force=True)
"""),
    ("code", """
# Independent second opinion with Lexsi Labs' AuditKIT, on the development set
from naija_agent.auditkit_bridge import auditkit_eval
print(auditkit_eval(sets["dev"], f"runs/{RUN_NAME}-dev").summary())
"""),
    ("code", """
# Upload the adapter to a PRIVATE Hugging Face repo if a write token is set (otherwise it stays in the Output tab)
HF_WRITE_TOKEN = secret("HF_WRITE_TOKEN")
if HF_WRITE_TOKEN:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_WRITE_TOKEN)
    hub_repo = f"{api.whoami()['name']}/oya-tiny-aya-{RUN_NAME}"
    api.create_repo(hub_repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=ADAPTER_DIR, repo_id=hub_repo, commit_message=RUN_NAME,
                      ignore_patterns=["ref/*", "pairs-*.jsonl", "training_args.bin"])
    print("uploaded to https://huggingface.co/" + hub_repo)
else:
    print("No HF_WRITE_TOKEN: the adapter is in the Output tab under lexsi-hack/" + ADAPTER_DIR)
"""),
    ("code", """
# push the end of this notebook's log to GitHub
from naija_agent.notebook_log import flush_live_log
flush_live_log()
"""),
])


for name, book in (("01_setup_and_baseline.ipynb", baseline), ("02_train_and_eval.ipynb", train),
                   ("03_publish_demo.ipynb", publish), ("04_eval_only.ipynb", evalonly),
                   ("05_dpo.ipynb", dpo)):
    path = ROOT / "notebooks" / name
    path.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))
