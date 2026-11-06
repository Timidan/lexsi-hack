"""Compare Oya with other models on the 30 held-out requests: same prompt, same tasks, same scorer.

    python scripts/compare_models.py run qwen3-4b gpt-6-luna ...     # -> runs/compare-<name>/
    python scripts/compare_models.py table oya tiny-aya qwen3-4b ...  # -> runs/compare.md, runs/compare.json

Open models run in 4-bit on the local GPU (HFPolicy); OpenAI models through the API, with the key read
from the file named by OPENAI_KEY_FILE (never printed or saved). A model already scored in
runs/compare-<name>/ is reused. API cost uses list prices per 1M tokens as reported by third-party
pricing pages (not OpenAI's own page), so treat it as an estimate.

"Reviewed success" also counts tasks whose only failed check is "asked" and whose reply has no question
mark: polite requests such as "Abeg send im bank name and account number". Every such case in the
published comparison was read by hand; the count per model is "polite_asks" in runs/compare.json.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from naija_agent.evaluate import evaluate, load_tasks  # noqa: E402
from naija_agent.scoring import reply_ok  # noqa: E402
from naija_agent.tasks import SAFETY_CATEGORIES  # noqa: E402

TASKS = ROOT / "data" / "tasks_human.jsonl"
ADAPTER = os.environ.get("OYA_ADAPTER", str(ROOT / "outputs" / "sft-v2"))

# name: (label, size, Hugging Face repo, adapter) for open models, run in 4-bit (bitsandbytes nf4) on a 4 GB
# laptop GPU; the "-bnb-4bit" repos are the same models stored already quantized, which is quicker to download
OPEN = {
    "oya": ("Oya (Tiny Aya + our LoRA)", "3.35B", "CohereLabs/tiny-aya-global", ADAPTER),
    "tiny-aya": ("Tiny Aya untrained", "3.35B", "CohereLabs/tiny-aya-global", None),
    "qwen3-4b": ("Qwen3 4B Instruct", "4.0B", "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit", None),
    "phi-4-mini": ("Phi-4 mini", "3.8B", "unsloth/Phi-4-mini-instruct-bnb-4bit", None),
    "llama-3.2-3b": ("Llama 3.2 3B", "3.2B", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", None),
}
# name: (label, size, runs on, run dir under runs/) for Oya formats scored on the same 30 tasks by the notebooks.
# Copy those runs from the results branch first: git show origin/results:runs/<dir>/<file> > runs/<dir>/<file>
# for metrics.json and scores.jsonl.
SCORED = {
    "oya-16bit": ("Oya (Tiny Aya + our LoRA)", "3.35B", "Kaggle T4, 16-bit", "sft-v2-human"),
    "oya-gguf": ("Oya (Tiny Aya + our LoRA)", "3.35B", "4 CPUs, GGUF Q4_K_M", "gguf-sft-v2-Q4_K_M-human"),
}
# name: (label, USD per 1M input tokens, per 1M output tokens)
API = {
    "gpt-6-astra": ("GPT-6 Astra", 10.0, 50.0),
    "gpt-6.1-sol": ("GPT-6.1 Sol", 2.0, 10.0),
    "gpt-6-luna": ("GPT-6 Luna", 0.10, 0.50),
}


class OpenAIPolicy:
    """Chat Completions, one request per conversation, a batch in parallel. Low reasoning effort."""

    def __init__(self, model: str):
        self.model, self.key = model, Path(os.environ["OPENAI_KEY_FILE"]).read_text().strip()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        self.effort = "low"

    def __call__(self, batch, tasks=None):
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(self._one, batch))

    def _one(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "max_completion_tokens": 2048}
        if self.effort:
            body["reasoning_effort"] = self.effort
        for attempt in range(6):
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {self.key}",
                                                  "Content-Type": "application/json"})
            try:
                out = json.load(urllib.request.urlopen(req, timeout=120))
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code == 400 and "reasoning_effort" in detail and "reasoning_effort" in body:
                    self.effort = None  # model without reasoning controls
                    body.pop("reasoning_effort")
                    continue
                if e.code in (429, 500, 502, 503) and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"{self.model}: HTTP {e.code}: {detail}") from None
            u = out.get("usage", {})
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += u.get("completion_tokens", 0)
            self.usage["requests"] += 1
            return (out["choices"][0]["message"].get("content") or "").strip()
        raise RuntimeError(f"{self.model}: gave up after retries")


def run(name: str) -> None:
    out = ROOT / "runs" / f"compare-{name}"
    if (out / "metrics.json").exists():
        print(f"{name}: already scored in {out}")
        return
    if name in API:
        policy = OpenAIPolicy(name)
    else:
        from naija_agent.hf_policy import HFPolicy
        _, _, repo, adapter = OPEN[name]
        policy = HFPolicy(repo, adapter=adapter, load_in_4bit=not repo.endswith("bnb-4bit"))
    t0 = time.time()
    m = evaluate(load_tasks(TASKS), policy, out, batch_size=8 if name in API else 2, verbose=False, publish=False,
                 config={"model": name})
    if name in API:
        (out / "usage.json").write_text(json.dumps(policy.usage, indent=2))
    print(f"{name}: success {m['overall']['success_rate']:.1%} in {time.time() - t0:.0f}s", flush=True)


def polite_ask(r: dict) -> bool:
    """A failed task whose only failed check is "asked", with a reply that has no question mark."""
    return (not r["success"] and "?" not in (r["final"] or "")
            and [k for k, v in r["checks"].items() if not v] == ["asked"])


def reviewed_rate(scores: list[dict]) -> float:
    """Success, also counting polite requests that only missed the question mark (see module docstring)."""
    return sum(1 for r in scores if r["success"] or polite_ask(r)) / len(scores)


def table(names: list[str]) -> None:
    rows, data = [], []
    tasks = {t.id: t for t in load_tasks(TASKS)}
    for name in names:
        run_dir = ROOT / "runs" / (SCORED[name][3] if name in SCORED else f"compare-{name}")
        m = json.loads((run_dir / "metrics.json").read_text())
        o, s = m["overall"], m["safety"]
        scores = [json.loads(line) for line in open(run_dir / "scores.jsonl", encoding="utf-8")]
        for r in scores:  # the current reply check, also for runs saved before it existed
            r["reply_ok"] = reply_ok(tasks[r["task_id"]], r["final"])
        reviewed = reviewed_rate(scores)
        if name in API:
            label, pin, pout = API[name]
            u = json.loads((run_dir / "usage.json").read_text())
            cost = (u["prompt_tokens"] * pin + u["completion_tokens"] * pout) / 1e6 / o["n"] * 1000
            size, where = "not disclosed", "OpenAI API"
        elif name in SCORED:
            label, size, where, _ = SCORED[name]
            cost = 0.0
        else:
            label, size, cost, where = OPEN[name][0], OPEN[name][1], 0.0, "laptop GPU, 4-bit"
        row = {"name": name, "label": label, "size": size, "runs_on": where, "n": o["n"],
               "success": o["success_rate"], "reviewed_success": reviewed,
               "polite_asks": sum(polite_ask(r) for r in scores),
               "safety": reviewed_rate([r for r in scores if r["category"] in SAFETY_CATEGORIES]),
               "success_right_reply": sum(r["success"] and r["reply_ok"] for r in scores) / len(scores),
               "wrong_payment": o["unexpected_payment_rate"], "tool_f1": o["tool_f1"],
               "pidgin": reviewed_rate([r for r in scores if r["lang"] == "pcm"]),
               "yoruba": reviewed_rate([r for r in scores if r["lang"] == "yo"]),
               "usd_per_1000_requests": round(cost, 2)}
        data.append(row)
        rows.append(f"| {label} | {size} | {where} | {row['success']:.0%} | {reviewed:.0%} | {row['safety']:.0%} "
                    f"| {row['success_right_reply']:.0%} | {row['wrong_payment']:.0%} | {row['pidgin']:.0%} "
                    f"| {row['yoruba']:.0%} | {'$0 (self-hosted)' if name not in API else f'~${cost:.2f}'} |")
    md = ("| Model | Size | Runs on | Task success | Reviewed success | Safety tasks (reviewed) | Success and right reply "
          "| Tasks with a wrong payment | Pidgin (reviewed) | Yoruba (reviewed) | Cost per 1,000 requests |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (ROOT / "runs" / "compare.md").write_text(md)
    (ROOT / "runs" / "compare.json").write_text(json.dumps(data, indent=2))
    print(md)


if __name__ == "__main__":
    step, names = sys.argv[1], sys.argv[2:]
    if step == "run":
        for n in names:
            run(n)
    else:
        table(names)
