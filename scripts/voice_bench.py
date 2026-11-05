"""Voice benchmark: speech model -> transcript -> Oya.

The 26 held-out requests (human_eval/requests.txt) were read aloud by a native speaker; one
clip per line, named 1Y.wav ... 13P.wav. The clips stay private; transcripts are kept here.

    python scripts/voice_bench.py transcribe naijavox --clips DIR   # -> human_eval/voice/naijavox.txt
    python scripts/voice_bench.py score typed naijavox ... --adapter DIR   # --adapter none: untrained Tiny Aya

`score` reports word error rate, whether the details that decide where money goes survived
(amount, number, name, network), and Oya's task success and wrong payments on each transcript
set ("typed" = the written requests). Oya runs in 4-bit so it fits a 4 GB GPU; a set already
scored in runs/voice-<name>/ is reused (delete that folder to rerun it).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from naija_agent.human import load_human_tasks, parse_requests  # noqa: E402
from naija_agent.lang import strip_diacritics  # noqa: E402

VOICE = ROOT / "human_eval" / "voice"
TYPED = ROOT / "human_eval" / "requests.txt"

# Whisper-family models; lang maps the clip suffix (Y/P) to the decoder's language token (None: the
# model's own prompt). generate holds decoding settings a model card recommends; default is greedy.
MODELS = {
    "whisper-turbo": {"repo": "openai/whisper-large-v3-turbo", "lang": {"Y": "<|yo|>", "P": "<|en|>"}},
    "naijavox": {"repo": "Axiveri/NaijaVox-V1", "lang": {"Y": "<|yo|>", "P": "<|pcm|>"}},
    "whisper-pidgin": {"repo": "openai/whisper-large-v3-turbo", "adapter": "michaelodafe/whisper-pidgin-v1",
                       "lang": {"P": "<|en|>"}},
    "buzzasr-yoruba": {"repo": "BuzzASR/yoruba", "lang": {"Y": None},
                       "generate": {"no_repeat_ngram_size": 3, "repetition_penalty": 1.2}},
}

# What must survive transcription for the money to go to the right place, per scenario.
DETAILS = {1: ["08031234567", "1000", "mtn"], 2: ["09055512345", "2gb", "glo"], 3: ["45123456789", "5000", "ikeja"],
           4: ["mama", "2000"], 5: ["tunde", "1gb"], 6: ["0123456789", "15000", "gtbank"], 7: ["bisi", "3000"],
           10: ["kunle", "10000"], 11: ["98765432109", "50000", "eko"], 12: ["08123456789"]}


def norm(text: str) -> str:
    t = strip_diacritics(text).lower().replace("₦", " ").replace("gt bank", "gtbank")
    t = re.sub(r"(\d)[,\s-]+(?=\d)", r"\1", t)                       # 1,000 / 0 1 2 3 -> 1000 / 0123
    t = re.sub(r"(\d+)\s*k\b", lambda m: str(int(m.group(1)) * 1000), t)  # 5k -> 5000
    t = re.sub(r"(\d)\s*(gigabytes?|gb)\b", r"\1gb", t)
    t = re.sub(r"\bn(?=\d)|\b(naira|nera)\b", " ", t)
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def transcribe(name: str, clips: Path) -> None:
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    cfg = MODELS[name]
    proc = WhisperProcessor.from_pretrained(cfg["repo"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["repo"], dtype=torch.float16, attn_implementation="sdpa")
    if cfg.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter"]).merge_and_unload()
    model.to("cuda").eval()
    vocab = proc.tokenizer.get_vocab()
    lines, secs = [], []
    for scenario, suffix in [(n, s) for s in "YP" for n in range(1, 14)]:
        if suffix not in cfg["lang"]:
            continue
        audio, sr = sf.read(clips / f"{scenario}{suffix}.wav", dtype="float32")
        feats = proc.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features.to("cuda", torch.float16)
        kw = dict(cfg.get("generate", {}), max_new_tokens=200)
        if cfg["lang"][suffix]:
            kw["decoder_input_ids"] = torch.tensor([[vocab["<|startoftranscript|>"], vocab[cfg["lang"][suffix]],
                                                     vocab["<|transcribe|>"], vocab["<|notimestamps|>"]]], device="cuda")
        t0 = time.time()
        with torch.no_grad():
            ids = model.generate(input_features=feats, **kw)
        secs.append(time.time() - t0)
        text = " ".join(proc.tokenizer.decode(ids[0], skip_special_tokens=True).split()) or "-"
        lines.append(f"{scenario}{suffix}: {text}")
        print(lines[-1], flush=True)
    VOICE.mkdir(parents=True, exist_ok=True)
    (VOICE / f"{name}.txt").write_text(
        f"# {name} ({cfg['repo']}{' + ' + cfg['adapter'] if cfg.get('adapter') else ''}), "
        f"decoding: {cfg.get('generate') or 'greedy'}\n"
        f"# seconds_per_clip: {sum(secs) / len(secs):.2f} (RTX 3050 laptop GPU, fp16)\n" + "\n".join(lines) + "\n",
        encoding="utf-8")


def score(names: list[str], adapter: str) -> str:
    import jiwer
    from naija_agent.evaluate import evaluate
    from naija_agent.hf_policy import HFPolicy

    refs = {(n, lang): t for n, lang, t in parse_requests(TYPED.read_text(encoding="utf-8"))}
    policy = None
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    rows = ["| Speech model | Word errors: Pidgin | Word errors: Yoruba (no tone marks) | Key details exactly right "
            "| Oya success: Pidgin | Oya success: Yoruba | Oya success and right reply | Tasks with a wrong payment "
            "| Seconds per clip |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        path = TYPED if name == "typed" else VOICE / f"{name}.txt"
        text = path.read_text(encoding="utf-8")
        hyps = {(n, lang): t for n, lang, t in parse_requests(text)}
        wer = {}
        for lang in ("pcm", "yo"):
            keys = [k for k in hyps if k[1] == lang]
            wer[lang] = jiwer.wer([norm(refs[k]) for k in keys], [norm(hyps[k]) for k in keys]) if keys and name != "typed" else None
        checked = [(n, h) for (n, _), h in hyps.items() if n in DETAILS]
        kept = sum(all(d in norm(h) for d in DETAILS[n]) for n, h in checked) / len(checked)
        untrained = adapter == "none"
        run = ROOT / "runs" / f"voice-{name}{'-untrained' if untrained else ''}"
        if not (run / "metrics.json").exists():
            policy = policy or HFPolicy("CohereLabs/tiny-aya-global", adapter=None if untrained else adapter,
                                        load_in_4bit=True)
            evaluate(load_human_tasks(path), policy, run, batch_size=2, verbose=False, publish=False)
        m = json.loads((run / "metrics.json").read_text())
        by_lang = m["by_lang"]
        speed = re.search(r"seconds_per_clip: ([\d.]+)", text)
        rows.append(f"| {name}{' (untrained)' if untrained else ''} | {pct(wer['pcm'])} | {pct(wer['yo'])} | {pct(kept)} "
                    f"| {pct(by_lang.get('pcm', {}).get('success_rate'))} | {pct(by_lang.get('yo', {}).get('success_rate'))} "
                    f"| {pct(m['overall']['strict_success_rate'])} | {pct(m['overall']['unexpected_payment_rate'])} "
                    f"| {speed.group(1) if speed else '-'} |")
        print(rows[-1], flush=True)
    table = "\n".join(rows) + "\n"
    out = ROOT / "runs" / f"voice-comparison{'-untrained' if adapter == 'none' else ''}.md"
    out.write_text(table)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["transcribe", "score"])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--clips", type=Path)
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "sft-v2"))
    a = ap.parse_args()
    if a.step == "transcribe":
        for name in a.models:
            transcribe(name, a.clips)
    else:
        print(score(a.models, a.adapter))
