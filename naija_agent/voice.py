"""Speech to text in front of Oya: Whisper-family models, one clip at a time.

Used by scripts/voice_bench.py and the Space (GPU, fp16; the Space loads both models at startup).
Needs torch, transformers and peft, plus the ffmpeg binary for decoding audio or video.
"""

from __future__ import annotations

import subprocess

import numpy as np

SR = 16_000

# lang maps the clip language (Y = Yoruba, P = Pidgin) to the decoder's language token (None: the
# model's own prompt). generate holds decoding settings a model card recommends; default is greedy.
MODELS = {
    "whisper-turbo": {"repo": "openai/whisper-large-v3-turbo", "lang": {"Y": "<|yo|>", "P": "<|en|>"}},
    "naijavox": {"repo": "Axiveri/NaijaVox-V1", "lang": {"Y": "<|yo|>", "P": "<|pcm|>"}},
    "whisper-pidgin": {"repo": "openai/whisper-large-v3-turbo", "adapter": "michaelodafe/whisper-pidgin-v1",
                       "lang": {"P": "<|en|>"}},
    "buzzasr-yoruba": {"repo": "BuzzASR/yoruba", "lang": {"Y": None},
                       "generate": {"no_repeat_ngram_size": 3, "repetition_penalty": 1.2}},
}


def load_audio(path: str, max_seconds: int = 30) -> np.ndarray:
    """Any audio or video file -> 16 kHz mono float32, first max_seconds only (one Whisper window)."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-t", str(max_seconds), "-vn",
                          "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


class Transcriber:
    def __init__(self, name: str, device: str = "cpu"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.name, self.cfg, self.device = name, MODELS[name], device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = WhisperProcessor.from_pretrained(self.cfg["repo"])
        model = WhisperForConditionalGeneration.from_pretrained(self.cfg["repo"], dtype=self.dtype,
                                                                attn_implementation="sdpa")
        if self.cfg.get("adapter"):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.cfg["adapter"], torch_device="cpu").merge_and_unload()
        self.model = model.to(device).eval()
        self.vocab = self.proc.tokenizer.get_vocab()

    def __call__(self, audio: np.ndarray, lang: str) -> str:
        import torch

        feats = self.proc.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
        kw = dict(self.cfg.get("generate", {}), max_new_tokens=200)
        token = self.cfg["lang"][lang]
        if token:
            kw["decoder_input_ids"] = torch.tensor([[self.vocab["<|startoftranscript|>"], self.vocab[token],
                                                     self.vocab["<|transcribe|>"], self.vocab["<|notimestamps|>"]]],
                                                   device=self.device)
        with torch.no_grad():
            ids = self.model.generate(input_features=feats.to(self.device, self.dtype), **kw)
        return " ".join(self.proc.tokenizer.decode(ids[0], skip_special_tokens=True).split())
