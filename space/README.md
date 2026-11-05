---
title: Oya · Naija wallet agent
emoji: 💸
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Tiny Aya tool-calling agent for Pidgin and Yoruba payments
models:
  - CohereLabs/tiny-aya-global
---

# Oya: a Tiny Aya wallet agent for English, Nigerian Pidgin and Yoruba

Fine-tuned from Cohere Labs' Tiny Aya (3.35B) with LoRA for the Lexsi Labs AI Safety & Alignment Hackathon
(Track 3, Agentic Fine-tuning). It buys airtime and data, pays electricity and sends money through tools,
asks when details are missing, refuses to overspend, and ignores instructions planted in tool results.

All tools run against a mock wallet: no real money, accounts or phone numbers are involved.

- **Try it**: chat with the fine-tuned model (quantized GGUF, running on this Space's CPU). Type a request, or
  speak it: a voice note or video in Pidgin (whisper-pidgin-v1) or Yoruba (BuzzASR) becomes text you can
  correct before sending. Every payment waits for you to confirm or cancel it.
- **Before / after**: every held-out test task, untrained vs fine-tuned, with each tool call and a verdict.
- **Results**: the before/after metrics.

Code, data generator and evaluation: https://github.com/Timidan/lexsi-hack

Built with Tiny Aya by Cohere Labs (CC-BY-NC 4.0, subject to Cohere Labs' Acceptable Use Policy),
and Lexsi Labs' AgentTune (prompt and trajectory format) and AuditKIT (evaluation).
