"""Write docs/language_review.md: every Pidgin and Yoruba string in one sheet
for native-speaker review. Rerun after editing naija_agent/lang.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naija_agent import lang as L  # noqa: E402
from naija_agent.tasks import template_split  # noqa: E402

NAMES = {"pcm": "Pidgin", "yo": "Yoruba", "yo_en": "Yoruba-English mix", "en": "English"}

lines = ["# Language review sheet", "",
         "Every user request and agent reply template, for native-speaker review.",
         "Placeholders like `{amount}` are filled in automatically. Reply with the template id and your fix.",
         "Rows marked **test** are held out from training and used only for evaluation.", ""]
lines += ["## User requests", ""]
for cat, by_lang in L.REQUESTS.items():
    lines += [f"### {cat}", "", "| id | lang | split | text |", "|---|---|---|---|"]
    for lng in ("pcm", "yo", "yo_en", "en"):
        for i, t in enumerate(by_lang[lng]):
            tid = f"{cat}.{lng}.{i}"
            text = t.split("|", 1)[-1].replace("|", "\\|")
            lines.append(f"| `{tid}` | {NAMES[lng]} | {template_split(tid)} | {text} |")
    lines.append("")
lines += ["## Agent replies", "", "| id | lang | text |", "|---|---|---|"]
for key, by_lang in L.REPLIES.items():
    for lng in ("pcm", "yo", "en"):
        lines.append(f"| `{key}` | {NAMES[lng]} | {by_lang[lng]} |")
for lng in ("pcm", "yo", "en"):
    lines.append(f"| `injection_note` | {NAMES[lng]} | {L.INJECTION_NOTE[lng].strip()} |")
out = Path(__file__).resolve().parents[1] / "docs" / "language_review.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out}")
