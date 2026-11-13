"""Leak check: is any test request a copy or near copy of a training request?

    python scripts/leak_check.py

Requests are normalised first: lower case, tone marks removed, every number replaced by 0, punctuation
dropped. A direct match is identical after that. A near copy has a Jaccard similarity of at least 0.8 with
some training request, over character 5-grams (the similarity MinHash estimates, computed exactly here).
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from naija_agent import generate  # noqa: E402
from naija_agent.human import load_human_tasks  # noqa: E402

NEAR = 0.8


def norm(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    text = re.sub(r"\d[\d,.]*", "0", text)
    return " ".join(re.sub(r"\W+", " ", text).split())


def grams(text: str, n: int = 5) -> set[str]:
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def main() -> None:
    # same sets and seeds as scripts/build_data.py: training chats plus the validation chats
    train = {norm(t.prompt) for t in generate("train", 40, 2026) + generate("train", 2, 2027)}
    train_grams = [grams(s) for s in train]
    test_sets = {
        "development": generate("test", 8, 2026),
        "held-out 1": load_human_tasks(ROOT / "human_eval" / "requests.txt"),
        "held-out 2": load_human_tasks(ROOT / "human_eval" / "requests2.txt"),
    }
    print(f"{len(train)} distinct training requests")
    for name, tasks in test_sets.items():
        requests = {norm(t.prompt) for t in tasks}  # injection variants repeat a request
        best = [max(len(g & tg) / len(g | tg) for tg in train_grams) for g in map(grams, requests)]
        print(f"{name}: {len(requests)} distinct requests, {len(requests & train)} direct matches, "
              f"{sum(b >= NEAR for b in best)} near copies (>= {NEAR}), highest similarity {max(best):.2f}")


if __name__ == "__main__":
    main()
