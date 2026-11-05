#!/usr/bin/env python3
"""Render three benchmark designs at 800 px, with 2x PNG companions."""

import json
from functools import cache
from io import BytesIO
import os
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path
import tempfile
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "oya-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
WIDTH = 800
MARGIN = 24
FONT = 8.5  # 11.8 px at the intended 800 px display width.
C_MODELS = ("oya", "gpt-6-astra", "qwen3-4b", "tiny-aya", "llama-3.2-3b")
C_LOGOS = {
    "oya": "icon/oya-icon.svg",
    "gpt-6-astra": "logos/openai.svg",
    "qwen3-4b": "logos/qwen.svg",
    "tiny-aya": "logos/aya.svg",
    "llama-3.2-3b": "logos/meta.svg",
}
THEMES = {
    "light": dict(accent="#2a78d6", gray="#898781", surface="#fcfcfb",
                  ink="#0b0b0b", secondary="#52514e", grid="#e1e0d9", baseline="#c3c2b7"),
    "dark": dict(accent="#3987e5", gray="#898781", surface="#1a1a19",
                 ink="#f0efec", secondary="#c3c2b7", grid="#2c2c2a", baseline="#383835"),
}
SUBTITLE = (
    "Same prompt, same tasks, same scorer. Open models run in 4-bit on a 4 GB "
    "laptop GPU; OpenAI models via the API."
)
FOOTNOTE = (
    "Task success counts polite requests without a question mark as asking "
    "({polite} such replies, all checked by hand). Oya at full precision: 100% (Kaggle). "
    "API cost from third-party list prices. {n} tasks is a small sample."
)


def is_open(row):
    return row["runs_on"] == "laptop GPU, 4-bit"


def label(row):
    return "Oya (ours)" if row["name"] == "oya" else row["label"]


def percent(value):
    return f"{(Decimal(str(value)) * 100).to_integral_value(rounding=ROUND_HALF_UP)}%"


def runs_where(row):
    return f"{row['size']} · 4-bit laptop GPU" if is_open(row) else "OpenAI API"


def cost(row):
    amount = Decimal(str(row["usd_per_1000_requests"]))
    if is_open(row) and amount == 0:
        return "$0, self-hosted"
    value = amount.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    return f"${value}, self-hosted" if is_open(row) else f"~${value}"


def text(fig, theme, x, y, value, size=FONT, muted=False, bold=False, ha="left"):
    height = fig.get_figheight() * 100
    return fig.text(x / WIDTH, 1 - y / height, value, fontsize=size,
                    color=theme["secondary" if muted else "ink"],
                    weight="bold" if bold else "normal", ha=ha, va="top",
                    linespacing=1.35)


def wrap(fig, value, width, size=FONT, bold=False):
    renderer = fig.canvas.get_renderer()
    font = FontProperties(size=size, weight="bold" if bold else "normal")
    lines = []
    for paragraph in value.splitlines():
        line = ""
        for word in paragraph.split():
            # Future model identifiers may be longer than an entire column.
            while renderer.get_text_width_height_descent(word, font, False)[0] > width:
                if line:
                    lines.append(line)
                    line = ""
                cut = len(word) - 1
                while renderer.get_text_width_height_descent(word[:cut], font, False)[0] > width:
                    cut -= 1
                hyphen = word.rfind("-", 0, cut + 1)
                if hyphen > 0:
                    cut = hyphen + 1
                lines.append(word[:cut])
                word = word[cut:]
            candidate = f"{line} {word}".strip()
            if line and renderer.get_text_width_height_descent(candidate, font, False)[0] > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return "\n".join(lines)


def rule(fig, theme, y, x=MARGIN, end=WIDTH - MARGIN):
    height = fig.get_figheight() * 100
    fig.add_artist(Line2D([x / WIDTH, end / WIDTH], [1 - y / height] * 2,
                          transform=fig.transFigure, color=theme["grid"], linewidth=0.7))


def canvas(theme, height, n):
    fig = plt.figure(figsize=(WIDTH / 100, height / 100), dpi=100, facecolor=theme["surface"])
    text(fig, theme, MARGIN, 22,
         f"Oya vs other models on {n} held-out Yoruba and Pidgin requests", size=15, bold=True)
    text(fig, theme, MARGIN, 57, SUBTITLE, muted=True)
    return fig


def footer(fig, theme, top, n):
    rule(fig, theme, top)
    note = wrap(fig, FOOTNOTE.format(n=n, polite=POLITE), WIDTH - 2 * MARGIN)
    text(fig, theme, MARGIN, top + 14, note, muted=True)


def axes(fig, theme, left, top, width, height, horizontal):
    total_height = fig.get_figheight() * 100
    ax = fig.add_axes([left / WIDTH, 1 - (top + height) / total_height,
                       width / WIDTH, height / total_height], facecolor=theme["surface"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    axis = ax.xaxis if horizontal else ax.yaxis
    axis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.set_ticks([0, 50, 100] if horizontal else [0, 25, 50, 75, 100])
    ax.tick_params(axis="both", length=0, labelsize=FONT, colors=theme["secondary"], pad=7)
    ax.set_axisbelow(True)
    if horizontal:
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.axvline(0, color=theme["baseline"], linewidth=1, clip_on=False, zorder=3)
    else:
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.grid(axis="y", color=theme["grid"], linewidth=0.6)
        ax.axhline(0, color=theme["baseline"], linewidth=1, clip_on=False, zorder=3)
    return ax


def design_a(rows, theme, n):
    opens = sorted((r for r in rows if is_open(r)),
                   key=lambda r: (r["name"] != "oya", -r["reviewed_success"], r["label"]))
    apis = sorted((r for r in rows if not is_open(r)),
                  key=lambda r: (-r["reviewed_success"], r["label"]))
    ordered = opens + apis
    # Extra room is reserved for wrapped names, without changing font size.
    probe = plt.figure(figsize=(8, 2), dpi=100)
    names = [wrap(probe, label(r), 175, bold=r["name"] == "oya") for r in ordered]
    plt.close(probe)
    row_height = max(52, 20 + max(len(s.splitlines()) for s in names) * 17 + 16)
    gap = 22 if opens and apis else 0
    top = 168
    plot_height = len(ordered) * row_height + gap
    fig = canvas(theme, top + plot_height + 142, n)
    text(fig, theme, 24, 108, "Model", bold=True)
    text(fig, theme, 222, 108, "Task success", bold=True)
    text(fig, theme, 447, 108, "Tasks with a wrong payment\n(lower is better)", bold=True)
    text(fig, theme, 656, 108, "Cost per 1,000\nrequests", bold=True)
    panels = [axes(fig, theme, left, top, width, plot_height, True)
              for left, width in [(222, 174), (447, 155)]]
    for ax in panels:
        ax.set_ylim(plot_height, 0)
    for i, (row, name) in enumerate(zip(ordered, names)):
        y = (i + 0.5) * row_height + (gap if i >= len(opens) else 0)
        ours = row["name"] == "oya"
        text(fig, theme, 24, top + y - 18, name, bold=ours)
        text(fig, theme, 24, top + y - 18 + 17 * len(name.splitlines()),
             runs_where(row),
             muted=True)
        text(fig, theme, 656, top + y - 7, cost(row), bold=ours)
        for ax, key in zip(panels, ("reviewed_success", "wrong_payment")):
            value = row[key] * 100
            ax.barh(y, value, height=16, color=theme["accent" if ours else "gray"], linewidth=0)
            ax.annotate(percent(row[key]), (value, y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=FONT, color=theme["ink"],
                        weight="bold" if ours else "normal", clip_on=False)
    if gap:
        rule(fig, theme, top + len(opens) * row_height + gap / 2)
    footer(fig, theme, top + plot_height + 44, n)
    return fig


def design_b(rows, theme, n):
    ordered = sorted(rows, key=lambda r: (-r["reviewed_success"], r["name"] != "oya", r["label"]))
    left, width, top, plot_height = 170, 606, 154, 258
    cell = width / len(ordered)
    probe = plt.figure(figsize=(8, 2), dpi=100)
    names = [wrap(probe, label(r), cell - 8, bold=r["name"] == "oya") for r in ordered]
    details = [wrap(probe, runs_where(r), cell - 8) for r in ordered]
    costs = [wrap(probe, cost(r), cell - 8) for r in ordered]
    plt.close(probe)
    names_top = top + plot_height + 17
    details_top = names_top + 17 * max(len(s.splitlines()) for s in names) + 6
    strip_top = details_top + 17 * max(len(s.splitlines()) for s in details) + 20
    cost_top = strip_top + 49
    foot_top = cost_top + 17 * max(len(s.splitlines()) for s in costs) + 22
    fig = canvas(theme, foot_top + 98, n)
    text(fig, theme, 24, 108, "Task success", size=11, bold=True)
    ax = axes(fig, theme, left, top, width, plot_height, False)
    ax.set_xlim(-0.5, len(ordered) - 0.5)
    for i, (row, name, detail, price) in enumerate(zip(ordered, names, details, costs)):
        ours = row["name"] == "oya"
        value = row["reviewed_success"] * 100
        ax.bar(i, value, width=0.36, color=theme["accent" if ours else "gray"], linewidth=0)
        ax.annotate(percent(row["reviewed_success"]), (i, value), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10,
                    color=theme["ink"], weight="bold" if ours else "normal", clip_on=False)
        x = left + (i + 0.5) * cell
        text(fig, theme, x, names_top, name, bold=ours, ha="center")
        text(fig, theme, x, details_top, detail, muted=True, ha="center")
        text(fig, theme, x, strip_top + 14, percent(row["wrong_payment"]), ha="center", bold=ours)
        text(fig, theme, x, cost_top, price, ha="center")
    rule(fig, theme, strip_top)
    text(fig, theme, 24, strip_top + 14, "Wrong payments", muted=True)
    text(fig, theme, 24, cost_top, "Cost / 1,000 requests", muted=True)
    footer(fig, theme, foot_top, n)
    return fig


@cache
def c_logo(name):
    svg = ET.fromstring((ROOT / "docs/img" / C_LOGOS[name]).read_text())
    svg.set("color", "#ffffff")
    svg.set("fill", "#ffffff")
    for element in svg.iter():
        if "fill" in element.attrib and element.get("fill") != "none":
            element.set("fill", "#ffffff")
    png = subprocess.run(
        ["rsvg-convert", "--width=256", "--height=256"],
        input=ET.tostring(svg), capture_output=True, check=True,
    ).stdout
    return plt.imread(BytesIO(png), format="png")


def design_c(rows, theme, n):
    by_name = {row["name"]: row for row in rows}
    ordered = [by_name[name] for name in C_MODELS]
    groups = [(f"All {n} tasks", "reviewed_success"), ("Pidgin", "pidgin"),
              ("Yoruba", "yoruba"), ("Safety tasks", "safety")]
    left, width, top, plot_height = 82, 694, 154, 300
    cell = width / (len(groups) * len(ordered) + 0.5 * (len(groups) - 1))
    bar_width = cell * 0.75
    group_gap = cell * 0.5  # Together with slot padding, one bar between groups.
    group_width = cell * len(ordered)
    label_size = FONT

    # Measure rotated names so future model names cannot run into the footer.
    probe = plt.figure(figsize=(8, 2), dpi=100)
    renderer = probe.canvas.get_renderer()
    name_height = max(probe.text(0, 0, label(row), fontsize=label_size,
                                weight="bold" if row["name"] == "oya" else "normal",
                                rotation=45, rotation_mode="anchor", ha="right", va="top")
                      .get_window_extent(renderer).height for row in ordered)
    title = wrap(probe, f"Task success on {n} held-out Yoruba and Pidgin requests",
                 WIDTH - 2 * MARGIN, size=15, bold=True)
    also_tested = ", ".join(
        f"{by_name[name]['label']} {percent(by_name[name]['reviewed_success'])}"
        for name in ("gpt-6.1-sol", "gpt-6-luna", "phi-4-mini")
    )
    extra_note = wrap(probe, f"Also tested: {also_tested} (full table in the write-up). "
                      "Logos: LobeHub icons (MIT); trademarks of their owners.",
                      WIDTH - 2 * MARGIN)
    note = wrap(probe, FOOTNOTE.format(n=n, polite=POLITE), WIDTH - 2 * MARGIN)
    plt.close(probe)
    top = 112 + 28 * (len(title.splitlines()) - 1)
    names_top = top + plot_height + 16
    groups_top = names_top + name_height + 20
    foot_top = groups_top + 40
    extra_top = foot_top + 14 + 17 * len(note.splitlines()) + 10
    height = extra_top + 17 * len(extra_note.splitlines()) + 24
    fig = plt.figure(figsize=(WIDTH / 100, height / 100), dpi=100,
                     facecolor=theme["surface"])
    text(fig, theme, WIDTH / 2, 22, title, size=15, bold=True, ha="center")
    text(fig, theme, WIDTH / 2, 57 + (len(title.splitlines()) - 1) * 28,
         SUBTITLE, muted=True, ha="center")
    ax = axes(fig, theme, left, top, width, plot_height, False)
    ax.set_xlim(0, width)
    for group_index, (group_name, key) in enumerate(groups):
        start = group_index * (group_width + group_gap)
        for i, row in enumerate(ordered):
            ours = row["name"] == "oya"
            x = start + (i + 0.5) * cell
            value = row[key] * 100
            lo, hi = x - bar_width / 2, x + bar_width / 2
            # x units are display pixels; y units are percentages. Only the
            # top corners curve, with a 4 px radius and a flat zero baseline.
            rx = min(4, (hi - lo) / 2)
            ry = min(4 * 100 / plot_height, value)
            path = PlotPath([(lo, 0), (hi, 0), (hi, value - ry),
                             (hi, value), (hi - rx, value), (lo + rx, value),
                             (lo, value), (lo, value - ry), (lo, 0), (lo, 0)],
                            [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO,
                             PlotPath.CURVE3, PlotPath.CURVE3, PlotPath.LINETO,
                             PlotPath.CURVE3, PlotPath.CURVE3, PlotPath.LINETO,
                             PlotPath.CLOSEPOLY])
            ax.add_patch(PathPatch(path, facecolor=theme["accent" if ours else "gray"],
                                   edgecolor="none", zorder=2))
            logo_size = bar_width * 0.55
            padding = 5
            bar_height = value * plot_height / 100
            inside = bar_height >= logo_size + 2 * padding
            logo_y = (-padding - logo_size / 2 if inside else padding + logo_size / 2)
            # OffsetImage uses points, so convert our intended 100-dpi pixel size.
            logo = OffsetImage(c_logo(row["name"]), zoom=logo_size * 72 / 100 / 256)
            ax.add_artist(AnnotationBbox(
                logo, (x, value), xybox=(0, logo_y * 72 / 100),
                boxcoords="offset points", frameon=False, pad=0,
                annotation_clip=False, zorder=4,
            ))
            value_offset = padding if inside else logo_size + 2 * padding
            ax.annotate(percent(row[key]), (x, value), xytext=(0, value_offset * 72 / 100),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=label_size, color=theme["ink"],
                        weight="bold" if ours else "normal", clip_on=False)
            name = text(fig, theme, left + x, names_top, label(row), size=label_size,
                        bold=ours, muted=not ours, ha="right")
            name.set_rotation(45)
            name.set_rotation_mode("anchor")
        text(fig, theme, left + start + group_width / 2, groups_top,
             group_name, muted=True, ha="center")
    footer(fig, theme, foot_top, n)
    text(fig, theme, MARGIN, extra_top, extra_note, muted=True)
    return fig


POLITE = 0  # replies counted as asking without a question mark, summed over models (set in main)


def main():
    global POLITE
    rows = json.loads((ROOT / "runs/compare.json").read_text())
    POLITE = sum(r["polite_asks"] for r in rows)
    if not rows or len({r["n"] for r in rows}) != 1:
        raise ValueError("Expected a nonempty comparison with the same task count for every model")
    for row in rows:
        for key in ("reviewed_success", "wrong_payment", "pidgin", "yoruba", "safety"):
            if not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid {key} for {row['name']}")
        if row["runs_on"] not in ("OpenAI API", "laptop GPU, 4-bit"):
            raise ValueError(f"Unknown run environment for {row['name']}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "text.parse_math": False, "axes.unicode_minus": False})
    destination = ROOT / "docs/img"
    destination.mkdir(parents=True, exist_ok=True)
    for design, render in (("a", design_a), ("b", design_b), ("c", design_c)):
        for mode, theme in THEMES.items():
            fig = render(rows, theme, rows[0]["n"])
            stem = destination / f"compare-{design}-{mode}"
            fig.savefig(stem.with_suffix(".svg"), facecolor=theme["surface"])
            fig.savefig(stem.with_suffix(".png"), dpi=200, facecolor=theme["surface"])
            plt.close(fig)
            print(f"Rendered {stem.relative_to(ROOT)}.svg and .png")


if __name__ == "__main__":
    main()
