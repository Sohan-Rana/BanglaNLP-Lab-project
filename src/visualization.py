"""Charts, word cloud and banner for BanglaNLP Lab.

Design rules (kept consistent across every figure)
--------------------------------------------------
* One validated colour-blind-safe palette: blue / orange / aqua for the three sentiment
  classes (same colour = same class in every chart), one-hue blue ramp for magnitude.
* Thin marks, recessive grid, direct value labels, text in ink colours (never series colours).
* Bangla text is shaped correctly by matplotlib's text engine; we ship Noto Sans Bengali
  and fall back to DejaVu Sans for Latin glyphs (Noto Sans Bengali has none).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

from .config import (
    AXIS,
    BLUE_RAMP,
    CLASS_COLORS,
    FONT_BOLD,
    FONT_REGULAR,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    LABELS,
    LABELS_BN,
    SURFACE,
)
from .fonts import ensure_fonts
from .preprocessing import tokenize

_FONT_FAMILY = ["Noto Sans Bengali", "DejaVu Sans"]
_fonts_ready = False
FOOTNOTE = "BanglaNLP Lab"
DEMO_NOTE = "BanglaNLP Lab · demo data (108 hand-written sentences) - not a real-world performance estimate"


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def setup_fonts() -> None:
    """Register the bundled Bangla font with matplotlib and set global style (idempotent)."""
    global _fonts_ready
    if _fonts_ready:
        return
    ensure_fonts()
    for font_file in (FONT_REGULAR, FONT_BOLD):
        if Path(font_file).exists():
            fm.fontManager.addfont(str(font_file))
    matplotlib.rcParams.update(
        {
            "font.family": _FONT_FAMILY,
            "axes.unicode_minus": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
        }
    )
    _fonts_ready = True


def _header(fig: Figure, title: str, subtitle: str | None = None) -> float:
    """Left-aligned title + subtitle; returns the top of the plotting area (figure fraction)."""
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.22 / h, title, fontsize=15, fontweight="bold", color=INK, ha="left", va="top")
    top_in = 0.62
    if subtitle:
        fig.text(0.02, 1 - 0.58 / h, subtitle, fontsize=10, color=INK_SECONDARY, ha="left", va="top")
        top_in = 0.98
    return 1 - top_in / h


def _footnote(fig: Figure, text: str = FOOTNOTE) -> None:
    fig.text(0.02, 0.012, text, fontsize=8, color=INK_MUTED, ha="left", va="bottom")


def _style_axes(ax, grid: str | None = "x", hide_left: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=10, pad=6)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _finish(fig: Figure, save_path: str | Path | None, dpi: int = 200) -> Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=SURFACE)
    return fig


def _class_tick_label(label: str) -> str:
    return f"{LABELS_BN.get(label, label)}\n{label}"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_word_frequency(
    freq_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Most frequent words",
    subtitle: str | None = "Stopwords removed · words shorter than 2 characters ignored",
    color: str = CLASS_COLORS["positive"],
    save_path: str | Path | None = None,
) -> Figure:
    """Horizontal bar chart of the most frequent words (highest on top)."""
    setup_fonts()
    data = freq_df.head(top_n).iloc[::-1].reset_index(drop=True)
    if data.empty:
        raise ValueError("No words to plot.")

    height = max(3.2, 0.36 * len(data) + 1.7)
    fig, ax = plt.subplots(figsize=(8, height), dpi=120)
    top = _header(fig, title, subtitle)

    y = np.arange(len(data))
    ax.barh(y, data["count"], height=0.55, color=color, zorder=3)
    ax.set_yticks(y, labels=data["word"], fontsize=12, color=INK)
    ax.set_xlim(0, data["count"].max() * 1.12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for yi, count in zip(y, data["count"]):
        ax.text(count + data["count"].max() * 0.012, yi, f"{count}", va="center", fontsize=10, color=INK_SECONDARY)
    ax.set_xlabel("Occurrences")
    _style_axes(ax, grid="x")
    _footnote(fig)
    fig.tight_layout(rect=(0, 0.03, 1, top))
    return _finish(fig, save_path)


def make_wordcloud(
    frequencies: Mapping[str, float] | pd.DataFrame,
    *,
    width: int = 1200,
    height: int = 600,
    max_words: int = 80,
    save_path: str | Path | None = None,
) -> Image.Image:
    """Create a word cloud (PIL image) from word frequencies.

    We always pass *frequencies* to WordCloud (never raw text), because WordCloud's own
    tokenizer would split Bangla words at vowel signs.
    """
    ensure_fonts()
    if isinstance(frequencies, pd.DataFrame):
        freq = dict(zip(frequencies["word"], frequencies["count"]))
    else:
        freq = dict(frequencies)
    if not freq:
        raise ValueError("No words to build a word cloud from.")

    top = max(freq.values())
    ramp = BLUE_RAMP[3:]  # step 250 .. 700: light enough to stay visible, dark enough to read

    def color_func(word, font_size=None, position=None, orientation=None, random_state=None, **kwargs):
        ratio = freq.get(word, 0) / top
        return ramp[min(len(ramp) - 1, int(ratio * (len(ramp) - 1) + 0.5))]

    cloud = WordCloud(
        font_path=str(FONT_REGULAR) if Path(FONT_REGULAR).exists() else None,
        width=width,
        height=height,
        background_color=SURFACE,
        max_words=max_words,
        prefer_horizontal=1.0,  # rotated Bangla text is hard to read
        min_font_size=14,
        relative_scaling=0.6,
        margin=8,
        random_state=42,
        color_func=color_func,
    ).generate_from_frequencies(freq)

    image = cloud.to_image()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)
    return image


def plot_sentiment_distribution(
    counts: Mapping[str, int],
    title: str = "Sentiment distribution",
    subtitle: str | None = "Number of texts per class",
    footnote: str = FOOTNOTE,
    save_path: str | Path | None = None,
) -> Figure:
    """Bar chart with one bar per sentiment class (fixed class colours)."""
    setup_fonts()
    labels = [lab for lab in LABELS if lab in counts] + [lab for lab in counts if lab not in LABELS]
    values = [int(counts[lab]) for lab in labels]

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=120)
    top = _header(fig, title, subtitle)
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.5, color=[CLASS_COLORS.get(l, INK_MUTED) for l in labels], zorder=3)
    ax.set_xticks(x, labels=[_class_tick_label(l) for l in labels], fontsize=11, color=INK)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for xi, v in zip(x, values):
        ax.text(xi, v + max(values) * 0.02, str(v), ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_ylabel("Texts")
    _style_axes(ax, grid="y")
    _footnote(fig, footnote)
    fig.tight_layout(rect=(0, 0.03, 1, top))
    return _finish(fig, save_path)


def plot_length_distribution(
    texts: Sequence[str],
    labels: Sequence[str],
    title: str = "Text length by sentiment class",
    subtitle: str | None = "Words per text - the dashed line marks the median",
    footnote: str = FOOTNOTE,
    save_path: str | Path | None = None,
) -> Figure:
    """Small multiples: histogram of words per text for each class (shared axes)."""
    setup_fonts()
    df = pd.DataFrame({"label": list(labels), "words": [len(tokenize(t)) for t in texts]})
    classes = [lab for lab in LABELS if lab in set(df["label"])]
    max_words = int(df["words"].max())
    bins = np.arange(0, max_words + 3, 2)

    fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes) + 0.6, 4.2), dpi=120, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    top = _header(fig, title, subtitle)

    ymax = max(np.histogram(df.loc[df["label"] == c, "words"], bins=bins)[0].max() for c in classes)
    for ax, cls in zip(axes, classes):
        values = df.loc[df["label"] == cls, "words"]
        counts, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=1.7, color=CLASS_COLORS[cls], zorder=3)
        median = float(values.median())
        ax.axvline(median, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)
        ax.text(median + 0.4, ymax * 1.17, f"median {median:g}", fontsize=9, color=INK_SECONDARY, va="top")
        ax.set_title(_class_tick_label(cls).replace("\n", " · "), fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlabel("Words per text")
        ax.set_ylim(0, ymax * 1.22)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _style_axes(ax, grid="y")
    axes[0].set_ylabel("Texts")
    _footnote(fig, footnote)
    fig.tight_layout(rect=(0, 0.03, 1, top))
    return _finish(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    title: str = "Confusion matrix",
    subtitle: str | None = None,
    footnote: str = FOOTNOTE,
    save_path: str | Path | None = None,
) -> Figure:
    """Annotated confusion matrix (rows = actual, columns = predicted) in a one-hue blue ramp."""
    setup_fonts()
    cm = np.asarray(cm)
    n = len(labels)
    cmap = LinearSegmentedColormap.from_list("bn_blue", BLUE_RAMP[:9])

    fig, ax = plt.subplots(figsize=(5.8, 5.6), dpi=120)
    top = _header(fig, title, subtitle)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
    ticks = [_class_tick_label(l) for l in labels]
    ax.set_xticks(range(n), labels=ticks, fontsize=10, color=INK)
    ax.set_yticks(range(n), labels=ticks, fontsize=10, color=INK)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.xaxis.set_label_position("bottom")
    for i in range(n):
        for j in range(n):
            value = int(cm[i, j])
            dark_cell = cm[i, j] > cm.max() * 0.55
            ax.text(j, i, str(value), ha="center", va="center", fontsize=16, fontweight="bold",
                    color="#ffffff" if dark_cell else INK)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    _footnote(fig, footnote)
    fig.tight_layout(rect=(0, 0.03, 1, top))
    return _finish(fig, save_path)


def plot_model_comparison(
    summary: Mapping,
    title: str = "Macro-F1: lexicon baseline vs. trained model",
    subtitle: str | None = None,
    footnote: str = DEMO_NOTE,
    save_path: str | Path | None = None,
) -> Figure:
    """Horizontal bars comparing the lexicon baseline with the ML model (macro-F1).

    ``summary`` is ``EvaluationResult.summary()``. The cross-validation bar shows ±1 std.
    """
    setup_fonts()
    model, base = summary["model"], summary["lexicon_baseline"]
    rows = [
        ("Lexicon baseline\n(hold-out test)", base["macro_f1"], None),
        ("TF-IDF + Logistic Regression\n(hold-out test)", model["macro_f1"], None),
    ]
    if model.get("cv_macro_f1_mean") is not None:
        rows.append(("TF-IDF + Logistic Regression\n(5-fold cross-validation)", model["cv_macro_f1_mean"], model["cv_macro_f1_std"]))
    rows = rows[::-1]

    if subtitle is None:
        subtitle = f"Test set = {summary['n_test']} texts out of {summary['n_total']} - tiny samples, so scores are very noisy"
    fig, ax = plt.subplots(figsize=(8, 4.3), dpi=120)
    top = _header(fig, title, subtitle)
    y = np.arange(len(rows))
    values = [r[1] for r in rows]
    ax.barh(y, values, height=0.45, color=CLASS_COLORS["positive"], zorder=3)
    for yi, (_, v, err) in zip(y, rows):
        if err:
            ax.errorbar(v, yi, xerr=err, color=INK_SECONDARY, capsize=4, linewidth=1.2, zorder=4)
        ax.text(v + (err or 0) + 0.015, yi, f"{v:.2f}", va="center", fontsize=11, color=INK)
    ax.set_yticks(y, labels=[r[0] for r in rows], fontsize=10, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro-F1 (0 - 1, higher is better)")
    _style_axes(ax, grid="x")
    _footnote(fig, footnote)
    fig.tight_layout(rect=(0, 0.03, 1, top))
    return _finish(fig, save_path)


def plot_top_features(
    top_features: Mapping[str, Sequence[tuple[str, float]]],
    n: int = 8,
    title: str = "Which words push the model towards each class?",
    subtitle: str | None = "Largest logistic-regression weights (word features only)",
    footnote: str = DEMO_NOTE,
    save_path: str | Path | None = None,
) -> Figure:
    """One panel per class listing its most influential words - a simple transparency check."""
    setup_fonts()
    classes = [c for c in LABELS if c in top_features]
    fig, axes = plt.subplots(1, len(classes), figsize=(3.6 * len(classes) + 0.4, 0.34 * n + 2.6), dpi=120)
    axes = np.atleast_1d(axes)
    top = _header(fig, title, subtitle)
    xmax = max(w for c in classes for _, w in top_features[c][:n])
    for ax, cls in zip(axes, classes):
        items = list(top_features[cls][:n])[::-1]
        y = np.arange(len(items))
        ax.barh(y, [w for _, w in items], height=0.55, color=CLASS_COLORS[cls], zorder=3)
        ax.set_yticks(y, labels=[f for f, _ in items], fontsize=11, color=INK)
        ax.set_xlim(0, xmax * 1.1)
        ax.set_title(_class_tick_label(cls).replace("\n", " · "), fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlabel("Weight")
        _style_axes(ax, grid="x")
    _footnote(fig, footnote)
    fig.tight_layout(rect=(0, 0.03, 1, top))
    return _finish(fig, save_path)


def figure_to_png_bytes(fig: Figure, dpi: int = 200) -> bytes:
    """Return a figure as PNG bytes (handy for download buttons)."""
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=SURFACE)
    return buffer.getvalue()


def image_to_png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# README banner
# --------------------------------------------------------------------------- #
def make_banner(save_path: str | Path, width: int = 1600, height: int = 420) -> Image.Image:
    """Draw the README banner with Pillow (Bangla shaping works through libraqm)."""
    ensure_fonts()
    latin_bold = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    latin = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"

    image = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(image)

    draw.text((80, 70), "BanglaNLP Lab", font=ImageFont.truetype(str(latin_bold), 92), fill=INK)
    draw.rounded_rectangle((84, 192, 244, 200), radius=4, fill=CLASS_COLORS["positive"])
    draw.text((80, 222), "বাংলা ভাষার জন্য এনএলপি টুলকিট", font=ImageFont.truetype(str(FONT_BOLD), 50), fill=INK_SECONDARY)
    tagline_font = ImageFont.truetype(str(latin), 24)
    draw.text((84, 312), "Text cleaning  ·  Word frequency  ·  Sentiment analysis", font=tagline_font, fill=INK_MUTED)
    draw.text((84, 350), "Word cloud  ·  Streamlit dashboard", font=tagline_font, fill=INK_MUTED)

    # decorative mini word-cloud on the right; rows are laid out by measuring each word
    # so nothing can overlap (words, font size, ramp step) - fixed order -> reproducible
    rows = [
        (50, [("বাংলা", 96, 11), ("ভাষা", 62, 9)]),
        (190, [("শব্দ", 58, 8), ("অনুভূতি", 50, 6)]),
        (280, [("তথ্য", 44, 5), ("গবেষণা", 44, 4), ("প্রযুক্তি", 40, 3)]),
    ]
    gap = 34
    for y, words in rows:
        x = 1120
        for word, size, ramp_idx in words:
            font = ImageFont.truetype(str(FONT_BOLD), size)
            draw.text((x, y), word, font=font, fill=BLUE_RAMP[ramp_idx])
            x += int(font.getlength(word)) + gap

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    return image
