"""Drawings for the Bag of Visual Words tools (matplotlib Agg -> PIL, no windows).

They use ``Figure`` + ``FigureCanvasAgg`` directly instead of pyplot, so they are
safe to call from a worker thread and never open a window.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
import numpy as np
from PIL import Image

from .bow_model import PATCH_SIDE, BowModel, ImageAnalysis
from .image_lookup import load_image

DISPLAY_MODES = ("RGB", "RGBA", "L", "LA", "P", "1", "CMYK", "YCbCr")
MAX_LINKS = 60  # keypoint -> bar lines drawn in the histogram figure
THUMB_HEIGHT = 140


def figure_to_image(fig: Figure) -> Image.Image:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = Image.frombuffer("RGBA", canvas.get_width_height(), canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    return rgba.convert("RGB")


def word_colors(k: int) -> np.ndarray:
    """One distinct colour per visual word (RGBA rows), shared by every figure."""
    if k <= 20:
        return colormaps["tab20"](np.arange(k))
    return colormaps["turbo"](np.linspace(0.05, 0.95, k))


def stack_vertical(images: list[Image.Image], gap: int = 8) -> Image.Image:
    """Paste images one under the other, centred, on a white background."""
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    top = 0
    for image in images:
        canvas.paste(image.convert("RGB"), ((width - image.width) // 2, top))
        top += image.height + gap
    return canvas


def display_copy(img: Image.Image, gray: Image.Image) -> np.ndarray:
    """Colour version of ``img`` at the size of its working copy ``gray`` (for plotting)."""
    if img.mode in DISPLAY_MODES:
        return np.asarray(img.convert("RGB").resize(gray.size, Image.Resampling.LANCZOS))
    return np.asarray(gray.convert("RGB"))  # high-depth modes: show the normalised copy


# --------------------------------------------------------------------------- 7.1 codebook


def word_tile(exemplars: np.ndarray) -> np.ndarray:
    """2 x 3 montage: the mean patch (top left) and up to five nearest training patches."""
    blank = np.full((PATCH_SIDE, PATCH_SIDE), 200, dtype=np.uint8)
    mean = exemplars.mean(axis=0).astype(np.uint8) if len(exemplars) else blank
    cells = [mean, *list(exemplars[:5])]
    cells += [blank] * (6 - len(cells))
    rows = [np.hstack([np.pad(c, 1, constant_values=255) for c in cells[r * 3 : r * 3 + 3]]) for r in range(2)]
    return np.vstack(rows)


def render_codebook(model: BowModel) -> Image.Image:
    """Every visual word: its patches and the class most of its training occurrences come from."""
    k = model.k
    cols = 6 if k <= 60 else 10
    rows = int(np.ceil(k / cols))
    colors = word_colors(k)
    fig = Figure(figsize=(cols * 1.9, rows * 1.55 + 0.6), dpi=100)
    fig.suptitle(
        f"Visual vocabulary: K = {k} words  (big patch = mean, small = closest training patches)",
        fontsize=11,
    )
    for word in range(rows * cols):
        ax = fig.add_subplot(rows, cols, word + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if word >= k:
            ax.axis("off")
            continue
        exemplar = model.exemplars[word]
        ax.imshow(word_tile(exemplar.patches), cmap="gray", vmin=0, vmax=255)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[word])
            spine.set_linewidth(3)
        if exemplar.class_share.sum() > 0:
            top = int(exemplar.class_share.argmax())
            label = f"vw {word} · {exemplar.class_share[top]:.0%} {model.classes[top][:9]}"
        else:
            label = f"vw {word} · unused"
        ax.set_title(label, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return figure_to_image(fig)


def render_confusion(model: BowModel) -> Image.Image:
    """Confusion matrix of the held-out test images (rows = true class)."""
    matrix = np.asarray(model.metrics["confusion"])
    n = len(model.classes)
    fig = Figure(figsize=(1.6 + 1.1 * n, 1.2 + 1.0 * n), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "black"
        ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=11)
    ax.set_xticks(range(n), model.classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n), model.classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test accuracy {model.metrics['accuracy']:.1%}", fontsize=10)
    fig.tight_layout()
    return figure_to_image(fig)


# --------------------------------------------------------------------------- 7.2 histogram


def render_word_histogram(
    img: Image.Image, analysis: ImageAnalysis, model: BowModel, title: str = "", warn: bool = False
) -> Image.Image:
    """Image with every feature coloured by its visual word, linked to the word's bar.

    ``warn`` draws the title in red (used for training images).
    """
    colors = word_colors(model.k)
    fig = Figure(figsize=(12.5, 5.0), dpi=100)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.5), wspace=0.18)
    ax_img = fig.add_subplot(grid[0])
    ax_bar = fig.add_subplot(grid[1])

    ax_img.imshow(display_copy(img, analysis.gray))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(title or "Features coloured by visual word", fontsize=10, color="#c62828" if warn else "black")
    features = analysis.features
    if len(features):
        ax_img.scatter(
            features.positions[:, 1], features.positions[:, 0], s=14, c=colors[analysis.words],
            edgecolors="black", linewidths=0.4,
        )

    frequency = analysis.word_frequency
    ax_bar.bar(np.arange(model.k), frequency, color=colors, edgecolor="black", linewidth=0.5)
    ax_bar.set_xlim(-0.8, model.k - 0.2)
    ax_bar.set_ylim(0, max(float(frequency.max()) * 1.15, 1e-3))
    ax_bar.set_xlabel("Visual word index")
    ax_bar.set_ylabel("Normalised frequency")
    ax_bar.set_title(f"Histogram of visual words ({len(features)} features)", fontsize=10)
    ax_bar.grid(axis="y", alpha=0.3)

    # a few thin lines from features to their bar, like the slide (all of them would be a blur)
    if len(features):
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(features), min(MAX_LINKS, len(features)), replace=False)
        for index in chosen:
            word = analysis.words[index]
            row, col = features.positions[index]
            fig.add_artist(
                ConnectionPatch(
                    xyA=(col, row), coordsA=ax_img.transData, xyB=(word, frequency[word]), coordsB=ax_bar.transData,
                    color=colors[word], alpha=0.35, linewidth=0.6,
                )
            )
    return figure_to_image(fig)


def render_neighbours(analysis: ImageAnalysis) -> Image.Image | None:
    """Thumbnails of the training images whose histogram is closest (chi-squared)."""
    thumbs = []
    for path, label, distance in analysis.neighbours:
        try:
            image = load_image(Path(path)).convert("RGB")
        except OSError:
            continue
        scale = THUMB_HEIGHT / image.height
        thumbs.append((image.resize((max(1, round(image.width * scale)), THUMB_HEIGHT)), label, distance, Path(path).name))
    if not thumbs:
        return None
    fig = Figure(figsize=(2.3 * len(thumbs), 2.4), dpi=100)
    fig.suptitle("Most similar training images (chi-squared distance between histograms)", fontsize=10)
    for index, (thumb, label, distance, name) in enumerate(thumbs):
        ax = fig.add_subplot(1, len(thumbs), index + 1)
        ax.imshow(np.asarray(thumb))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}\n{name} · d={distance:.3f}", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return figure_to_image(fig)


# --------------------------------------------------------------------------- 7.3 classification


def render_classification(
    img: Image.Image, analysis: ImageAnalysis, ranked: list[tuple[str, float]], warning: str | None = None
) -> Image.Image:
    """The image under ``p(class) = 0.64`` (as in the slide) with every probability below."""
    best, probability = ranked[0]
    fig = Figure(figsize=(6.5, 6.0), dpi=100)
    grid = fig.add_gridspec(2, 1, height_ratios=(4.0, 1.0), hspace=0.3)
    ax_img = fig.add_subplot(grid[0])
    ax_img.imshow(display_copy(img, analysis.gray))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"p({best}) = {probability:.2f}", fontsize=16)
    if warning:
        fig.text(0.5, 0.97, "TRAINING IMAGE: optimistic result", ha="center", va="top", color="#c62828", fontsize=11)
    ax_p = fig.add_subplot(grid[1])
    names = [name for name, _ in ranked][::-1]
    values = [p for _, p in ranked][::-1]
    ax_p.barh(names, values, color=["#999999"] * (len(names) - 1) + ["#2a7ab0"])
    ax_p.set_xlim(0, 1)
    ax_p.tick_params(labelsize=8)
    for y, value in enumerate(values):
        ax_p.text(min(value + 0.01, 0.9), y, f"{value:.2f}", va="center", fontsize=8)
    return figure_to_image(fig)
