"""Set 10: figures of the CLIP tools (matplotlib, no Tk, no PyTorch)."""

from __future__ import annotations

from collections.abc import Sequence
import textwrap

from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from .bow_render import figure_to_image
from .clip_model import Gallery, ZeroShotResult, clip_view

BAR_COLOR = "#3b6ea8"  # the notebook's colour
BEST_COLOR = "#2e9e4f"
COS_COLOR = "#8a8a8a"
MAX_BARS = 25


def _short(text: str, width: int = 34) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    shortened = textwrap.shorten(text, width, placeholder="…")
    # shorten() drops whole words, so one long word would become just "…": cut it instead
    return shortened if len(shortened) > width // 2 else text[:width - 1] + "…"


def render_zero_shot(image: Image.Image, results: Sequence[ZeroShotResult], show_cosine: bool = True,
                     title: str = "") -> Image.Image:
    """The image (and what CLIP sees) next to one probability chart per result, as in the slides."""
    rows = len(results)
    n_labels = min(len(results[0].labels), MAX_BARS)
    row_height = max(2.4, 0.32 * n_labels + 1.2)
    cols = 3 if show_cosine else 2
    single = rows == 1
    fig = Figure(figsize=(5.5 + 5 * (cols - 1), max(3.4 if single else 5.2, row_height * rows + 0.6)), dpi=100)
    grid = fig.add_gridspec(1 if single else rows, cols, width_ratios=[1.3 if single else 1.1] + [1.4] * (cols - 1),
                            hspace=0.65, wspace=0.55, top=0.86 if single else 0.92, bottom=0.14 if single else 0.06)
    if single:
        image_cells = grid[0, 0].subgridspec(1, 2, wspace=0.08)
        first, second = image_cells[0, 0], image_cells[0, 1]
    else:
        image_cells = grid[:, 0].subgridspec(2, 1, hspace=0.25)
        first, second = image_cells[0, 0], image_cells[1, 0]
    ax = fig.add_subplot(first)
    ax.imshow(image.convert("RGB"))
    ax.set_title(f"Image ({image.width}×{image.height})", fontsize=9)
    ax.axis("off")
    ax = fig.add_subplot(second)
    ax.imshow(clip_view(image))
    ax.set_title("What CLIP sees (224×224)", fontsize=9)
    ax.axis("off")

    for r, result in enumerate(results):
        probs = result.probs
        order = np.argsort(probs)[-n_labels:]  # largest at the top
        labels = [_short(result.labels[i]) for i in order]
        colors = [BEST_COLOR if i == result.best else BAR_COLOR for i in order]
        ax = fig.add_subplot(grid[r, 1])
        ax.barh(labels, probs[order], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability", fontsize=8)
        model = result.model.split("/")[-1]
        ax.set_title(f"{result.title}  [{model}]", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        for y, value in enumerate(probs[order]):
            ax.text(min(value + 0.02, 0.86), y, f"{value:.2f}", va="center", fontsize=8)
        if show_cosine:
            ax = fig.add_subplot(grid[r, 2])
            cos = result.cosine[order]
            ax.barh(labels, cos, color=[BEST_COLOR if i == result.best else COS_COLOR for i in order])
            low = min(0.0, float(cos.min()) - 0.02)
            ax.set_xlim(low, max(0.4, float(cos.max()) + 0.05))
            ax.set_xlabel(f"cosine similarity (logit = {result.logit_scale:.0f} × cos)", fontsize=8)
            ax.set_title("raw similarity (does not sum to 1)", fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
            for y, value in enumerate(cos):
                ax.text(value + 0.005, y, f"{value:.3f}", va="center", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=12)
    return figure_to_image(fig)


def render_search(gallery: Gallery, queries: Sequence[str], similarities: np.ndarray, top_k: int,
                  show_matrix: bool = False, model: str = "") -> Image.Image:
    """For each query, the ``top_k`` most similar images; optionally the whole similarity matrix."""
    top_k = min(top_k, len(gallery.names))
    rows = len(queries)
    matrix_rows = 1 if show_matrix else 0
    fig = Figure(figsize=(max(10, 2.6 * top_k), 2.9 * rows + (4.8 if show_matrix else 0)), dpi=100)
    # with the matrix, leave room on the left for the queries written as its row labels
    grid = fig.add_gridspec(rows + matrix_rows, top_k, height_ratios=[1] * rows + ([1.7] * matrix_rows),
                            hspace=0.55, wspace=0.08, left=0.25 if show_matrix else 0.05, right=0.9)
    for q, query in enumerate(queries):
        sims = similarities[q]
        best = np.argsort(-sims)[:top_k]
        for k, i in enumerate(best):
            ax = fig.add_subplot(grid[q, k])
            ax.imshow(gallery.thumbnails[i])
            ax.set_title(f"{_short(gallery.names[i], 24)}\nsim = {sims[i]:.3f}", fontsize=8)
            ax.axis("off")
            if k == 0:
                ax.text(-0.05, 1.32, f'query: "{_short(query, 90)}"', transform=ax.transAxes, fontsize=10,
                        fontweight="bold")
    if show_matrix:
        ax = fig.add_subplot(grid[rows, :])
        shown = ax.imshow(similarities, aspect="auto", cmap="viridis")
        ax.set_yticks(range(rows))
        ax.set_yticklabels([_short(q, 40) for q in queries], fontsize=8)
        if len(gallery.names) <= 40:
            ax.set_xticks(range(len(gallery.names)))
            ax.set_xticklabels([_short(n, 18) for n in gallery.names], rotation=60, ha="right", fontsize=7)
        else:
            ax.set_xlabel(f"{len(gallery.names)} gallery images")
        best = similarities.argmax(axis=1)
        ax.scatter(best, np.arange(rows), marker="*", color="white", s=60)
        fig.colorbar(shown, ax=ax, fraction=0.02, pad=0.01, label="cosine similarity")
        ax.set_title("Text × image similarity matrix (★ = best image per query): CLIP is trained so that "
                     "matching pairs are the largest values", fontsize=9)
    fig.suptitle(f"CLIP text → image search{f' ({model})' if model else ''}: {len(gallery.names)} images", fontsize=12)
    return figure_to_image(fig)
