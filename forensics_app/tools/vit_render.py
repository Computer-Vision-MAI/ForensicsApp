"""Set 10 extra: figures of the ViT tools (matplotlib, no Tk, no PyTorch)."""

from __future__ import annotations

from collections.abc import Sequence
import textwrap

from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from .bow_render import figure_to_image
from .vit_data import extract_patches, resize, shifted_images

TRAIN_COLOR = "#2a7ab0"
VAL_COLOR = "#f0a202"
TEST_COLOR = "#2e9e4f"
WRONG_COLOR = "#d1495b"


def _grid_lines(ax, size: int, patch: int, color: str = "white") -> None:
    for k in range(0, size + 1, patch):
        ax.axhline(k - 0.5, color=color, linewidth=0.4)
        ax.axvline(k - 0.5, color=color, linewidth=0.4)


def _mosaic(patches: np.ndarray, gap: int = 1, fill: int = 255) -> np.ndarray:
    """(rows, cols, p, p, C) patches -> one image with a gap between the patches."""
    rows, cols, p, _, c = patches.shape
    out = np.full((rows * (p + gap) - gap, cols * (p + gap) - gap, c), fill, dtype=np.uint8)
    for r in range(rows):
        for col in range(cols):
            out[r * (p + gap):r * (p + gap) + p, col * (p + gap):col * (p + gap) + p] = patches[r, col]
    return out


def render_patches(image: np.ndarray, image_size: int, patch: int, title: str, spt: bool = True) -> Image.Image:
    """The notebook's patch visualisation, and the four shifted images of SPT with one token's content."""
    big = resize(image, image_size)
    patches = extract_patches(big, patch)
    rows, cols = patches.shape[:2]
    fig = Figure(figsize=(15, 9.2 if spt else 4.8), dpi=100)
    top = fig.add_gridspec(1, 3, left=0.03, right=0.98, top=0.9 if spt else 0.85, bottom=0.52 if spt else 0.05,
                           wspace=0.15)
    ax = fig.add_subplot(top[0, 0])
    ax.imshow(image, interpolation="nearest")
    ax.set_title(f"{title}\n({image.shape[1]}×{image.shape[0]})", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(top[0, 1])
    ax.imshow(big)
    _grid_lines(ax, image_size, patch)
    ax.set_title(f"Resized to {image_size}×{image_size}, cut into {rows}×{cols} = {rows * cols} patches of "
                 f"{patch}×{patch}", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(top[0, 2])
    ax.imshow(_mosaic(patches))
    ax.set_title(f"Patches: each one is a token of {patch}·{patch}·3 = {patch * patch * 3} numbers", fontsize=10)
    ax.axis("off")
    if spt:
        bottom = fig.add_gridspec(1, 6, left=0.03, right=0.98, top=0.42, bottom=0.04, wspace=0.12)
        shifts = shifted_images(big, patch)
        r, c = rows // 2, cols // 2
        y0, x0 = r * patch, c * patch
        for k, (name, shifted) in enumerate(shifts):
            ax = fig.add_subplot(bottom[0, k])
            ax.imshow(shifted)
            _grid_lines(ax, image_size, patch, "#ffffff55")
            ax.add_patch(_rect(x0, y0, patch))
            ax.set_title(f"shift {name} (½ patch = {patch // 2} px)", fontsize=9)
            ax.axis("off")
        ax = fig.add_subplot(bottom[0, 4:])
        stack = [big] + [s for _, s in shifts]
        views = np.stack([s[y0:y0 + patch, x0:x0 + patch] for s in stack])[None]
        ax.imshow(_mosaic(views, gap=1), interpolation="nearest")
        ax.set_title(f"SPT token of the marked patch:\nthe 5 views stacked = {patch}×{patch}×15 = {patch * patch * 15} "
                     "numbers\n(original, left-up, left-down, right-up, right-down)", fontsize=9)
        ax.axis("off")
        fig.text(0.03, 0.46, "Shifted Patch Tokenization (Exercise): the image is shifted diagonally by half a patch "
                 "in 4 directions and stacked with itself on the channel axis before cutting the patches",
                 fontsize=10, fontweight="bold")
    return figure_to_image(fig)


def _rect(x: int, y: int, patch: int):
    from matplotlib.patches import Rectangle

    return Rectangle((x - 0.5, y - 0.5), patch, patch, fill=False, edgecolor="#ff2fb4", linewidth=1.8)


# --------------------------------------------------------------------------- passport


def _epochs_axis(ax, n: int) -> None:
    if n <= 16:
        ax.set_xticks(np.arange(1, n + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)


def _plot_history(ax_loss, ax_acc, passport) -> None:
    h = passport.history
    epochs = np.arange(1, len(h.get("loss", [])) + 1)
    best = passport.data.get("training", {}).get("best_epoch")
    ax_loss.plot(epochs, h.get("loss", []), "-o", ms=3, color=TRAIN_COLOR, label="train")
    ax_loss.plot(epochs, h.get("val_loss", []), "-o", ms=3, color=VAL_COLOR, label="validation")
    ax_loss.set_title("Train and validation loss", fontsize=10)
    ax_acc.plot(epochs, h.get("accuracy", []), "-o", ms=3, color=TRAIN_COLOR, label="train accuracy")
    ax_acc.plot(epochs, h.get("val_accuracy", []), "-o", ms=3, color=VAL_COLOR, label="validation accuracy")
    ax_acc.plot(epochs, h.get("val_top5", []), "--", color=VAL_COLOR, label="validation top-5")
    ax_acc.set_title("Accuracy (train includes the augmentation and dropout)", fontsize=10)
    for ax in (ax_loss, ax_acc):
        if best:
            ax.axvline(best, color="#888", linestyle="--", linewidth=1)
        _epochs_axis(ax, len(epochs))
        ax.legend(fontsize=8)


def _passport_text(passport) -> str:
    d = passport.data
    t = d.get("training", {})
    a = d.get("architecture", {})
    test = passport.test
    lines = [
        passport.name,
        f"Dataset: {d['dataset']['label'].split(' (')[0]}",
        f"  train {d['dataset']['train']:,}, validation {d['dataset']['validation']:,}, test {d['dataset']['test']:,}",
        f"Tokens: {a.get('tokens')} ({a.get('tokenizer')}), attention {a.get('attention')}",
        f"Parameters: {a.get('parameters', 0):,}",
        f"{t.get('optimizer')}",
        f"Batch {t.get('batch')}" + (f" (micro-batches of {t.get('micro_batch')})" if t.get("micro_batch") else ""),
        f"Epochs {t.get('epochs_done')} of {t.get('epochs')}, best {t.get('best_epoch')}"
        + (" (stopped early)" if t.get("stopped_early") else ""),
        f"Device {t.get('device')}, {t.get('seconds', 0) / 60:.1f} min",
        "",
        f"Test accuracy: {test.get('accuracy', 0):.2%}",
        f"Test top-5 accuracy: {test.get('top5', 0):.2%}",
        f"Test loss: {test.get('loss', 0):.3f}",
        f"Macro F1: {test.get('macro_f1', 0):.3f}",
    ]
    return "\n".join(lines)


def _plot_confusion(ax, confusion: np.ndarray, names: Sequence[str]) -> None:
    rows = confusion.sum(axis=1, keepdims=True)
    norm = confusion / np.maximum(rows, 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=max(0.5, float(norm.max())))
    n = len(names)
    if n <= 12:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=60, fontsize=7, ha="right")
        ax.set_yticklabels(names, fontsize=7)
        for (r, c), v in np.ndenumerate(norm):
            if v >= 0.05:
                ax.text(c, r, f"{v:.0%}", ha="center", va="center", fontsize=6, color="white" if v > 0.5 else "#222")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Test confusion matrix ({n} classes, rows normalised)", fontsize=10)


def _plot_f1(ax, f1: np.ndarray, names: Sequence[str], k: int = 10) -> None:
    order = np.argsort(f1)
    chosen = np.concatenate([order[:k], order[-k:]]) if len(f1) > 2 * k else order
    colors = [WRONG_COLOR if i in order[:k] else TEST_COLOR for i in chosen]
    ax.barh(np.arange(len(chosen)), f1[chosen], color=colors)
    ax.set_yticks(np.arange(len(chosen)))
    ax.set_yticklabels([names[i] for i in chosen], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 on test")
    ax.set_title(f"Classification report: worst and best {k} classes" if len(f1) > 2 * k else "F1 per class",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)


def render_vit_passport(passport) -> Image.Image:
    test = passport.test
    names = passport.class_names
    fig = Figure(figsize=(16, 10), dpi=90)
    grid = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.92, bottom=0.06, hspace=0.35, wspace=0.3)
    _plot_history(fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), passport)
    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    ax.text(0, 1, _passport_text(passport), va="top", fontsize=9, family="monospace")
    ax.set_title("Passport", loc="left", fontsize=10)
    if test.get("confusion"):
        _plot_confusion(fig.add_subplot(grid[1, 0:2]), np.array(test["confusion"]), names)
        _plot_f1(fig.add_subplot(grid[1, 2]), np.array(test["f1"]), names)
    fig.suptitle(f"ViT passport: {passport.name}", fontsize=13)
    return figure_to_image(fig)


# --------------------------------------------------------------------------- classification


def render_vit_classification(image: np.ndarray, network_input: np.ndarray, probs: np.ndarray, names: Sequence[str],
                              rollout: np.ndarray | None, true_label: int | None, title: str,
                              image_size: int, top_k: int = 10) -> Image.Image:
    fig = Figure(figsize=(15, 4.8), dpi=100)
    cols = 4 if rollout is not None else 3
    grid = fig.add_gridspec(1, cols, left=0.03, right=0.98, top=0.82, bottom=0.12, wspace=0.35,
                            width_ratios=[1, 1] + ([1] if rollout is not None else []) + [1.6])
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image)
    ax.set_title(f"Image ({image.shape[1]}×{image.shape[0]})", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(network_input, interpolation="nearest")
    ax.set_title("Network input: 32×32 (resized to\n" f"{image_size}×{image_size} inside the model)", fontsize=9)
    ax.axis("off")
    if rollout is not None:
        ax = fig.add_subplot(grid[0, 2])
        big = resize(network_input, image_size)
        heat = np.asarray(Image.fromarray(((rollout - rollout.min()) / (np.ptp(rollout) + 1e-12) * 255).astype(np.uint8))
                          .resize((image_size, image_size), Image.Resampling.BILINEAR)) / 255.0
        colors = colormaps["jet"](heat)[..., :3] * 255
        ax.imshow(np.clip(0.55 * big + 0.45 * colors, 0, 255).astype(np.uint8))
        ax.set_title("Attention rollout: patches the\ntokens attend to most (red)", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(grid[0, -1])
    order = np.argsort(probs)[-top_k:]
    colors = []
    for i in order:
        if i == int(np.argmax(probs)):
            colors.append(TEST_COLOR if true_label is None or i == true_label else WRONG_COLOR)
        else:
            colors.append(VAL_COLOR if true_label is not None and i == true_label else TRAIN_COLOR)
    ax.barh([textwrap.shorten(names[i], 22) for i in order], probs[order], color=colors)
    for y, v in enumerate(probs[order]):
        ax.text(min(v + 0.01, 0.9), y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"Top {top_k} classes" + ("" if true_label is None else f" (true: {names[true_label]})"), fontsize=10)
    fig.suptitle(title, fontsize=12)
    return figure_to_image(fig)


# --------------------------------------------------------------------------- comparison


def _mcnemar(only_a: int, only_b: int) -> float:
    """McNemar's chi-squared (with continuity correction) on the images where the models disagree."""
    total = only_a + only_b
    return 0.0 if total == 0 else (abs(only_a - only_b) - 1) ** 2 / total


def render_vit_comparison(passports: Sequence) -> Image.Image:
    fig = Figure(figsize=(16, 9.5), dpi=90)
    grid = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.92, bottom=0.08, hspace=0.38, wspace=0.3)
    colors = colormaps["tab10"](np.arange(len(passports)) % 10)
    ax_acc = fig.add_subplot(grid[0, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    n = 1
    for p, color in zip(passports, colors):
        h = p.history
        epochs = np.arange(1, len(h.get("val_accuracy", [])) + 1)
        n = max(n, len(epochs))
        ax_acc.plot(epochs, h.get("val_accuracy", []), "-o", ms=3, color=color, label=p.name)
        ax_acc.plot(epochs, h.get("accuracy", []), ":", color=color)
        ax_loss.plot(epochs, h.get("val_loss", []), "-o", ms=3, color=color, label=p.name)
    ax_acc.set_title("Validation accuracy (solid) and train accuracy (dotted)", fontsize=10)
    ax_loss.set_title("Validation loss", fontsize=10)
    for ax in (ax_acc, ax_loss):
        _epochs_axis(ax, n)
        ax.legend(fontsize=7)

    ax = fig.add_subplot(grid[0, 2])
    metrics = (("accuracy", "test accuracy"), ("top5", "top-5"), ("macro_f1", "macro F1"))
    x = np.arange(len(metrics))
    width = 0.8 / len(passports)
    for k, (p, color) in enumerate(zip(passports, colors)):
        values = [p.test.get(m, 0) for m, _ in metrics]
        bars = ax.bar(x + (k - (len(passports) - 1) / 2) * width, values, width, color=color, label=p.name)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Test set", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    if len(passports) >= 2 and passports[0].test.get("f1") and passports[1].test.get("f1"):
        a, b = passports[0], passports[1]
        fa, fb = np.array(a.test["f1"]), np.array(b.test["f1"])
        ax = fig.add_subplot(grid[1, 0])
        ax.scatter(fa, fb, s=14, color=TRAIN_COLOR, alpha=0.7)
        ax.plot([0, 1], [0, 1], color="#888", linewidth=0.8)
        ax.set_xlabel(f"F1 {a.name}")
        ax.set_ylabel(f"F1 {b.name}")
        ax.set_title(f"F1 per class: {int(np.sum(fb > fa))} classes better, {int(np.sum(fb < fa))} worse", fontsize=10)
        ax.grid(alpha=0.3)
        names = a.class_names
        diff = fb - fa
        order = np.argsort(diff)
        chosen = np.concatenate([order[:8], order[-8:]]) if len(diff) > 16 else order
        ax = fig.add_subplot(grid[1, 1])
        ax.barh(np.arange(len(chosen)), diff[chosen], color=[WRONG_COLOR if diff[i] < 0 else TEST_COLOR for i in chosen])
        ax.set_yticks(np.arange(len(chosen)))
        ax.set_yticklabels([names[i] for i in chosen], fontsize=7)
        ax.axvline(0, color="#444", linewidth=0.8)
        ax.set_title(f"Largest F1 changes ({b.name} − {a.name})", fontsize=10)
        pa, pb, truth = a.test.get("predictions"), b.test.get("predictions"), a.test.get("labels")
        ax = fig.add_subplot(grid[1, 2])
        ax.axis("off")
        if pa and pb and truth and len(pa) == len(pb) == len(truth):
            pa, pb, truth = np.array(pa), np.array(pb), np.array(truth)
            ra, rb = pa == truth, pb == truth
            lines = [
                f"A = {a.name}",
                f"B = {b.name}",
                f"Test images: {len(truth):,}",
                f"Same prediction: {np.mean(pa == pb):.1%}",
                "",
                f"Both right:  {np.sum(ra & rb):6,}",
                f"Only A:      {np.sum(ra & ~rb):6,}",
                f"Only B:      {np.sum(~ra & rb):6,}",
                f"Both wrong:  {np.sum(~ra & ~rb):6,}",
                "",
                f"McNemar: chi2 = {_mcnemar(int(np.sum(ra & ~rb)), int(np.sum(~ra & rb))):.1f}",
                "(> 3.84: the difference is significant, p < 0.05)",
            ]
            ax.text(0, 1, "\n".join(lines), va="top", fontsize=10, family="monospace")
            ax.set_title("Image by image", loc="left", fontsize=10)
    fig.suptitle("ViT models compared", fontsize=13)
    return figure_to_image(fig)
