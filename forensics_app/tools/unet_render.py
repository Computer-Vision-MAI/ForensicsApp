"""Set 11: figures of the U-Net tools (matplotlib, no Tk, no PyTorch).

* ``render_dataset_overview``: class frequencies and a few images with their masks;
* ``render_segmentation``: original / mask / overlay as in the slides, plus the ground
  truth, the error map (TP, FP, FN) and the probability map when they are available;
* ``render_passport``: the model passport (training curves, Dice vs threshold, per-class
  Dice, Dice vs IoU, confusion matrix) and the worst validation images;
* ``render_comparison``: several models side by side (Exercises 2 and 3).
"""

from __future__ import annotations

from collections.abc import Sequence
import textwrap

from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from .bow_render import figure_to_image
from .segmentation_metrics import iou_from_dice

BACKGROUND_COLOR = "#2b2b3d"
TRAIN_COLOR = "#2a7ab0"
VAL_COLOR = "#f0a202"
TEST_COLOR = "#2e9e4f"
TP_COLOR = np.array([46, 158, 79])  # food found
FP_COLOR = np.array([209, 73, 91])  # predicted, but not in the truth
FN_COLOR = np.array([42, 122, 176])  # in the truth, but missed
WRONG_COLOR = np.array([240, 162, 2])  # both foreground, different class
OVERLAY_ALPHA = 0.5


def class_colors(n: int) -> np.ndarray:
    """RGB (0-255) per class; class 0 (background / not food) is dark."""
    if n == 2:
        rows = ["#2b2b3d", "#ff2fb4"]  # magenta: food itself is often orange or green
        return np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in rows], dtype=np.uint8)
    if n == 3:
        rows = ["#2b2b3d", "#4e79a7", "#ff2fb4"]
        return np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in rows], dtype=np.uint8)
    palette = np.vstack([colormaps[name](np.arange(20))[:, :3] for name in ("tab20", "tab20b", "tab20c")])
    colors = np.vstack([[0.17, 0.17, 0.24], palette[: n - 1]])
    return (colors * 255).astype(np.uint8)


def colorize(mask: np.ndarray, n: int) -> np.ndarray:
    return class_colors(n)[np.clip(mask, 0, n - 1)]


def overlay(image: np.ndarray, mask: np.ndarray, n: int, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """The image with every non-background class tinted with its colour."""
    image = np.asarray(image, dtype=np.float64)
    colors = colorize(mask, n).astype(np.float64)
    out = image.copy()
    fg = mask > 0
    out[fg] = (1 - alpha) * image[fg] + alpha * colors[fg]
    out[~fg] = 0.75 * image[~fg]  # background slightly darker, so the classes stand out
    return np.clip(out, 0, 255).astype(np.uint8)


def error_map(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Grey image with TP green, FP red, FN blue, wrong class orange."""
    grey = np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("L"), dtype=np.float64)
    out = np.repeat((0.35 * grey)[..., None], 3, axis=2)
    pred, gt = np.asarray(pred), np.asarray(gt)
    out[(pred > 0) & (pred == gt)] = TP_COLOR
    out[(pred > 0) & (gt == 0)] = FP_COLOR
    out[(pred == 0) & (gt > 0)] = FN_COLOR
    out[(pred > 0) & (gt > 0) & (pred != gt)] = WRONG_COLOR
    return out.astype(np.uint8)


def _legend_for(ax, mask_list: Sequence[np.ndarray], names: Sequence[str], max_items: int = 14) -> None:
    present = np.unique(np.concatenate([np.unique(m) for m in mask_list]))
    present = [int(c) for c in present if 0 <= c < len(names)]
    colors = class_colors(len(names)) / 255.0
    handles = [Patch(color=colors[c], label=names[c]) for c in present[:max_items]]
    if len(present) > max_items:
        handles.append(Patch(color="white", label=f"+{len(present) - max_items} more"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)


def _error_legend(ax, multiclass: bool) -> None:
    handles = [Patch(color=TP_COLOR / 255, label="correct (TP)"),
               Patch(color=FP_COLOR / 255, label="false alarm (FP)"),
               Patch(color=FN_COLOR / 255, label="missed (FN)")]
    if multiclass:
        handles.append(Patch(color=WRONG_COLOR / 255, label="wrong class"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)


def _no_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------- dataset


def render_dataset_overview(stats, class_names: Sequence[str], samples: Sequence[tuple[str, np.ndarray, np.ndarray]],
                            task_names: Sequence[str], task_lookup: Sequence[int]) -> Image.Image:
    """``stats``: DatasetStats of the splits; ``samples``: (name, image, source mask)."""
    train = stats[0]
    n = len(class_names)
    fig = Figure(figsize=(14, 8.6), dpi=100)
    top = fig.add_gridspec(1, 4, left=0.06, right=0.98, top=0.95, bottom=0.62, wspace=0.25)
    grid = fig.add_gridspec(2, 4, left=0.03, right=0.98, top=0.5, bottom=0.02, hspace=0.3, wspace=0.08)

    ax = fig.add_subplot(top[0, :3])
    share = train.pixels_per_class / max(train.pixels_per_class.sum(), 1)
    colors = class_colors(n) / 255.0
    ax.bar(np.arange(n), share * 100, color=colors, edgecolor="#444", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([textwrap.shorten(name, 16, placeholder="…") for name in class_names], rotation=70,
                       ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_ylabel("% of the training pixels (log)")
    ax.set_title(f"Class frequencies in XTrain ({train.images} masks): background + tray "
                 f"{share[:2].sum():.0%}, food {share[5:].sum():.1%}; missing bars = class never in XTrain",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(top[0, 3])
    ax.hist(train.food_fraction * 100, bins=30, color=TP_COLOR / 255)
    ax.set_xlabel("% of food pixels per image")
    ax.set_ylabel("images")
    ax.set_title(f"Food per image (mean {train.food_fraction.mean():.1%})", fontsize=10)

    task_n = len(task_names)
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[: len(task_lookup)] = task_lookup
    task_text = ", ".join(task_names) if task_n <= 3 else f"all {task_n} classes"
    for k, (name, image, mask) in enumerate(samples[:4]):
        ax = fig.add_subplot(grid[0, k])
        ax.imshow(image)
        ax.set_title(name, fontsize=9)
        _no_axes(ax)
        ax = fig.add_subplot(grid[1, k])
        ax.imshow(overlay(image, lookup[mask], task_n))
        present = [class_names[c] for c in np.unique(mask) if c >= 5]
        ax.set_title(f"mask ({task_text})\n" + textwrap.shorten(", ".join(present) or "no food", 52), fontsize=7.5)
        _no_axes(ax)
    return figure_to_image(fig)


# --------------------------------------------------------------------------- segmentation


def render_segmentation(
    image: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    class_names: Sequence[str],
    gt: np.ndarray | None = None,
    threshold: float | None = None,
    title: str = "",
    network_input: np.ndarray | None = None,
) -> Image.Image:
    """Original / predicted mask / overlay (as in the slides) and, with a ground truth,
    the true mask, the error map and the probability (or confidence) map."""
    n = len(class_names)
    binary = n == 2
    rows = 2
    fig = Figure(figsize=(15, 7.4), dpi=100)
    grid = fig.add_gridspec(rows, 3, hspace=0.18, wspace=0.42, left=0.02, right=0.94, top=0.9, bottom=0.03)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image)
    ax.set_title(f"Original image ({image.shape[1]}×{image.shape[0]})", fontsize=10)
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(colorize(pred, n), interpolation="nearest")
    ax.set_title("Segmentation mask (prediction)" + (f", threshold {threshold:.2f}" if binary and threshold is not None else ""),
                 fontsize=10)
    _legend_for(ax, [pred] + ([gt] if gt is not None else []), class_names)
    ax = fig.add_subplot(grid[0, 2])
    ax.imshow(overlay(image, pred, n))
    ax.set_title("Overlayed mask", fontsize=10)

    if gt is not None:
        ax = fig.add_subplot(grid[1, 0])
        ax.imshow(colorize(gt, n), interpolation="nearest")
        ax.set_title("Ground truth mask", fontsize=10)
        ax = fig.add_subplot(grid[1, 1])
        ax.imshow(error_map(image, pred, gt))
        ax.set_title("Errors: where the prediction differs from the truth", fontsize=10)
        _error_legend(ax, not binary)
    elif network_input is not None:
        ax = fig.add_subplot(grid[1, 0])
        ax.imshow(network_input)
        ax.set_title(f"Network input ({network_input.shape[1]}×{network_input.shape[0]})", fontsize=10)
    ax = fig.add_subplot(grid[1, 2] if gt is not None or network_input is not None else grid[1, 1])
    shown = ax.imshow(confidence, cmap="magma", vmin=0, vmax=1)
    ax.set_title("P(food) per pixel" if binary else "Confidence (highest class probability)", fontsize=10)
    fig.colorbar(shown, ax=ax, fraction=0.035, pad=0.02)
    for a in fig.axes:
        if a.get_label() != "<colorbar>":
            _no_axes(a)
    if title:
        fig.suptitle(title, fontsize=12)
    return figure_to_image(fig)


# --------------------------------------------------------------------------- passport


def _epoch_ticks(ax, n: int) -> None:
    if n <= 16:
        ax.set_xticks(np.arange(1, n + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0.5, n + 0.5)


def _plot_curves(ax, history: dict, best_epoch: int | None) -> None:
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if not epochs.size:
        ax.text(0.5, 0.5, "no history", ha="center", va="center")
        return
    ax.plot(epochs, history["train_loss"], "-o", ms=3, color=TRAIN_COLOR, label="train loss")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], "-o", ms=3, color=VAL_COLOR, label="validation loss")
    if best_epoch:
        ax.axvline(best_epoch, color="#888", linestyle="--", linewidth=1, label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    _epoch_ticks(ax, len(epochs))
    ax.set_title("Loss", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _plot_dice_history(ax, history: dict, best_epoch: int | None) -> None:
    epochs = np.arange(1, len(history.get("val_dice", [])) + 1)
    if not epochs.size:
        ax.text(0.5, 0.5, "no history", ha="center", va="center")
        return
    ax.plot(epochs, history["val_dice"], "-o", ms=3, color=VAL_COLOR, label="validation Dice")
    if history.get("val_iou"):
        ax.plot(epochs, history["val_iou"], "-s", ms=3, color=TEST_COLOR, label="validation IoU")
    if best_epoch:
        ax.axvline(best_epoch, color="#888", linestyle="--", linewidth=1)
        ax.annotate(f"{history['val_dice'][best_epoch - 1]:.3f}", (best_epoch, history["val_dice"][best_epoch - 1]),
                    textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax.set_xlabel("epoch")
    _epoch_ticks(ax, len(epochs))
    ax.set_title("Validation Dice and IoU", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def _plot_threshold(ax, thresholds: np.ndarray, dice: np.ndarray, chosen: float | None,
                    label: str = "validation") -> None:
    if not thresholds.size:
        ax.text(0.5, 0.5, "only for the binary task", ha="center", va="center")
        _no_axes(ax)
        return
    ax.plot(thresholds, dice, "-o", ms=3, color=VAL_COLOR)
    best = int(np.argmax(dice))
    ax.plot(thresholds[best], dice[best], "o", color=FP_COLOR / 255, ms=7,
            label=f"best {thresholds[best]:.2f}: {dice[best]:.3f}")
    if chosen is not None:
        ax.axvline(chosen, color="#888", linestyle="--", linewidth=1, label=f"used {chosen:.2f}")
    ax.set_xlabel("threshold on P(food)")
    ax.set_ylabel("mean Dice")
    ax.set_title(f"Exercise 1: Dice vs threshold ({label})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _plot_class_dice(ax, class_names: Sequence[str], val: np.ndarray, test: np.ndarray | None,
                     label: str = "validation") -> None:
    val = np.asarray(val, dtype=float)
    has_test = test is not None and np.asarray(test).size
    test = np.asarray(test, dtype=float) if has_test else None
    # only the classes that appear in the evaluated split(s): 43 names do not fit otherwise
    present = ~np.isnan(val) if test is None else ~(np.isnan(val) & np.isnan(test))
    keep = np.flatnonzero(present) if present.any() else np.arange(len(class_names))
    names = [class_names[i] for i in keep]
    n = len(names)
    order = np.arange(n)
    width = 0.4 if test is not None else 0.8
    ax.barh(order - (width / 2 if width < 0.8 else 0), np.nan_to_num(val[keep]), height=width, color=VAL_COLOR,
            label=label)
    if test is not None:
        ax.barh(order + width / 2, np.nan_to_num(test[keep]), height=width, color=TEST_COLOR, label="test")
    ax.set_yticks(order)
    ax.set_yticklabels(names, fontsize=9 if n <= 10 else 7 if n <= 20 else 5.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Dice")
    title = "Dice of each class" + (" (Exercise 6)" if len(class_names) == 3 else "")
    if n < len(class_names):
        title += f" ({n} present)"
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def _plot_dice_vs_iou(ax, dice: np.ndarray, iou: np.ndarray, label: str) -> None:
    ax.scatter(dice, iou, s=12, color=VAL_COLOR, alpha=0.7, label=f"{label} images")
    d = np.linspace(0, 1, 101)
    ax.plot(d, iou_from_dice(d), color="#444", linewidth=1, label="IoU = Dice / (2 − Dice)")
    ax.plot(d, d, color="#bbb", linewidth=0.8, linestyle=":", label="IoU = Dice")
    low = float(min(np.min(dice, initial=1.0), np.min(iou, initial=1.0)))
    ax.set_xlim(max(0, low - 0.05), 1.0)
    ax.set_ylim(max(0, low - 0.05), 1.0)
    ax.set_xlabel("Dice")
    ax.set_ylabel("IoU")
    ax.set_title("Exercise 4: Dice vs IoU per image", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")


def _plot_confusion(ax, confusion: np.ndarray, class_names: Sequence[str], label: str = "validation") -> None:
    rows = confusion.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(rows > 0, confusion / rows, 0)
    n = len(class_names)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    if n <= 6:
        for (r, c), value in np.ndenumerate(norm):
            ax.text(c, r, f"{value:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.6 else "#222")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, fontsize=8, rotation=20)
        ax.set_yticklabels(class_names, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Pixel confusion matrix ({label}, % of each true class)", fontsize=10)


def _plot_dice_histogram(ax, dice: np.ndarray, label: str, test: np.ndarray | None = None) -> None:
    ax.hist(dice, bins=20, color=VAL_COLOR, label=label)
    if test is not None and test.size:
        ax.hist(test, bins=20, color=TEST_COLOR, alpha=0.8, label="test")
    ax.set_xlabel("Dice per image")
    ax.set_ylabel("images")
    ax.set_title("Distribution of the per-image Dice", fontsize=10)
    ax.legend(fontsize=8)


def _passport_text(passport) -> str:
    data = passport.data
    training = data.get("training", {})
    architecture = data.get("architecture", {})
    dataset = data.get("dataset", {})
    val = passport.split("val")
    test = passport.split("test")
    lines = [
        f"{passport.name}",
        f"Task: {data['task']['label']}",
        f"Input: {data['input']['width']}×{data['input']['height']}, {architecture.get('output', '')}",
        f"Parameters: {architecture.get('parameters', 0):,}, base {architecture.get('base_channels', '?')} ch, "
        f"skip connections {'yes' if architecture.get('skip_connections', True) else 'NO'}",
        f"Loss: {training.get('loss', '?')}, {training.get('optimizer', '?')}, batch {training.get('batch_size', '?')}",
        f"Augmentation: {training.get('augmentation', '?')}",
        f"Epochs: {training.get('epochs_done', '?')} of {training.get('epochs', '?')}, best {training.get('best_epoch', '?')}"
        + (" (stopped early)" if training.get("stopped_early") else ""),
        f"Images: train {dataset.get('train_images', '?')}, val {dataset.get('val_images', '?')}, "
        f"test {dataset.get('test_images', '?')}",
        f"Device: {training.get('device', '?')}, {training.get('seconds', 0) / 60:.1f} min",
    ]
    if val:
        lines.append(f"Validation: Dice {val.get('dice_mean', 0):.3f}, IoU {val.get('iou_mean', 0):.3f}, "
                     f"pixel acc. {val.get('pixel_accuracy', 0):.3f}")
    if test:
        lines.append(f"Test: Dice {test.get('dice_mean', 0):.3f} ± {test.get('dice_std', 0):.3f}, "
                     f"IoU {test.get('iou_mean', 0):.3f} ({test.get('images', '?')} images)")
    return "\n".join(lines)


def render_passport(passport, evaluation=None, worst: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = (),
                    split_label: str = "validation") -> Image.Image:
    """The passport figure. ``evaluation`` (a SplitEvaluation) replaces the saved validation
    metrics when the model was re-evaluated; ``worst``: (name, image, gt, pred, dice)."""
    from .segmentation_metrics import split_from_dict

    names = passport.class_names
    saved_val = split_from_dict(passport.split("val"), names)
    saved_test = split_from_dict(passport.split("test"), names)
    if evaluation is not None:
        current = {
            "dice": evaluation.dice, "iou": evaluation.iou, "class_dice": evaluation.class_dice,
            "confusion": evaluation.confusion.astype(float), "sweep_thresholds": evaluation.sweep_thresholds,
            "sweep_dice": evaluation.sweep_dice,
        }
    else:
        current = saved_val
    # inches: title 0.5, two rows of charts 7.6, then a header and 2.6 per worst image
    plots_h, worst_h, header_h = 7.6, 2.6, 0.5
    height = 0.5 + plots_h + (header_h + 0.35 + worst_h * len(worst) if worst else 0) + 0.3
    fig = Figure(figsize=(16, height), dpi=90)
    top = 1 - 0.5 / height
    grid = fig.add_gridspec(2, 4, left=0.05, right=0.98, top=top - 0.25 / height, bottom=top - plots_h / height,
                            hspace=0.42, wspace=0.32)

    _plot_curves(fig.add_subplot(grid[0, 0]), passport.history, passport.best_epoch)
    _plot_dice_history(fig.add_subplot(grid[0, 1]), passport.history, passport.best_epoch)
    if passport.binary:
        _plot_threshold(fig.add_subplot(grid[0, 2]), current["sweep_thresholds"], current["sweep_dice"],
                        evaluation.threshold if evaluation is not None else passport.threshold, split_label)
    else:
        _plot_dice_histogram(fig.add_subplot(grid[0, 2]), current["dice"], split_label,
                             saved_test["dice"] if evaluation is None else None)
    ax = fig.add_subplot(grid[0, 3])
    ax.axis("off")
    ax.text(0, 1, _passport_text(passport), va="top", ha="left", fontsize=8.2, family="monospace", wrap=True)
    ax.set_title("Passport", fontsize=10, loc="left")

    if passport.binary:
        _plot_dice_vs_iou(fig.add_subplot(grid[1, 0]), current["dice"], current["iou"], split_label)
        _plot_dice_histogram(fig.add_subplot(grid[1, 1]), current["dice"], split_label,
                             saved_test["dice"] if evaluation is None else None)
        _plot_confusion(fig.add_subplot(grid[1, 2:]), current["confusion"], names, split_label)
    else:
        _plot_dice_vs_iou(fig.add_subplot(grid[1, 0]), current["dice"], current["iou"], split_label)
        _plot_class_dice(fig.add_subplot(grid[1, 1]), names, current["class_dice"],
                         saved_test["class_dice"] if evaluation is None else None, split_label)
        _plot_confusion(fig.add_subplot(grid[1, 2:]), current["confusion"], names, split_label)

    n = len(names)
    if worst:
        worst_top = top - (plots_h + header_h + 0.35) / height
        fig.text(0.05, worst_top + 0.3 / height, f"Exercise 5: the {len(worst)} {split_label} images with the lowest "
                 "Dice (image, ground truth, prediction, errors)", fontsize=11, fontweight="bold")
        rows = fig.add_gridspec(len(worst), 4, left=0.05, right=0.9, top=worst_top,
                                bottom=worst_top - worst_h * len(worst) / height, hspace=0.25, wspace=0.05)
        for r, (name, image, gt, pred, dice) in enumerate(worst):
            panels = ((image, name), (colorize(gt, n), "ground truth"),
                      (colorize(pred, n), f"prediction, Dice {dice:.3f}"), (error_map(image, pred, gt), "errors"))
            for c, (shown, text) in enumerate(panels):
                ax = fig.add_subplot(rows[r, c])
                ax.imshow(shown, interpolation="nearest")
                ax.set_title(text, fontsize=9)
                _no_axes(ax)
            if r == 0:
                _error_legend(fig.axes[-1], n > 2)
    fig.suptitle(f"U-Net passport: {passport.name}", fontsize=13, y=1 - 0.15 / height)
    return figure_to_image(fig)


# --------------------------------------------------------------------------- comparison


def render_comparison(passports: Sequence) -> Image.Image:
    fig = Figure(figsize=(16, 9), dpi=90)
    grid = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    colors = colormaps["tab10"](np.arange(len(passports)) % 10)
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_dice = fig.add_subplot(grid[0, 1])
    ax_thr = fig.add_subplot(grid[0, 2])
    for passport, color in zip(passports, colors):
        history = passport.history
        epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
        ax_loss.plot(epochs, history.get("train_loss", []), "-", color=color, label=passport.name)
        ax_dice.plot(epochs, history.get("val_dice", []), "-o", ms=2.5, color=color, label=passport.name)
        sweep = passport.split("val").get("threshold_sweep", {})
        if sweep.get("thresholds"):
            ax_thr.plot(sweep["thresholds"], sweep["dice"], "-", color=color, label=passport.name)
    ax_loss.set_title("Training loss (each model with its own loss!)", fontsize=10)
    ax_loss.set_xlabel("epoch")
    ax_dice.set_title("Validation Dice per epoch", fontsize=10)
    ax_dice.set_xlabel("epoch")
    ax_thr.set_title("Validation Dice vs threshold (binary models)", fontsize=10)
    ax_thr.set_xlabel("threshold")
    for ax in (ax_loss, ax_dice, ax_thr):
        ax.grid(alpha=0.3)
    ax_dice.legend(fontsize=7, loc="lower right")

    ax = fig.add_subplot(grid[1, :2])
    x = np.arange(len(passports))
    metrics = [("val Dice", "val", "dice_mean", VAL_COLOR), ("test Dice", "test", "dice_mean", TEST_COLOR),
               ("val IoU", "val", "iou_mean", "#b07aa1"), ("test IoU", "test", "iou_mean", "#59a14f")]
    width = 0.2
    for k, (label, split, key, color) in enumerate(metrics):
        values = [p.split(split).get(key) or 0 for p in passports]
        bars = ax.bar(x + (k - 1.5) * width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    short = [p.name.removeprefix("unet_").replace("_", " ") for p in passports]
    ax.set_xticklabels([textwrap.fill(name, 18) for name in short], fontsize=7)
    lows = [p.split(s).get(k) for p in passports for _, s, k, _ in metrics if p.split(s).get(k)]
    ax.set_ylim(max(0.0, min(lows, default=0) - 0.1), 1.02)
    ax.set_title("Main metric (binary: food; several classes: mean over the foreground classes)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    lines = []
    for p in passports:
        t = p.data.get("training", {})
        a = p.data.get("architecture", {})
        lines.append(f"{p.name}\n  {p.data['task']['key']}, loss {t.get('loss')}, skip "
                     f"{'yes' if a.get('skip_connections', True) else 'no'}, {a.get('parameters', 0):,} params,\n"
                     f"  best epoch {t.get('best_epoch')} of {t.get('epochs_done')}, {t.get('seconds', 0) / 60:.1f} min")
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=7.5, family="monospace")
    fig.suptitle("U-Net models compared", fontsize=13)
    return figure_to_image(fig)


def segmentation_cmap(n: int) -> ListedColormap:
    return ListedColormap(class_colors(n) / 255.0)
