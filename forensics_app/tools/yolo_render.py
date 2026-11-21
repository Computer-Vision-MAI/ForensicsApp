"""Drawings for the YOLO tools: boxes on the image and matplotlib figures (Agg -> PIL).

Like ``bow_render`` they use ``Figure`` directly instead of pyplot, so they are safe
to call from a worker thread and never open a window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bow_render import figure_to_image
from .detection_metrics import Detections, EvaluationResult, ImageMatch
from .yolo_dataset import DatasetStats, ground_truth
from .yolo_model import to_rgb

# The colours of the course notebook for the blood cells
CELL_COLORS = {"PC": (220, 30, 30), "RBC": (20, 150, 40), "WBC": (30, 60, 220)}
TP_COLOR, FP_COLOR, FN_COLOR = "#1fa83a", "#e0262b", "#f2a900"


def class_color(cls: int, name: str = "") -> tuple[int, int, int]:
    """A stable colour per class: the notebook colours for cells, tab20 for the rest."""
    if name in CELL_COLORS:
        return CELL_COLORS[name]
    r, g, b, _ = colormaps["tab20"](int(cls) % 20)
    return int(r * 255), int(g * 255), int(b * 255)


def _hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: Detections,
    names: Mapping[int, str],
    show_scores: bool = True,
) -> Image.Image:
    """The boxes on a copy of the image, with a filled label "class 0.87" like the slides."""
    canvas = to_rgb(image).copy()
    draw = ImageDraw.Draw(canvas)
    side = max(canvas.size)
    width = max(2, round(side / 300))
    font = _font(max(11, round(side / 45)))
    order = np.argsort(detections.scores) if detections.scores is not None else np.arange(len(detections))
    for i in order:  # best boxes drawn last, on top
        x1, y1, x2, y2 = (float(v) for v in detections.boxes[i])
        cls = int(detections.classes[i])
        name = names.get(cls, str(cls))
        color = class_color(cls, name)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text = f"{name} {detections.scores[i]:.2f}" if show_scores and detections.scores is not None else name
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1
        draw.rectangle((x1, ty, x1 + tw + 6, ty + th + 4), fill=color)
        draw.text((x1 + 3, ty + 2 - top), text, fill=(255, 255, 255), font=font)
    return canvas


def _boxes_on_axes(ax, detections: Detections, names: Mapping[int, str], label: bool = True,
                   linewidth: float = 1.5, linestyle: str = "-", color: str | None = None, scores: bool = False) -> None:
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.boxes[i]
        cls = int(detections.classes[i])
        name = names.get(cls, str(cls))
        edge = color or _hex(class_color(cls, name))
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=edge,
                               linewidth=linewidth, linestyle=linestyle))
        if label:
            text = name if not scores or detections.scores is None else f"{name} {detections.scores[i]:.2f}"
            ax.text(x1 + 2, y1 + 2, text, color="white", fontsize=6, va="top",
                    bbox={"facecolor": edge, "edgecolor": "none", "pad": 1, "alpha": 0.85})


def render_before_after_nms(image: Image.Image, raw: Detections, kept: Detections,
                            names: Mapping[int, str], iou: float) -> Image.Image:
    """Candidates before NMS next to the boxes that survive it."""
    rgb = np.asarray(to_rgb(image))
    aspect = min(max(image.width / max(image.height, 1), 0.4), 2.5)
    height = 7.0 if aspect < 1.2 else 5.0
    fig = Figure(figsize=(2 * height * aspect + 0.3, height + 0.5), dpi=100)
    for index, (dets, title) in enumerate(
        ((raw, f"Before NMS: {len(raw)} candidate boxes"), (kept, f"After our greedy NMS (IoU > {iou:.2f} removed): {len(kept)}"))
    ):
        ax = fig.add_subplot(1, 2, index + 1)
        ax.imshow(rgb)
        _boxes_on_axes(ax, dets, names, label=index == 1, linewidth=0.8 if index == 0 else 1.6, scores=True)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.03)
    return figure_to_image(fig)


def render_gt_vs_pred(
    image: Image.Image,
    pred: Detections,
    gt: Detections | None,
    names: Mapping[int, str],
    match: ImageMatch | None,
    iou_threshold: float = 0.5,
    title: str = "",
) -> Image.Image:
    """Set 9.3: ground truth | prediction | errors (TP green, FP red, missed yellow)."""
    rgb = np.asarray(to_rgb(image))
    panels = 3 if gt is not None and match is not None else 1
    fig = Figure(figsize=(5.2 * panels, 5.6), dpi=100)
    if title:
        fig.suptitle(title, fontsize=10)
    index = 1
    if gt is not None and match is not None:
        ax = fig.add_subplot(1, panels, index)
        ax.imshow(rgb)
        _boxes_on_axes(ax, gt, names, linewidth=1.6)
        ax.set_title(f"Ground truth: {len(gt)} cells", fontsize=10)
        ax.axis("off")
        index += 1
    ax = fig.add_subplot(1, panels, index)
    ax.imshow(rgb)
    _boxes_on_axes(ax, pred, names, linewidth=1.6, scores=True)
    ax.set_title(f"Prediction: {len(pred)} cells", fontsize=10)
    ax.axis("off")
    if gt is not None and match is not None:
        ax = fig.add_subplot(1, panels, index + 1)
        ax.imshow(rgb)
        _boxes_on_axes(ax, pred.subset(np.flatnonzero(match.pred_is_tp)), names, label=False, color=TP_COLOR)
        _boxes_on_axes(ax, pred.subset(np.flatnonzero(~match.pred_is_tp)), names, linewidth=2.2, color=FP_COLOR)
        _boxes_on_axes(ax, gt.subset(np.flatnonzero(~match.gt_found)), names, linewidth=2.2,
                       linestyle="--", color=FN_COLOR)
        ax.set_title(
            f"IoU ≥ {iou_threshold:.2f}: {match.true_positives} TP (green), "
            f"{match.false_positives} FP (red), {match.false_negatives} missed (yellow)",
            fontsize=9,
        )
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88 if title else 0.93, bottom=0.01, wspace=0.04)
    return figure_to_image(fig)


def render_dataset_overview(
    stats: Sequence[DatasetStats], names: Sequence[str], samples: Sequence[Path] = ()
) -> Image.Image:
    """Class distribution per split, cells per image, box sizes and a few labelled images."""
    rows = 2 if samples else 1
    fig = Figure(figsize=(15, 4.4 * rows), dpi=100)
    grid = fig.add_gridspec(rows, 4, hspace=0.3, wspace=0.3)

    ax = fig.add_subplot(grid[0, 0:2])
    x = np.arange(len(names))
    width = 0.8 / max(len(stats), 1)
    for i, s in enumerate(stats):
        bars = ax.bar(x + (i - (len(stats) - 1) / 2) * width, s.per_class, width, label=f"{s.split} ({s.images} images)")
        for bar, value in zip(bars, s.per_class):
            ax.text(bar.get_x() + bar.get_width() / 2, max(value, 1) * 1.08, str(int(value)), ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("Boxes (log scale)")
    ax.set_title("Boxes per class: heavily imbalanced (RBC ≫ WBC, PC)", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[0, 2])
    train = stats[0]
    if train.boxes_per_image.size:
        ax.hist(train.boxes_per_image, bins=range(0, int(train.boxes_per_image.max()) + 2), color="#5a7bb0")
    ax.set_xlabel("Boxes in the image")
    ax.set_ylabel("Images")
    ax.set_title(f"Cells per image ({train.split}), mean {train.boxes_per_image.mean():.1f}", fontsize=10)

    ax = fig.add_subplot(grid[0, 3])
    for c in np.argsort(-train.per_class, kind="stable"):  # most common first, rare classes on top
        name = names[c]
        mine = train.box_classes == c
        if mine.any():
            ax.scatter(train.sizes[mine, 0], train.sizes[mine, 1], s=3, alpha=0.3,
                       color=_hex(class_color(c, name)), label=name)
    ax.set_xlabel("Box width (fraction of the image)")
    ax.set_ylabel("Box height")
    ax.set_title("Box sizes: PC small, WBC large", fontsize=10)
    ax.legend(fontsize=8, markerscale=4)

    for i, path in enumerate(samples[:4]):
        ax = fig.add_subplot(grid[1, i])
        with Image.open(path) as handle:
            image = to_rgb(handle)
        gt = ground_truth(path, image.width, image.height) or Detections.empty()
        ax.imshow(np.asarray(image))
        _boxes_on_axes(ax, gt, dict(enumerate(names)), linewidth=1)
        ax.set_title(f"{path.name[:28]}…  ({len(gt)} boxes)", fontsize=8)
        ax.axis("off")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.04)
    return figure_to_image(fig)


def render_training_history(history: Mapping[str, Sequence[float]], title: str = "") -> Image.Image:
    """Losses and validation metrics per epoch from ``results.csv``."""
    epochs = np.asarray(history.get("epoch", []), dtype=float)
    fig = Figure(figsize=(14, 4.4), dpi=100)
    if title:
        fig.suptitle(title, fontsize=11)
    panels = (
        ("Box regression loss", ("train/box_loss", "val/box_loss")),
        ("Classification loss", ("train/cls_loss", "val/cls_loss")),
        ("Distribution focal loss", ("train/dfl_loss", "val/dfl_loss")),
        ("Validation metrics", ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")),
    )
    for i, (name, keys) in enumerate(panels):
        ax = fig.add_subplot(1, 4, i + 1)
        for key in keys:
            values = history.get(key)
            if values is not None and len(values) == len(epochs):
                ax.plot(epochs, values, marker="." if len(epochs) < 30 else None, label=key.split("/")[-1].replace("(B)", "")
                        if "metrics" in key else key.split("/")[0])
        ax.set_xlabel("Epoch")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        else:  # e.g. YOLO26 trains without the distribution focal loss
            ax.text(0.5, 0.5, "not used by this model", ha="center", va="center", transform=ax.transAxes, color="#777")
        if "metrics" in keys[0]:
            ax.set_ylim(0, 1)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.84 if title else 0.9, bottom=0.13, wspace=0.28)
    return figure_to_image(fig)


def render_evaluation(result: EvaluationResult, title: str = "") -> Image.Image:
    """PR curves, AP against the IoU threshold, miss rate against FPPI and confusion matrix."""
    fig = Figure(figsize=(16, 4.6), dpi=100)
    if title:
        fig.suptitle(title, fontsize=11)
    names = result.class_names
    first = result.iou_thresholds[0]

    ax = fig.add_subplot(1, 4, 1)
    ap_first = result.ap[:, 0]
    for c, curve in sorted(result.curves.items()):
        color = _hex(class_color(c, names[c]))
        # draw the interpolated envelope, the curve the AP integrates
        envelope = np.maximum.accumulate(curve.precision[::-1])[::-1]
        ax.plot(curve.recall, envelope, color=color, label=f"{names[c]}  AP={ap_first[c]:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall at IoU {first:.2f}  (mAP {result.map_at(first):.3f})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = fig.add_subplot(1, 4, 2)
    thresholds = np.asarray(result.iou_thresholds)
    for c, name in enumerate(names):
        if not np.all(np.isnan(result.ap[c])):
            ax.plot(thresholds, result.ap[c], marker="o", markersize=3, color=_hex(class_color(c, name)), label=name)
    ax.plot(thresholds, result.map_per_threshold(), "k--", label=f"mean (mAP@[.5:.95] = {result.map:.3f})")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1.02)
    ax.set_title("Stricter localisation → lower AP", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    ax = fig.add_subplot(1, 4, 3)
    if result.fppi.size:
        mask = result.fppi > 0
        ax.plot(result.fppi[mask], result.miss_rate[mask], color="#444")
        ax.set_xscale("log")
    ax.set_xlabel("False positives per image (FPPI)")
    ax.set_ylabel("Miss rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Miss rate vs FPPI (all classes, IoU {first:.2f})", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(1, 4, 4)
    if result.confusion is not None:
        labels = [*names, "background"]
        matrix = result.confusion.astype(float)
        normalised = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1)
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        for (row, col), value in np.ndenumerate(result.confusion):
            if row == col == len(names):
                continue
            ax.text(col, row, str(value), ha="center", va="center", fontsize=8,
                    color="white" if normalised[row, col] > 0.5 else "black")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix (conf ≥ 0.25, IoU {first:.2f})", fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.84 if title else 0.9, bottom=0.17, wspace=0.32)
    return figure_to_image(fig)
