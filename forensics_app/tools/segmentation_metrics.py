"""Set 11: metrics for semantic segmentation (NumPy only, no PyTorch).

A segmentation is a class index per pixel. Two masks are compared class by class:

* **Dice** = 2 |P ∩ G| / (|P| + |G|): the overlap counted twice over the total size.
* **IoU** (Jaccard) = |P ∩ G| / |P ∪ G|.

Both are 1 for a perfect match and 0 when the masks do not overlap. They are linked:
with I = |P ∩ G| and S = |P| + |G|, the union is S − I, so

    Dice = 2I / S,   IoU = I / (S − I)   =>   IoU = Dice / (2 − Dice),   Dice = 2 IoU / (1 + IoU)

(Exercise 4 of the notebook). Dice ≥ IoU always, and they are equal only at 0 and 1.

Unlike pixel accuracy, neither metric is fooled by a large background: predicting
"no food" everywhere is ~89 % accurate on the trays but has a food Dice of 0.

When a class is absent from both masks the ratio is 0/0. The notebook adds a tiny eps,
so that case counts as a perfect 1; ``empty`` chooses that value (1.0 as in the
notebook, or NaN to leave the image out of that class's average).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import warnings

import numpy as np


def iou_from_dice(dice: np.ndarray | float) -> np.ndarray | float:
    """IoU = Dice / (2 − Dice)."""
    return np.asarray(dice) / (2.0 - np.asarray(dice))


def dice_from_iou(iou: np.ndarray | float) -> np.ndarray | float:
    """Dice = 2 IoU / (1 + IoU)."""
    return 2.0 * np.asarray(iou) / (1.0 + np.asarray(iou))


def overlap_counts(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per class: intersection |P ∩ G|, predicted pixels |P| and true pixels |G|.

    ``pred`` and ``gt`` are integer masks of the same shape (any number of images).
    """
    pred = np.asarray(pred).ravel().astype(np.int64)
    gt = np.asarray(gt).ravel().astype(np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"The masks have different sizes: {pred.shape} and {gt.shape}.")
    predicted = np.bincount(pred, minlength=num_classes)[:num_classes]
    true = np.bincount(gt, minlength=num_classes)[:num_classes]
    inter = np.bincount(gt[pred == gt], minlength=num_classes)[:num_classes]
    return inter, predicted, true


def dice_scores(pred: np.ndarray, gt: np.ndarray, num_classes: int, empty: float = np.nan) -> np.ndarray:
    """Dice of every class between two masks; ``empty`` where the class is in neither."""
    inter, predicted, true = overlap_counts(pred, gt, num_classes)
    total = predicted + true
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2.0 * inter / total
    return np.where(total == 0, empty, dice)


def iou_scores(pred: np.ndarray, gt: np.ndarray, num_classes: int, empty: float = np.nan) -> np.ndarray:
    """IoU of every class between two masks; ``empty`` where the class is in neither."""
    inter, predicted, true = overlap_counts(pred, gt, num_classes)
    union = predicted + true - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = inter / union
    return np.where(union == 0, empty, iou)


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """Pixel counts: row = true class, column = predicted class."""
    pred = np.asarray(pred).ravel().astype(np.int64)
    gt = np.asarray(gt).ravel().astype(np.int64)
    return np.bincount(gt * num_classes + pred, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice of the foreground (value 1) exactly as in the notebook: 1 when both masks are empty."""
    return float(dice_scores(np.asarray(pred) > 0, np.asarray(gt) > 0, 2, empty=1.0)[1])


def threshold_masks(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Binary task: the probability of 'food' becomes a 0/1 mask (strictly above the threshold)."""
    return (np.asarray(probabilities) > threshold).astype(np.uint8)


def threshold_sweep(probabilities: np.ndarray, gts: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Mean per-image Dice for each threshold (Exercise 1). ``probabilities``: (n, H, W) in [0, 1]."""
    out = np.zeros(len(thresholds))
    for k, t in enumerate(thresholds):
        out[k] = np.mean([binary_dice(threshold_masks(p, t), g) for p, g in zip(probabilities, gts)])
    return out


@dataclass
class SplitEvaluation:
    """Metrics of a model on one split (validation or test)."""

    names: list[str]
    class_names: list[str]
    # per image: the main Dice (food for the binary task, mean over the classes present otherwise)
    dice: np.ndarray
    iou: np.ndarray
    # per class (rows = images): NaN where the class is in neither mask
    class_dice_per_image: np.ndarray
    class_iou_per_image: np.ndarray
    confusion: np.ndarray
    threshold: float | None = None
    sweep_thresholds: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sweep_dice: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def class_dice(self) -> np.ndarray:
        """Mean Dice of each class over the images where it appears (in the truth or the prediction)."""
        return _nanmean(self.class_dice_per_image, axis=0)

    @property
    def class_iou(self) -> np.ndarray:
        return _nanmean(self.class_iou_per_image, axis=0)

    @property
    def pixel_accuracy(self) -> float:
        total = self.confusion.sum()
        return float(np.trace(self.confusion) / total) if total else 0.0

    @property
    def mean_dice(self) -> float:
        return float(np.mean(self.dice)) if self.dice.size else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.iou)) if self.iou.size else 0.0

    def worst(self, n: int = 3) -> np.ndarray:
        """Indices of the ``n`` images with the lowest Dice (Exercise 5)."""
        return np.argsort(self.dice, kind="stable")[:n]

    def to_dict(self) -> dict:
        return {
            "images": len(self.names),
            "dice_mean": round(self.mean_dice, 5),
            "dice_std": round(float(np.std(self.dice)), 5) if self.dice.size else 0.0,
            "iou_mean": round(self.mean_iou, 5),
            "pixel_accuracy": round(self.pixel_accuracy, 5),
            "threshold": self.threshold,
            "class_dice": [_round_or_none(v) for v in self.class_dice],
            "class_iou": [_round_or_none(v) for v in self.class_iou],
            "per_image": {
                "names": list(self.names),
                "dice": [round(float(v), 5) for v in self.dice],
                "iou": [round(float(v), 5) for v in self.iou],
            },
            "confusion": self.confusion.astype(int).tolist(),
            "threshold_sweep": {
                "thresholds": [round(float(t), 3) for t in self.sweep_thresholds],
                "dice": [round(float(d), 5) for d in self.sweep_dice],
            },
        }


def _round_or_none(value: float) -> float | None:
    return None if np.isnan(value) else round(float(value), 5)


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.full(values.shape[1 - axis] if values.ndim == 2 else 0, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "mean of empty slice" for a class never seen
        return np.nanmean(values, axis=axis)


def evaluate_masks(
    preds: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    class_names: Sequence[str],
    names: Sequence[str] | None = None,
    threshold: float | None = None,
) -> SplitEvaluation:
    """Compare predicted and true masks image by image.

    Binary task (2 classes): the main Dice/IoU of an image is the one of class 1 (food),
    with 1.0 when neither mask has food, exactly as the notebook. Multi-class: it is the
    mean over the foreground classes (all but 0) that appear in the truth or the
    prediction of that image.
    """
    k = len(class_names)
    n = len(preds)
    class_dice = np.full((n, k), np.nan)
    class_iou = np.full((n, k), np.nan)
    main_dice = np.zeros(n)
    main_iou = np.zeros(n)
    confusion = np.zeros((k, k), dtype=np.int64)
    for i, (p, g) in enumerate(zip(preds, gts)):
        class_dice[i] = dice_scores(p, g, k)
        class_iou[i] = iou_scores(p, g, k)
        confusion += confusion_matrix(p, g, k)
        if k == 2:
            main_dice[i] = dice_scores(p, g, 2, empty=1.0)[1]
            main_iou[i] = iou_scores(p, g, 2, empty=1.0)[1]
        else:
            foreground_d = class_dice[i, 1:]
            foreground_i = class_iou[i, 1:]
            present = ~np.isnan(foreground_d)
            main_dice[i] = float(foreground_d[present].mean()) if present.any() else 1.0
            main_iou[i] = float(foreground_i[present].mean()) if present.any() else 1.0
    return SplitEvaluation(
        names=list(names) if names is not None else [str(i) for i in range(n)],
        class_names=list(class_names),
        dice=main_dice,
        iou=main_iou,
        class_dice_per_image=class_dice,
        class_iou_per_image=class_iou,
        confusion=confusion,
        threshold=threshold,
    )


def split_from_dict(data: dict, class_names: Sequence[str]) -> dict:
    """The saved metrics of one split, with arrays (for the passport figures)."""
    per_image = data.get("per_image", {})
    sweep = data.get("threshold_sweep", {})

    def arr(values) -> np.ndarray:
        return np.array([np.nan if v is None else v for v in (values or [])], dtype=float)

    return {
        "names": per_image.get("names", []),
        "dice": arr(per_image.get("dice")),
        "iou": arr(per_image.get("iou")),
        "class_dice": arr(data.get("class_dice")),
        "class_iou": arr(data.get("class_iou")),
        "confusion": np.array(data.get("confusion") or np.zeros((len(class_names),) * 2), dtype=float),
        "sweep_thresholds": arr(sweep.get("thresholds")),
        "sweep_dice": arr(sweep.get("dice")),
        "dice_mean": data.get("dice_mean"),
        "iou_mean": data.get("iou_mean"),
        "pixel_accuracy": data.get("pixel_accuracy"),
    }
