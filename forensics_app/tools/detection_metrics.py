"""Set 9: the maths of object detection in plain NumPy (no YOLO needed).

Everything the theory class uses to validate a detector is implemented here so it can
be tested and explained without a network:

* box formats: COCO ``[x, y, w, h]`` (top-left corner), YOLO ``class xc yc w h``
  (centre, normalised to [0, 1]) and the ``x1 y1 x2 y2`` corners used to compute;
* Intersection over Union (IoU) between every pair of boxes;
* greedy non-maximum suppression (NMS), for each class independently;
* matching of detections with the ground truth at an IoU threshold (each real object
  can be found only once: a second detection of the same cell is a false positive);
* the precision-recall curve, average precision (AP, all-point interpolation, as in
  PASCAL VOC 2010+/COCO) and the mAP: "the average precision of each class across all
  the IoU thresholds (0.50, 0.55, ..., 0.95), then averaged over the classes";
* the miss rate against false positives per image (FPPI), the curve used for
  pedestrian detection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import warnings

import numpy as np

# The COCO metric: 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05
COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


# --------------------------------------------------------------------------- formats


def _as_boxes(boxes: np.ndarray | Sequence) -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    array = array.reshape(-1, 4)
    return array


def xywh_to_xyxy(boxes: np.ndarray | Sequence) -> np.ndarray:
    """COCO ``[x, y, w, h]`` (top-left corner and size) to corners ``[x1, y1, x2, y2]``."""
    boxes = _as_boxes(boxes)
    return np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]])


def cxcywh_to_xyxy(boxes: np.ndarray | Sequence) -> np.ndarray:
    """Centre and size ``[xc, yc, w, h]`` to corners ``[x1, y1, x2, y2]``."""
    boxes = _as_boxes(boxes)
    half_w, half_h = boxes[:, 2] / 2, boxes[:, 3] / 2
    return np.column_stack([boxes[:, 0] - half_w, boxes[:, 1] - half_h, boxes[:, 0] + half_w, boxes[:, 1] + half_h])


def xyxy_to_cxcywh(boxes: np.ndarray | Sequence) -> np.ndarray:
    boxes = _as_boxes(boxes)
    return np.column_stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]
    )


def coco_to_yolo_box(bbox: Sequence[float], width: float, height: float) -> tuple[float, float, float, float]:
    """One COCO box in pixels to the YOLO ``xc yc w h`` normalised by the image size.

    The box is first clipped to the image, because some annotation tools export boxes
    that stick out by a pixel and YOLO expects every value in [0, 1].
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"The image size must be positive, got {width} x {height}.")
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = min(max(x, 0.0), width), min(max(y, 0.0), height)
    x2, y2 = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def yolo_to_xyxy(boxes: np.ndarray | Sequence, width: float, height: float) -> np.ndarray:
    """Normalised YOLO ``xc yc w h`` to pixel corners, for an image of ``width x height``."""
    corners = cxcywh_to_xyxy(boxes)
    return corners * np.array([width, height, width, height], dtype=np.float64)


# --------------------------------------------------------------------------- IoU and NMS


def box_area(boxes: np.ndarray | Sequence) -> np.ndarray:
    boxes = _as_boxes(boxes)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def box_iou(boxes_a: np.ndarray | Sequence, boxes_b: np.ndarray | Sequence) -> np.ndarray:
    """IoU of every box in ``boxes_a`` with every box in ``boxes_b`` (corners), shape (n, m).

    IoU = area of the intersection / area of the union. Two boxes that do not touch have
    IoU 0; a box with itself has IoU 1.
    """
    a, b = _as_boxes(boxes_a), _as_boxes(boxes_b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    sides = np.clip(bottom_right - top_left, 0, None)
    intersection = sides[..., 0] * sides[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def nms(boxes: np.ndarray | Sequence, scores: np.ndarray | Sequence, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns the indices kept, best score first.

    Take the box with the highest score, keep it, and remove every remaining box that
    overlaps it with IoU > ``iou_threshold``; repeat with the boxes left. A lower
    threshold removes more boxes (also neighbours that are different objects).
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"The IoU threshold must be between 0 and 1, got {iou_threshold}.")
    boxes = _as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError(f"{len(boxes)} boxes but {len(scores)} scores.")
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        rest = order[1:]
        if rest.size == 0:
            break
        overlaps = box_iou(boxes[best:best + 1], boxes[rest])[0]
        order = rest[overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray | Sequence,
    scores: np.ndarray | Sequence,
    classes: np.ndarray | Sequence,
    iou_threshold: float,
    class_agnostic: bool = False,
) -> np.ndarray:
    """NMS for each class independently (as R-CNN does), or across classes if ``class_agnostic``."""
    boxes = _as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    if class_agnostic:
        return nms(boxes, scores, iou_threshold)
    keep: list[np.ndarray] = []
    for cls in np.unique(classes):
        index = np.flatnonzero(classes == cls)
        keep.append(index[nms(boxes[index], scores[index], iou_threshold)])
    if not keep:
        return np.zeros(0, dtype=np.int64)
    merged = np.concatenate(keep)
    return merged[np.argsort(-scores[merged], kind="stable")]


# --------------------------------------------------------------------------- detections


@dataclass
class Detections:
    """Boxes of one image in pixel corners, with their class and (for predictions) score."""

    boxes: np.ndarray  # (n, 4) x1 y1 x2 y2
    classes: np.ndarray  # (n,) int
    scores: np.ndarray | None = None  # (n,) float, None for the ground truth

    def __post_init__(self) -> None:
        self.boxes = _as_boxes(self.boxes)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        if self.scores is not None:
            self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
            if len(self.scores) != len(self.boxes):
                raise ValueError("Each box needs exactly one score.")
        if len(self.classes) != len(self.boxes):
            raise ValueError("Each box needs exactly one class.")

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def empty(cls, with_scores: bool = False) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0, dtype=np.int64), np.zeros(0) if with_scores else None)

    def subset(self, index: np.ndarray) -> "Detections":
        return Detections(self.boxes[index], self.classes[index], None if self.scores is None else self.scores[index])

    def counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(self.classes, minlength=num_classes)[:num_classes] if len(self) else np.zeros(num_classes, int)


def match_detections(pred: Detections, gt: Detections, iou_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Decide which predictions are true positives at ``iou_threshold``.

    Predictions are visited from the highest score down. Each one is matched with the
    ground-truth box of the same class that it overlaps most among those still free; it
    is a true positive if that IoU >= ``iou_threshold``. Otherwise (no box left, wrong
    place, or the object was already found by a better detection) it is a false positive.

    Returns ``(is_tp, matched_gt)``: a boolean per prediction and, for each prediction,
    the index of its ground-truth box or -1.
    """
    if pred.scores is None:
        raise ValueError("The predictions need scores.")
    is_tp = np.zeros(len(pred), dtype=bool)
    matched_gt = np.full(len(pred), -1, dtype=np.int64)
    if len(pred) == 0 or len(gt) == 0:
        return is_tp, matched_gt
    iou = box_iou(pred.boxes, gt.boxes)
    iou[pred.classes[:, None] != gt.classes[None, :]] = 0.0  # a box can only find its own class
    used = np.zeros(len(gt), dtype=bool)
    for p in np.argsort(-pred.scores, kind="stable"):
        candidates = np.where(used, -1.0, iou[p])
        g = int(np.argmax(candidates))
        if candidates[g] >= iou_threshold and candidates[g] > 0:
            is_tp[p] = True
            matched_gt[p] = g
            used[g] = True
    return is_tp, matched_gt


# --------------------------------------------------------------------------- AP and mAP


def precision_recall_curve(scores: np.ndarray, is_tp: np.ndarray, num_gt: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall after accepting the detections above each confidence level.

    Detections are sorted by score; after the k first ones, precision = TP / k and
    recall = TP / number of real objects. Returns ``(precision, recall, scores_sorted)``.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    is_tp = np.asarray(is_tp, dtype=bool).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order])
    fp = np.cumsum(~is_tp[order])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / num_gt if num_gt > 0 else np.zeros_like(tp, dtype=np.float64)
    return precision, recall, scores[order]


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under the precision-recall curve with all-point interpolation.

    The curve is first made monotone (the precision at recall r is the best precision
    at any recall >= r: the "envelope"), then integrated where recall changes. A
    detector that finds every object before any mistake gets AP = 1.
    """
    if len(recall) == 0:
        return 0.0
    r = np.concatenate([[0.0], np.asarray(recall, dtype=np.float64), [1.0]])
    p = np.concatenate([[1.0], np.asarray(precision, dtype=np.float64), [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]  # envelope
    steps = np.flatnonzero(r[1:] != r[:-1])
    return float(np.sum((r[steps + 1] - r[steps]) * p[steps + 1]))


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    """Mean ignoring NaN; an all-NaN row (class without ground truth) stays NaN, silently."""
    if values.size == 0:
        return np.zeros(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


@dataclass
class ClassCurve:
    precision: np.ndarray
    recall: np.ndarray
    scores: np.ndarray


@dataclass
class EvaluationResult:
    """Everything the evaluation figure and the Results panel need."""

    class_names: list[str]
    iou_thresholds: tuple[float, ...]
    ap: np.ndarray  # (classes, thresholds); NaN for a class without ground truth
    num_gt: np.ndarray  # (classes,)
    num_pred: np.ndarray  # (classes,)
    num_images: int
    curves: dict[int, ClassCurve] = field(default_factory=dict)  # PR curve per class at the first threshold
    fppi: np.ndarray = field(default_factory=lambda: np.zeros(0))  # all classes, first threshold
    miss_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    confusion: np.ndarray | None = None  # (classes + 1, classes + 1), last row/column = background

    def ap_at(self, threshold: float) -> np.ndarray:
        """AP of every class at one IoU threshold (it must be one of the evaluated ones)."""
        matches = [i for i, t in enumerate(self.iou_thresholds) if abs(t - threshold) < 1e-9]
        if not matches:
            raise KeyError(f"IoU {threshold} was not evaluated.")
        return self.ap[:, matches[0]]

    @property
    def ap_per_class(self) -> np.ndarray:
        """AP averaged over all the IoU thresholds, per class (COCO AP@[.5:.95])."""
        return _nanmean(self.ap, axis=1)

    @property
    def map(self) -> float:
        """The mAP: average over classes (with ground truth) of the per-class AP over thresholds."""
        values = self.ap_per_class
        values = values[~np.isnan(values)]
        return float(values.mean()) if values.size else 0.0

    def map_at(self, threshold: float) -> float:
        values = self.ap_at(threshold)
        values = values[~np.isnan(values)]
        return float(values.mean()) if values.size else 0.0

    def map_per_threshold(self) -> np.ndarray:
        return _nanmean(self.ap, axis=0)


def confusion_matrix(
    preds: Sequence[Detections], gts: Sequence[Detections], num_classes: int, iou_threshold: float = 0.5
) -> np.ndarray:
    """Detection confusion matrix (rows = true class, columns = predicted class).

    The extra last row/column is "background": a prediction that overlaps no object
    is counted in the background row (false positive), and an object that no
    prediction overlaps in the background column (missed). Matching ignores the class,
    so a white cell detected as a red cell shows up off the diagonal.
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    background = num_classes
    for pred, gt in zip(preds, gts):
        used_gt = np.zeros(len(gt), dtype=bool)
        if len(pred) and len(gt):
            iou = box_iou(pred.boxes, gt.boxes)
            order = np.argsort(-pred.scores, kind="stable") if pred.scores is not None else np.arange(len(pred))
            for p in order:
                candidates = np.where(used_gt, -1.0, iou[p])
                g = int(np.argmax(candidates))
                if candidates[g] >= iou_threshold and candidates[g] > 0:
                    used_gt[g] = True
                    matrix[gt.classes[g], pred.classes[p]] += 1
                else:
                    matrix[background, pred.classes[p]] += 1
        else:
            for c in pred.classes:
                matrix[background, c] += 1
        for g in np.flatnonzero(~used_gt):
            matrix[gt.classes[g], background] += 1
    return matrix


def evaluate_detections(
    preds: Sequence[Detections],
    gts: Sequence[Detections],
    class_names: Sequence[str],
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
    confusion_confidence: float = 0.25,
) -> EvaluationResult:
    """AP of every class at every IoU threshold over a set of images.

    ``preds[i]`` and ``gts[i]`` belong to the same image. For each class and threshold the
    detections of all the images are pooled, sorted by score, and their PR curve gives
    one AP. The PR curves and the FPPI curve are kept for the first threshold.

    The confusion matrix only counts predictions with score >= ``confusion_confidence``
    (the AP needs every detection, but a confusion matrix of the 0.001-confidence
    leftovers would only show noise).
    """
    if len(preds) != len(gts):
        raise ValueError(f"{len(preds)} predictions for {len(gts)} images.")
    if not iou_thresholds:
        raise ValueError("At least one IoU threshold is needed.")
    num_classes = len(class_names)
    thresholds = tuple(float(t) for t in iou_thresholds)
    ap = np.full((num_classes, len(thresholds)), np.nan)
    num_gt = np.zeros(num_classes, dtype=np.int64)
    num_pred = np.zeros(num_classes, dtype=np.int64)
    for gt in gts:
        num_gt += gt.counts(num_classes)
    for pred in preds:
        num_pred += pred.counts(num_classes)

    curves: dict[int, ClassCurve] = {}
    fppi = miss_rate = np.zeros(0)
    for t_index, threshold in enumerate(thresholds):
        all_scores: list[np.ndarray] = []
        all_tp: list[np.ndarray] = []
        all_cls: list[np.ndarray] = []
        for pred, gt in zip(preds, gts):
            is_tp, _ = match_detections(pred, gt, threshold)
            all_scores.append(pred.scores)
            all_tp.append(is_tp)
            all_cls.append(pred.classes)
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        tps = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)
        classes = np.concatenate(all_cls) if all_cls else np.zeros(0, dtype=np.int64)
        for c in range(num_classes):
            if num_gt[c] == 0:
                continue
            mine = classes == c
            precision, recall, sorted_scores = precision_recall_curve(scores[mine], tps[mine], int(num_gt[c]))
            ap[c, t_index] = average_precision(precision, recall)
            if t_index == 0:
                curves[c] = ClassCurve(precision, recall, sorted_scores)
        if t_index == 0:
            order = np.argsort(-scores, kind="stable")
            total_gt = max(int(num_gt.sum()), 1)
            fppi = np.cumsum(~tps[order]) / max(len(gts), 1)
            miss_rate = 1.0 - np.cumsum(tps[order]) / total_gt

    return EvaluationResult(
        class_names=list(class_names),
        iou_thresholds=thresholds,
        ap=ap,
        num_gt=num_gt,
        num_pred=num_pred,
        num_images=len(gts),
        curves=curves,
        fppi=fppi,
        miss_rate=miss_rate,
        confusion=confusion_matrix(
            [p.subset(np.flatnonzero(p.scores >= confusion_confidence)) for p in preds], gts, num_classes, thresholds[0]
        ),
    )


@dataclass(frozen=True)
class ImageMatch:
    """How the detections of one image compare with its ground truth."""

    true_positives: int
    false_positives: int
    false_negatives: int
    pred_is_tp: np.ndarray
    gt_found: np.ndarray

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    @property
    def recall(self) -> float:
        real = self.true_positives + self.false_negatives
        return self.true_positives / real if real else 0.0


def compare_image(pred: Detections, gt: Detections, iou_threshold: float = 0.5) -> ImageMatch:
    """TP / FP / FN of a single image at ``iou_threshold`` (same class required)."""
    is_tp, matched = match_detections(pred, gt, iou_threshold)
    found = np.zeros(len(gt), dtype=bool)
    found[matched[matched >= 0]] = True
    tp = int(is_tp.sum())
    return ImageMatch(tp, int(len(pred) - tp), int(len(gt) - tp), is_tp, found)
