"""Set 9 extra: evaluate a trained YOLO on a whole split with our own metrics.

The course notebook computes a single, class-agnostic table and reuses the matches
made at IoU 0.5 for every threshold, so its "AP at IoU 0.95" is just the AP at 0.5.
Here every IoU threshold matches the detections again and each class gets its own AP,
exactly as described in the theory: AP per class over the IoU thresholds 0.50:0.95,
then averaged over the classes. The numbers can be compared with ``model.val``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from PIL import Image

from .detection_metrics import COCO_IOU_THRESHOLDS, Detections, EvaluationResult, evaluate_detections
from .dialogs import TaskCancelled
from .yolo_dataset import ground_truth, list_images, read_data_yaml, split_images_dir
from .yolo_model import predict, default_device, to_rgb

Report = Callable[[int, int, str], None]

# Like model.val: keep almost every detection so the PR curve reaches high recall
EVAL_CONFIDENCE = 0.001
EVAL_NMS_IOU = 0.7


@dataclass
class SplitEvaluation:
    result: EvaluationResult
    split: str
    images: int
    image_paths: list[Path]
    preds: list[Detections]
    gts: list[Detections]
    seconds: float
    device: str


def evaluate_split(
    model: Any,
    data_yaml: Path,
    split: str = "test",
    report: Report = lambda d, t, s: None,
    cancelled: Callable[[], bool] = lambda: False,
    conf: float = EVAL_CONFIDENCE,
    iou: float = EVAL_NMS_IOU,
    imgsz: int = 640,
    device: str | None = None,
    limit: int | None = None,
) -> SplitEvaluation:
    """Predict every image of ``split`` and compare with its label file."""
    data = read_data_yaml(data_yaml)
    names = data["names"]
    if not names:
        raise ValueError(f"{data_yaml} does not list the class names.")
    model_names = [str(v) for _, v in sorted(getattr(model, "names", {}).items())]
    if model_names and model_names != names:
        raise ValueError(
            f"The model detects {model_names} but the dataset has {names}. "
            "Choose the model trained on this dataset."
        )
    folder = split_images_dir(data, data_yaml, split)
    images = list_images(folder)
    if limit:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"There are no images in {folder}.")
    device = device or default_device()
    preds, gts, kept = [], [], []
    start = time.perf_counter()
    for index, path in enumerate(images):
        if cancelled():
            raise TaskCancelled()
        with Image.open(path) as handle:
            image = to_rgb(handle)
        gt = ground_truth(path, image.width, image.height)
        if gt is None:  # no label file: cannot be evaluated
            continue
        preds.append(predict(model, image, conf=conf, iou=iou, imgsz=imgsz, device=device).detections)
        gts.append(gt)
        kept.append(path)
        report(index + 1, len(images) + 1, f"Predicting {split} image {index + 1}/{len(images)}")
    if not kept:
        raise FileNotFoundError(f"None of the images in {folder} has a label file in ../labels.")
    report(len(images), len(images) + 1, "Computing AP at 10 IoU thresholds…")
    result = evaluate_detections(preds, gts, names, COCO_IOU_THRESHOLDS)
    return SplitEvaluation(result, split, len(kept), kept, preds, gts, time.perf_counter() - start, device)
