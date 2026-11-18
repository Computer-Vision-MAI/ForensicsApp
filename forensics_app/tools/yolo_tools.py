"""Set 9 tools: object detection with YOLO (Ultralytics).

* 9.1  ``YoloDetectTool``      pretrained COCO model on the working image (+ our own NMS)
* 9.2  ``YoloDatasetTool``     COCO -> YOLO conversion and dataset exploration
       ``YoloTrainTool``       train / fine-tune on the blood cells, with live progress
* 9.3  ``YoloBloodCellTool``   trained model on an unseen image: ground truth vs prediction
* extra ``YoloEvaluateTool``   our own mAP, PR curves, AP vs IoU, FPPI and confusion matrix

The tools only handle dialogs and packaging; the work is in ``yolo_model``,
``yolo_dataset``, ``yolo_eval``, ``detection_metrics`` and ``yolo_render``. Everything slow
runs on a worker thread through ``run_with_progress``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import display_path
from . import yolo_model as ym
from .base import ForensicsTool, ToolResult
from .detection_metrics import Detections, compare_image, yolo_to_xyxy
from .dialogs import TaskCancelled, ask_choice, run_with_progress
from .yolo_dataset import (
    SPLITS,
    convert_coco_dataset,
    dataset_stats,
    label_path_for,
    list_images,
    read_data_yaml,
    read_yolo_labels,
    split_images_dir,
)
from .yolo_dialogs import Field, ask_form
from .yolo_eval import evaluate_split
from .yolo_render import (
    draw_detections,
    render_before_after_nms,
    render_dataset_overview,
    render_evaluation,
    render_gt_vs_pred,
    render_training_history,
)

CATEGORY = "Object Detection (YOLO)"
MAX_LISTED = 25  # detections listed one by one in the Results panel
BGR_NOTE = (
    "Ultralytics/OpenCV arrays are BGR; plt.imshow(results[0].plot()) shows them as RGB, which "
    "swaps red and blue ('why are people blue?'). Here the boxes are drawn on the RGB image."
)
PT_TYPES = (("PyTorch weights", "*.pt"), ("All files", "*.*"))

# Loaded models are kept, so running a tool twice does not reload the weights
_MODEL_CACHE: dict[tuple[str, float], Any] = {}
_CACHE_LOCK = threading.Lock()


def get_model(weights: str | Path) -> Any:
    path = Path(weights)
    resolved = ym.weights_path(path.name) if not path.is_absolute() and path.name == str(path) else path
    stamp = resolved.stat().st_mtime if resolved.is_file() else 0.0
    key = (str(resolved), stamp)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE.clear()  # one model in memory at a time
            _MODEL_CACHE[key] = ym.load_model(weights)
        return _MODEL_CACHE[key]


def _class_counts(detections: Detections, names: dict[int, str]) -> str:
    if not len(detections):
        return "nothing detected"
    counts = np.bincount(detections.classes, minlength=max(names, default=0) + 1)
    parts = [f"{counts[c]} {names.get(c, c)}" for c in np.argsort(-counts) if counts[c]]
    return ", ".join(parts)


def _detection_rows(detections: Detections, names: dict[int, str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    order = np.argsort(-detections.scores) if detections.scores is not None else np.arange(len(detections))
    for rank, i in enumerate(order[:MAX_LISTED], start=1):
        x1, y1, x2, y2 = detections.boxes[i]
        score = f" {detections.scores[i]:.2f}" if detections.scores is not None else ""
        rows[f"#{rank}"] = (
            f"{names.get(int(detections.classes[i]), detections.classes[i])}{score} "
            f"at ({x1:.0f}, {y1:.0f})-({x2:.0f}, {y2:.0f}), {x2 - x1:.0f}×{y2 - y1:.0f} px"
        )
    if len(detections) > MAX_LISTED:
        rows["…"] = f"{len(detections) - MAX_LISTED} more"
    return rows


def _timing(prediction: ym.Prediction) -> str:
    """The network time alone, and the total (the first run also loads the model on the device)."""
    device = ym.device_label(ym.default_device())
    total = f"{prediction.seconds * 1000:.0f} ms total"
    if prediction.speed:
        return f"network {prediction.speed.get('inference', 0):.0f} ms, {total} on {device}"
    return f"{total} on {device}"


def _require_ultralytics() -> None:
    ym.require_ultralytics()


# --------------------------------------------------------------------------- 9.1

DEDUP_OPTIONS = (
    "Ultralytics NMS (default)",
    "Our own greedy NMS, showing the boxes before / after it",
    "No NMS: YOLO26 one-to-one head (NMS-free)",
)


@dataclass(frozen=True)
class DetectSettings:
    weights: str | Path
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    own_nms: bool = False
    class_agnostic: bool = False
    preview_only: bool = True
    nms_free: bool = False


def detect(document: ImageDocument, settings: DetectSettings, model: Any) -> ToolResult:
    """Run the detector on the working image and package the result (no Tk)."""
    assert document.current is not None
    image = document.current
    prediction = ym.predict(
        model, image, conf=settings.conf, iou=settings.iou, imgsz=settings.imgsz,
        own_nms=settings.own_nms, class_agnostic=settings.class_agnostic, nms_free=settings.nms_free,
    )
    dets, names = prediction.detections, prediction.names
    if settings.own_nms and prediction.raw is not None:
        output = render_before_after_nms(image, prediction.raw, dets, names, settings.iou)
    else:
        output = draw_detections(image, dets, names)
    details: dict[str, object] = {
        "Operation": "YOLO object detection",
        "Model": Path(str(settings.weights)).name,
        "Classes known": len(names),
        "Detections": len(dets),
        "Per class": _class_counts(dets, names),
        "Confidence threshold": f"{settings.conf:.2f}",
        "NMS": prediction.nms if settings.nms_free else f"{prediction.nms}, IoU > {settings.iou:.2f} suppressed"
        + (" (class-agnostic)" if settings.class_agnostic else " (per class)"),
        "Input size": f"{image.width}×{image.height} → letterboxed to {settings.imgsz}",
        "Time": _timing(prediction),
    }
    if prediction.raw is not None:
        details["Candidates before NMS"] = len(prediction.raw)
        details["Removed by NMS"] = len(prediction.raw) - len(dets)
    details.update(_detection_rows(dets, names))
    details["Colours"] = BGR_NOTE
    message = f"YOLO found {len(dets)} objects: {_class_counts(dets, names)}."
    return ToolResult(message=message, image=output, details=details,
                      preview_only=settings.preview_only or settings.own_nms)


class YoloDetectTool(ForensicsTool):
    tool_id = "yolo_detect"
    title = "YOLO: Detect Objects (pretrained)"
    category = CATEGORY
    description = (
        "Set 9.1: runs a YOLO model pretrained on COCO (80 classes) on the working image and draws "
        "the boxes with class and confidence. Optionally applies our own greedy NMS."
    )

    def _model_options(self) -> list[tuple[str, str | None]]:
        options: list[tuple[str, str | None]] = [(label, name) for name, label in ym.PRETRAINED.items()]
        options += [(f"Blood cells: {p.name}", str(p)) for p in ym.blood_cell_models()]
        options.append(("Other .pt file (below)…", None))
        return options

    def ask(self, parent: tk.Misc) -> DetectSettings | None:
        options = self._model_options()
        fields = [
            Field("model", "Model", "choice", 0, [label for label, _ in options],
                  help="Pretrained weights are downloaded to models/yolo the first time (~5 MB)."),
            Field("file", "Other weights", "file", "", filetypes=PT_TYPES),
            Field("conf", "Confidence threshold", "float", 0.25, minimum=0, maximum=1,
                  help="Boxes with a lower score (objectness × class probability) are dropped."),
            Field("iou", "NMS IoU threshold", "float", 0.7, minimum=0, maximum=1,
                  help="Of two boxes that overlap more than this, only the best one is kept."),
            Field("imgsz", "Network input size", "int", 640, minimum=32, maximum=1920),
            Field("dedup", "Duplicate boxes", "choice", 0, DEDUP_OPTIONS),
            Field("agnostic", "Class-agnostic NMS (a bus box can suppress a person box)", "check", False),
            Field("preview", "Preview only (keep the working image and its history)", "check", True),
        ]

        def check(values: dict[str, Any]) -> str | None:
            if options[values["model"]][1] is None and not (values["file"] and values["file"].is_file()):
                return "Choose an existing .pt file for 'Other weights'."
            if values["imgsz"] % 32:
                return "The input size must be a multiple of 32 (YOLO downsamples by 32)."
            name = Path(str(options[values["model"]][1] or values["file"])).name
            if values["dedup"] == 2 and "yolo26" not in name:
                return "Only YOLO26 models have an NMS-free head: choose YOLO26n."
            return None

        values = ask_form(parent, self.title, fields, check=check)
        if values is None:
            return None
        weights = options[values["model"]][1] or values["file"]
        return DetectSettings(weights, values["conf"], values["iou"], values["imgsz"],
                              own_nms=values["dedup"] == 1, class_agnostic=values["agnostic"],
                              preview_only=values["preview"], nms_free=values["dedup"] == 2)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        _require_ultralytics()
        settings = self.ask(parent)
        if settings is None:
            return None

        def task(report, cancelled):
            report(0, 2, f"Loading {Path(str(settings.weights)).name}…")
            model = get_model(settings.weights)
            report(1, 2, f"Detecting on {ym.device_label(ym.default_device())}…")
            return detect(document, settings, model)

        try:
            return run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None


# --------------------------------------------------------------------------- 9.2 dataset


def _guess_coco_root() -> Path | None:
    """The lab materials are usually next to the repository; offer them as the start folder."""
    root = ym.PROJECT_ROOT.parent
    for candidate in sorted(root.glob("*/blood_cell*coco*")) + sorted(root.glob("blood_cell*coco*")):
        if (candidate / "train" / "_annotations.coco.json").is_file():
            return candidate
    return None


def explore_dataset(data_yaml: Path, seed: int = 0) -> ToolResult:
    data = read_data_yaml(data_yaml)
    names = data["names"]
    stats = []
    for split in SPLITS:
        key = "val" if split == "valid" else split
        if key in data:
            stats.append(dataset_stats(split_images_dir(data, data_yaml, key), len(names), split))
    if not stats or stats[0].images == 0:
        raise FileNotFoundError(f"No images found for the splits of {data_yaml}.")
    train_images = list_images(split_images_dir(data, data_yaml, "train"))
    samples = random.Random(seed).sample(train_images, min(4, len(train_images)))
    details: dict[str, object] = {"Operation": "YOLO dataset exploration", "Dataset": display_path(data_yaml),
                                  "Classes": ", ".join(f"{i} = {n}" for i, n in enumerate(names))}
    for s in stats:
        details[f"{s.split}: images"] = s.images
        details[f"{s.split}: boxes"] = f"{s.boxes} ({s.boxes / max(s.images, 1):.1f} per image)"
        details[f"{s.split}: per class"] = ", ".join(f"{n} {int(c)}" for n, c in zip(names, s.per_class))
        details[f"{s.split}: images containing"] = ", ".join(f"{n} {int(c)}" for n, c in zip(names, s.images_with_class))
        if s.background_images:
            details[f"{s.split}: images without boxes"] = s.background_images
    train = stats[0]
    share = train.per_class / max(train.per_class.sum(), 1)
    details["Imbalance"] = ", ".join(f"{n} {p:.1%}" for n, p in zip(names, share))
    for c, name in enumerate(names):
        mine = train.box_classes == c
        if mine.any():
            w, h = train.sizes[mine].mean(axis=0)
            details[f"Mean box {name}"] = f"{w:.3f} × {h:.3f} of the image"
    return ToolResult(
        message=f"{display_path(data_yaml)}: {sum(s.images for s in stats)} images, "
        f"{sum(s.boxes for s in stats)} boxes, classes {names}.",
        image=render_dataset_overview(stats, names, samples),
        details=details,
        preview_only=True,
    )


class YoloDatasetTool(ForensicsTool):
    tool_id = "yolo_dataset"
    title = "YOLO: COCO → YOLO Dataset"
    category = CATEGORY
    description = (
        "Set 9.2: converts the blood-cell dataset from COCO (one json per split) to the YOLO "
        "layout (images/, labels/*.txt, data.yaml) and shows its statistics."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        target = ym.BLOOD_CELL_DATASET
        yaml_path = target / "data.yaml"
        if yaml_path.is_file():
            choice = ask_choice(
                parent, self.title, f"A converted dataset already exists in {display_path(target)}.",
                ("Explore the existing dataset", "Convert again from a COCO folder"),
            )
            if choice is None:
                return None
            if choice == 0:
                return explore_dataset(yaml_path)
        start = _guess_coco_root()
        folder = filedialog.askdirectory(
            parent=parent, mustexist=True, initialdir=str(start) if start else None,
            title="COCO dataset folder (with train/, valid/, test/ and _annotations.coco.json)",
        )
        if not folder:
            return None
        if target.exists():
            if not messagebox.askyesno(self.title, f"Replace {display_path(target)} with the new conversion?",
                                       parent=parent):
                return None

        def task(report, cancelled):
            if target.exists():
                # only ever delete the dataset folder this tool created
                if target.resolve().parent != ym.DATASETS_DIR.resolve():
                    raise RuntimeError(f"Refusing to delete {target}.")
                shutil.rmtree(target)
            names, summaries, written = convert_coco_dataset(folder, target, report, cancelled)
            return names, summaries, written

        try:
            names, summaries, written = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
        result = explore_dataset(written)
        details: dict[str, object] = {"Operation": "COCO → YOLO conversion", "Source": folder,
                                      "Output": display_path(target), "Classes": ", ".join(names)}
        for s in summaries:
            details[f"Converted {s.split}"] = f"{s.images} images, {s.boxes} boxes" + (
                f", {s.skipped_boxes} empty boxes skipped" if s.skipped_boxes else "") + (
                f", {s.missing_images} images missing" if s.missing_images else "")
        details.update({k: v for k, v in result.details.items() if k not in ("Operation", "Classes")})
        return ToolResult(
            message=f"Converted {sum(s.images for s in summaries)} images to YOLO format in {display_path(target)}.",
            image=result.image, details=details, preview_only=True,
        )


# --------------------------------------------------------------------------- 9.2 training


START_OPTIONS = (
    ("Fine-tune yolo11n.pt (COCO pretrained, recommended)", "yolo11n.pt"),
    ("From scratch: yolo11n.yaml (random weights, as in the notebook)", "yolo11n.yaml"),
    ("Fine-tune yolo26n.pt (NMS-free YOLO26)", "yolo26n.pt"),
)


def training_result(outcome: ym.TrainOutcome, settings: ym.TrainSettings) -> ToolResult:
    history = outcome.history
    details: dict[str, object] = {
        "Operation": "YOLO training",
        "Started from": settings.start_from,
        "Epochs": f"{outcome.epochs_done} of {settings.epochs}" + (" (stopped early)" if outcome.stopped_early else ""),
        "Batch / image size": f"{settings.batch} / {settings.imgsz}",
        "Device": ym.device_label(settings.device or ym.default_device()),
        "Time": f"{outcome.seconds / 60:.1f} min",
    }
    map50 = history.get("metrics/mAP50(B)", [])
    map95 = history.get("metrics/mAP50-95(B)", [])
    if map95:
        best = int(np.argmax(map95))
        details["Best epoch (val mAP50-95)"] = best + 1
        details["Validation mAP@0.5"] = f"{map50[best]:.3f}"
        details["Validation mAP@[.5:.95]"] = f"{map95[best]:.3f}"
        details["Validation precision / recall"] = (
            f"{history.get('metrics/precision(B)', [0] * (best + 1))[best]:.3f} / "
            f"{history.get('metrics/recall(B)', [0] * (best + 1))[best]:.3f}"
        )
    details["Model saved as"] = display_path(outcome.installed_weights)
    details["Run folder (plots, results.csv)"] = display_path(outcome.run_dir)
    image = render_training_history(history, f"{settings.start_from} on {display_path(settings.data_yaml)}")
    return ToolResult(
        message=f"Training finished: {display_path(outcome.installed_weights)}"
        + (f", validation mAP50 {map50[int(np.argmax(map95))]:.3f}" if map95 else ""),
        image=image, details=details, preview_only=True,
    )


class YoloTrainTool(ForensicsTool):
    tool_id = "yolo_train"
    title = "YOLO: Train Blood-Cell Detector"
    category = CATEGORY
    description = (
        "Set 9.2: trains YOLO on the converted blood-cell dataset (PC, RBC, WBC) and saves the best "
        "weights in models/yolo. Shows the loss and mAP curves."
    )
    requires_image = False

    def ask(self, parent: tk.Misc) -> ym.TrainSettings | None:
        device = ym.default_device()
        fields = [
            Field("data", "Dataset (data.yaml)", "file", ym.BLOOD_CELL_DATASET / "data.yaml",
                  filetypes=(("YOLO dataset", "*.yaml"), ("All files", "*.*"))),
            Field("start", "Start from", "choice", 0, [label for label, _ in START_OPTIONS]),
            Field("epochs", "Epochs", "int", 50, minimum=1, maximum=1000),
            Field("batch", "Batch size", "int", 16, minimum=-1, maximum=256, help="-1 = automatic for the GPU memory."),
            Field("imgsz", "Image size", "int", 640, minimum=32, maximum=1920),
            Field("patience", "Early-stopping patience (epochs)", "int", 20, minimum=1, maximum=1000),
            Field("note", f"Device: {ym.device_label(device)}. "
                  + ("About 45 s per epoch on a laptop RTX GPU." if device != "cpu"
                     else "On CPU one epoch takes about 10 minutes: use few epochs or the GPU."), "note"),
        ]

        def check(values: dict[str, Any]) -> str | None:
            if values["data"] is None or not values["data"].is_file():
                return "The dataset does not exist yet: run 'YOLO: COCO → YOLO Dataset' first."
            if values["imgsz"] % 32:
                return "The image size must be a multiple of 32."
            if values["batch"] == 0:
                return "The batch size cannot be 0."
            return None

        values = ask_form(parent, self.title, fields, check=check)
        if values is None:
            return None
        start = START_OPTIONS[values["start"]][1]
        target = ym.installed_weights_for(start)
        if target.exists() and not messagebox.askyesno(
            self.title, f"{display_path(target)} already exists. Replace it with the new model when training ends?\n"
            "(The previous run stays in models/yolo/runs.)", parent=parent,
        ):
            return None
        return ym.TrainSettings(
            data_yaml=values["data"], epochs=values["epochs"], batch=values["batch"], imgsz=values["imgsz"],
            start_from=start, patience=values["patience"], device=device,
            name=f"blood_cell_{Path(start).stem}{'_scratch' if start.endswith('.yaml') else ''}_{values['epochs']}e",
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        _require_ultralytics()
        settings = self.ask(parent)
        if settings is None:
            return None
        # Cancel stops after the current batch; Ultralytics still validates and saves the
        # model trained so far, so it is shown (and installed) instead of being thrown away.
        finished: dict[str, ym.TrainOutcome] = {}

        def task(report, cancelled):
            finished["outcome"] = ym.train(settings, report, cancelled)
            return finished["outcome"]

        try:
            outcome = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            if "outcome" not in finished:
                return None
            outcome = finished["outcome"]
        return training_result(outcome, settings)


# --------------------------------------------------------------------------- 9.3


def find_ground_truth_label(image_path: Path | None) -> Path | None:
    """The label file of an image: ../labels/<name>.txt, or the same name in the converted dataset."""
    if image_path is None:
        return None
    label = label_path_for(image_path)
    if label.is_file():
        return label
    for split in SPLITS:
        candidate = ym.BLOOD_CELL_DATASET / split / "labels" / f"{Path(image_path).stem}.txt"
        if candidate.is_file():
            return candidate
    return None


def split_of(label: Path | None) -> str | None:
    if label is None:
        return None
    for part in reversed(label.parts[:-2]):
        if part in SPLITS:
            return part
    return None


@dataclass(frozen=True)
class CellSettings:
    weights: Path
    source: int = 0  # 0 = working image, 1 = random test image
    conf: float = 0.25
    iou: float = 0.7
    match_iou: float = 0.5
    seed: int | None = None


def detect_cells(image: Image.Image, image_path: Path | None, settings: CellSettings, model: Any) -> ToolResult:
    """Prediction on one image, compared with its ground truth when a label file exists (no Tk)."""
    prediction = ym.predict(model, image, conf=settings.conf, iou=settings.iou)
    pred, names = prediction.detections, prediction.names
    label = find_ground_truth_label(image_path)
    gt = None
    if label is not None:
        classes, boxes = read_yolo_labels(label)
        gt = Detections(yolo_to_xyxy(boxes, image.width, image.height), classes)
    match = compare_image(pred, gt, settings.match_iou) if gt is not None else None
    split = split_of(label)
    name = image_path.name if image_path is not None else "working image"
    title = f"{name}" + (f"  ({split} split{', seen during training' if split == 'train' else ', unseen'})" if split else "")
    output = render_gt_vs_pred(image, pred, gt, names, match, settings.match_iou, title)

    num_classes = max(len(names), int(pred.classes.max()) + 1 if len(pred) else 0)
    pred_counts = pred.counts(num_classes)
    details: dict[str, object] = {
        "Operation": "Blood-cell detection (YOLO)",
        "Image": name,
        "Model": display_path(Path(settings.weights)),
        "Thresholds": f"confidence {settings.conf:.2f}, NMS IoU {settings.iou:.2f}",
        "Time": _timing(prediction),
    }
    if gt is not None and match is not None:
        gt_counts = gt.counts(num_classes)
        details["Ground truth"] = display_path(label)
        if split:
            details["Split"] = split + (" (the model has seen this image!)" if split == "train" else " (unseen)")
        for c in range(num_classes):
            cls = names.get(c, str(c))
            details[f"Count {cls}"] = f"predicted {pred_counts[c]}, real {gt_counts[c]}"
        details[f"Matches at IoU ≥ {settings.match_iou:.2f}"] = (
            f"{match.true_positives} TP, {match.false_positives} FP, {match.false_negatives} missed"
        )
        details["Precision / recall"] = f"{match.precision:.3f} / {match.recall:.3f}"
        message = (f"{len(pred)} cells predicted, {len(gt)} annotated: {match.true_positives} correct, "
                   f"{match.false_positives} false, {match.false_negatives} missed.")
    else:
        for c in range(num_classes):
            details[f"Count {names.get(c, str(c))}"] = int(pred_counts[c])
        details["Ground truth"] = "no label file found (images/x.jpg → labels/x.txt)"
        message = f"{len(pred)} cells predicted: {_class_counts(pred, names)}."
    details.update(_detection_rows(pred, names))
    return ToolResult(message=message, image=output, details=details, preview_only=True)


class YoloBloodCellTool(ForensicsTool):
    tool_id = "yolo_blood_cells"
    title = "YOLO: Blood Cells (GT vs Prediction)"
    category = CATEGORY
    description = (
        "Set 9.3: loads the trained blood-cell model and tests it on the working image or a random "
        "unseen test image, drawing the ground truth next to the prediction and counting the cells."
    )
    requires_image = False

    def ask(self, parent: tk.Misc, has_image: bool) -> CellSettings | None:
        models = ym.blood_cell_models()
        options = [f"{p.name}" for p in models] + ["Other .pt file (below)…"]
        sources = (["Working image"] if has_image else []) + ["Random image from the test split (unseen)"]
        fields = [
            Field("model", "Model", "choice", 0, options),
            Field("file", "Other weights", "file", "", filetypes=PT_TYPES),
            Field("source", "Image", "choice", 0, sources),
            Field("conf", "Confidence threshold", "float", 0.25, minimum=0, maximum=1),
            Field("iou", "NMS IoU threshold", "float", 0.7, minimum=0, maximum=1),
            Field("match", "IoU to count a detection as correct", "float", 0.5, minimum=0.05, maximum=1),
        ]

        def check(values: dict[str, Any]) -> str | None:
            if values["model"] >= len(models) and not (values["file"] and values["file"].is_file()):
                return ("No blood-cell model selected. Train one with 'YOLO: Train Blood-Cell Detector' "
                        "or choose a .pt file.")
            return None

        values = ask_form(parent, self.title, fields, check=check)
        if values is None:
            return None
        weights = models[values["model"]] if values["model"] < len(models) else values["file"]
        source = values["source"] if has_image else 1
        return CellSettings(weights, source, values["conf"], values["iou"], values["match"])

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        _require_ultralytics()
        settings = self.ask(parent, document.current is not None)
        if settings is None:
            return None
        if settings.source == 0:
            assert document.current is not None
            image, path = document.current, document.path
        else:
            test_images = list_images(ym.BLOOD_CELL_DATASET / "test" / "images")
            if not test_images:
                raise FileNotFoundError("No test images: run 'YOLO: COCO → YOLO Dataset' first.")
            path = random.Random(settings.seed).choice(test_images)
            with Image.open(path) as handle:
                image = handle.convert("RGB")

        def task(report, cancelled):
            report(0, 2, f"Loading {Path(settings.weights).name}…")
            model = get_model(settings.weights)
            report(1, 2, f"Detecting cells in {path.name if path else 'the working image'}…")
            return detect_cells(image, path, settings, model)

        try:
            return run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None


# --------------------------------------------------------------------------- extra: evaluation


def evaluation_result(report, ultra: dict[str, Any] | None, weights: Path) -> ToolResult:
    result = report.result
    details: dict[str, object] = {
        "Operation": "Detection evaluation (our implementation)",
        "Model": display_path(weights),
        "Split": f"{report.split}: {report.images} images",
        "mAP@0.5": f"{result.map_at(0.5):.3f}",
        "mAP@0.75": f"{result.map_at(0.75):.3f}",
        "mAP@[.5:.95]": f"{result.map:.3f}",
    }
    ap75 = result.ap_at(0.75)
    for c, name in enumerate(result.class_names):
        details[f"{name}"] = (f"AP50 {result.ap[c, 0]:.3f}, AP75 {ap75[c]:.3f}, AP50-95 {result.ap_per_class[c]:.3f} "
                              f"({result.num_gt[c]} objects)")
    if result.confusion is not None:
        n = len(result.class_names)
        details["Missed objects (conf ≥ 0.25)"] = int(result.confusion[:n, n].sum())
        details["False alarms on background"] = int(result.confusion[n, :n].sum())
        details["Wrong class"] = int(result.confusion[:n, :n].sum() - np.trace(result.confusion[:n, :n]))
    if result.fppi.size:
        for target in (0.1, 1.0):
            reached = np.flatnonzero(result.fppi <= target)
            if reached.size:
                details[f"Miss rate at {target:g} FPPI"] = f"{result.miss_rate[reached[-1]]:.3f}"
    if ultra:
        details["Ultralytics model.val mAP@0.5"] = f"{ultra['map50']:.3f}"
        details["Ultralytics model.val mAP@[.5:.95]"] = f"{ultra['map']:.3f}"
        details["Why they can differ"] = (
            "Ultralytics interpolates the PR curve at 101 points, drops duplicate labels and matches "
            "boxes all at once; we use all-point interpolation and greedy matching by score."
        )
    details["Time"] = f"{report.seconds:.1f} s on {ym.device_label(report.device)}"
    details["Note"] = (
        "Each IoU threshold re-matches the detections. The course notebook reused the 0.5 matches, "
        "so its AP was the same for every threshold."
    )
    image = render_evaluation(result, f"{weights.name} on {report.split} ({report.images} images): "
                                      f"mAP@0.5 = {result.map_at(0.5):.3f}, mAP@[.5:.95] = {result.map:.3f}")
    return ToolResult(
        message=f"{report.split}: mAP@0.5 = {result.map_at(0.5):.3f}, mAP@[.5:.95] = {result.map:.3f}.",
        image=image, details=details, preview_only=True,
    )


class YoloEvaluateTool(ForensicsTool):
    tool_id = "yolo_evaluate"
    title = "YOLO: Evaluate (mAP, PR, FPPI)"
    category = CATEGORY
    description = (
        "Extra: evaluates the trained model on a whole split with our own implementation of IoU "
        "matching, PR curves, AP per class at IoU 0.50:0.95, mAP, miss rate vs FPPI and confusion matrix."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        _require_ultralytics()
        models = ym.blood_cell_models()
        splits = ("test", "valid")
        fields = [
            Field("model", "Model", "choice", 0, [p.name for p in models] + ["Other .pt file (below)…"]),
            Field("file", "Other weights", "file", "", filetypes=PT_TYPES),
            Field("data", "Dataset (data.yaml)", "file", ym.BLOOD_CELL_DATASET / "data.yaml",
                  filetypes=(("YOLO dataset", "*.yaml"), ("All files", "*.*"))),
            Field("split", "Split", "choice", 0, splits),
            Field("ultra", "Also run Ultralytics model.val to compare the numbers", "check", True),
        ]

        def check(values: dict[str, Any]) -> str | None:
            if values["model"] >= len(models) and not (values["file"] and values["file"].is_file()):
                return "Train a blood-cell model first, or choose a .pt file."
            if values["data"] is None or not values["data"].is_file():
                return "The dataset does not exist: run 'YOLO: COCO → YOLO Dataset' first."
            return None

        values = ask_form(parent, self.title, fields, check=check)
        if values is None:
            return None
        weights = models[values["model"]] if values["model"] < len(models) else values["file"]
        split = splits[values["split"]]

        def task(report, cancelled):
            report(0, 1, f"Loading {weights.name}…")
            model = get_model(weights)
            evaluation = evaluate_split(model, values["data"], split, report, cancelled)
            ultra = None
            if values["ultra"]:
                report(1, 1, "Running Ultralytics model.val for comparison…")
                ultra = ym.ultralytics_validation(model, values["data"], split)
            return evaluation, ultra

        try:
            evaluation, ultra = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
        return evaluation_result(evaluation, ultra, Path(weights))
