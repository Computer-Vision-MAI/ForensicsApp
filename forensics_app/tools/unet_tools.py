"""Set 11 tools: semantic segmentation of food trays with a U-Net (train, save, load, test).

* ``UnetDatasetTool``   explore the Tray Food Segmentation dataset and the chosen task
* ``UnetTrainTool``     11.1 train a U-Net in the app and save it with its passport
* ``UnetSegmentTool``   11.1 load a model and segment an image (original / mask / overlay)
* ``UnetEvaluateTool``  the passport and the notebook exercises (threshold, IoU, worst cases)
* ``UnetCompareTool``   several models side by side (skip connections, losses, tasks)

The tools only handle dialogs and packaging; the work is in ``tray_dataset``,
``unet_model``, ``segmentation_metrics`` and ``unet_render``. Everything slow runs on a
worker thread through ``run_with_progress``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time
import tkinter as tk
from tkinter import messagebox
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import display_path
from . import unet_model as um
from .base import ForensicsTool, ToolResult
from .cnn_features import to_rgb
from .dialogs import TaskCancelled, run_with_progress
from .segmentation_metrics import SplitEvaluation, binary_dice, dice_scores, evaluate_masks, iou_from_dice, iou_scores
from .tray_dataset import (
    IMAGE_SIZES,
    TASK_KEYS,
    LoadedSplit,
    dataset_stats,
    find_dataset_root,
    find_mask_for,
    is_dataset_root,
    make_task,
    read_class_names,
    read_image,
    read_mask,
    split_indices,
    split_pairs,
)
from .unet_render import render_comparison, render_dataset_overview, render_passport, render_segmentation
from .yolo_dialogs import Field, ask_form

CATEGORY = "Segmentation (U-Net)"
LOSS_OPTIONS = (("BCE + Dice (notebook)", "bce_dice"), ("BCE only (cross-entropy for 3+ classes)", "bce"),
                ("Dice only", "dice"))
OUTPUT_OPTIONS = ("Figure: original, mask, overlay, ground truth, errors",
                  "Overlay only (full resolution)", "Mask only (class colours)")


def _dataset_root_or_raise(root: Path | None) -> Path:
    if root is None or not is_dataset_root(root):
        raise FileNotFoundError(
            "TrayDataset not found. Unzip TrayDataset.zip from the Lab 10 materials into data/TrayDataset "
            "(folders XTrain, yTrain, XTest, yTest)."
        )
    return root


def _models_or_raise() -> list[um.UnetPassport]:
    models = um.list_models()
    if not models:
        raise ValueError(um.NO_MODELS)
    return models


def _percent_rows(mask: np.ndarray, names: list[str], prefix: str) -> dict[str, str]:
    counts = np.bincount(mask.ravel(), minlength=len(names))[: len(names)]
    total = max(int(counts.sum()), 1)
    rows = {}
    for c in np.argsort(-counts):
        if counts[c] and c > 0:
            rows[f"{prefix} {names[c]}"] = f"{counts[c] / total:.1%} of the pixels"
    return rows


def passport_details(passport: um.UnetPassport) -> dict[str, object]:
    data = passport.data
    training = data.get("training", {})
    architecture = data.get("architecture", {})
    dataset = data.get("dataset", {})
    rows: dict[str, object] = {
        "Model": passport.name,
        "Model folder": display_path(passport.folder),
        "Created": data.get("created"),
        "Framework": ", ".join(f"{k} {v}" for k, v in data.get("framework", {}).items()) or None,
        "Task": data["task"]["label"],
        "Classes": f"{passport.num_classes}: " + ", ".join(passport.class_names[:12])
        + (" …" if passport.num_classes > 12 else ""),
        "Input": data["input"].get("preprocessing"),
        "Architecture": architecture.get("summary"),
        "Parameters": f"{architecture.get('parameters', 0):,}",
        "Output": architecture.get("output"),
        "Loss": training.get("loss"),
        "Optimizer / batch": f"{training.get('optimizer')} / {training.get('batch_size')}",
        "Augmentation": training.get("augmentation"),
        "Epochs": f"{training.get('epochs_done')} of {training.get('epochs')}"
        + (" (stopped early)" if training.get("stopped_early") else ""),
        "Best epoch": f"{training.get('best_epoch')} ({training.get('selection')})",
        "Training time": f"{training.get('seconds', 0) / 60:.1f} min on {training.get('device')}",
        "Images (train / val / test)": f"{dataset.get('train_images')} / {dataset.get('val_images')} / "
        f"{dataset.get('test_images')} (val = {dataset.get('val_fraction', 0):.0%} of XTrain, seed {dataset.get('seed')}, "
        f"{dataset.get('split', 'random (notebook)')})",
    }
    for split, label in (("val", "Validation"), ("test", "Test")):
        saved = passport.split(split)
        if saved:
            rows[f"{label} Dice"] = f"{saved['dice_mean']:.3f} ± {saved.get('dice_std', 0):.3f} ({saved['images']} images)"
            rows[f"{label} IoU"] = f"{saved['iou_mean']:.3f}"
            rows[f"{label} pixel accuracy"] = f"{saved['pixel_accuracy']:.3f}"
    return {k: v for k, v in rows.items() if v is not None}


def worst_cases(evaluation: SplitEvaluation, split: LoadedSplit, masks: np.ndarray, gts: np.ndarray,
                n: int = 3) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]]:
    return [(split.names[i], split.images[i], gts[i], masks[i], float(evaluation.dice[i]))
            for i in evaluation.worst(n)]


def evaluation_rows(evaluation: SplitEvaluation, split_label: str, passport: um.UnetPassport) -> dict[str, object]:
    rows: dict[str, object] = {
        f"{split_label}: images": len(evaluation.names),
        f"{split_label}: Dice": f"{evaluation.mean_dice:.4f} ± {np.std(evaluation.dice):.4f}",
        f"{split_label}: IoU": f"{evaluation.mean_iou:.4f}",
        f"{split_label}: pixel accuracy": f"{evaluation.pixel_accuracy:.4f}",
    }
    if passport.binary:
        rows["Threshold used"] = f"{evaluation.threshold:.2f}"
        if evaluation.sweep_dice.size:
            best = int(np.argmax(evaluation.sweep_dice))
            by_threshold = dict(zip(np.round(evaluation.sweep_thresholds, 2), evaluation.sweep_dice))
            rows["Exercise 1: Dice at 0.3 / 0.5 / 0.7"] = " / ".join(
                f"{by_threshold.get(t, float('nan')):.4f}" for t in (0.3, 0.5, 0.7))
            rows["Exercise 1: best threshold"] = (
                f"{evaluation.sweep_thresholds[best]:.2f} (Dice {evaluation.sweep_dice[best]:.4f})")
        food = evaluation.class_dice_per_image[:, 1]
        rows["Exercise 4: IoU = Dice / (2 − Dice)"] = (
            f"per image the identity holds exactly (max error "
            f"{np.nanmax(np.abs(iou_from_dice(food) - evaluation.class_iou_per_image[:, 1]), initial=0):.1e}); "
            f"on averages it does not: IoU({evaluation.mean_dice:.4f}) = {iou_from_dice(evaluation.mean_dice):.4f} "
            f"vs mean IoU {evaluation.mean_iou:.4f}")
    else:
        for c, name in enumerate(evaluation.class_names):
            value = evaluation.class_dice[c]
            rows[f"Dice {name}"] = "not present" if np.isnan(value) else (
                f"{value:.3f} (IoU {evaluation.class_iou[c]:.3f}, in "
                f"{int(np.sum(~np.isnan(evaluation.class_dice_per_image[:, c])))} images)")
    worst = evaluation.worst(3)
    rows["Exercise 5: worst images"] = ", ".join(f"{evaluation.names[i]} ({evaluation.dice[i]:.3f})" for i in worst)
    return rows


# --------------------------------------------------------------------------- dataset


class UnetDatasetTool(ForensicsTool):
    tool_id = "unet_dataset"
    title = "U-Net: Explore Tray Dataset"
    category = CATEGORY
    description = (
        "Set 11: the Tray Food Segmentation dataset (1,241 + 8 images, 43 classes). Shows the class "
        "frequencies, the food fraction and sample masks merged into the chosen task."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        fields = [
            Field("root", "Dataset folder", "dir", find_dataset_root() or ""),
            Field("task", "Task", "choice", 0, [make_task(k).label for k in TASK_KEYS]),
        ]
        values = ask_form(parent, self.title, fields, check=lambda v: None if is_dataset_root(v["root"]) else
                          "Choose the TrayDataset folder (with XTrain and yTrain).")
        if values is None:
            return None
        root = _dataset_root_or_raise(values["root"])
        task_key = TASK_KEYS[values["task"]]

        def task(report, cancelled):
            return explore_dataset(root, task_key, report)

        try:
            return run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None


def explore_dataset(root: Path, task_key: str = "binary", report=None, seed: int = 0) -> ToolResult:
    names = list(read_class_names(root))
    task = make_task(task_key, names)
    stats = [dataset_stats(root, split, len(names), report) for split in ("train", "test")]
    pairs = split_pairs(root, "train")
    samples = []
    for image_path, mask_path in random.Random(seed).sample(pairs, min(4, len(pairs))):
        samples.append((image_path.name, np.array(read_image(image_path)), read_mask(mask_path)))
    image = render_dataset_overview(stats, names, samples, task.class_names, task.lookup)
    train = stats[0]
    share = train.pixels_per_class / max(train.pixels_per_class.sum(), 1)
    details: dict[str, object] = {
        "Operation": "Tray dataset exploration",
        "Dataset": display_path(root),
        "Images": ", ".join(f"{s.split} {s.images}" for s in stats),
        "Image sizes": ", ".join(f"{w}×{h} ({n})" for (w, h), n in train.size_counts.items()),
        "Classes": f"{len(names)} (0 background, 1 tray, 2-4 cutlery/form/straw, 5-42 food and drinks)",
        "Task": task.label,
        "Task classes": ", ".join(task.class_names) if task.num_classes <= 3 else f"{task.num_classes}",
        "Background + tray": f"{share[:2].sum():.1%} of the pixels",
        "Food (classes 5-42)": f"{share[5:].sum():.1%} of the pixels (notebook: 10.6% at 128×208)",
        "Food per image": f"{train.food_fraction.mean():.1%} on average, from {train.food_fraction.min():.1%} "
        f"to {train.food_fraction.max():.1%}",
        "Why Dice, not accuracy": f"'no food' everywhere would be {1 - share[5:].sum():.1%} accurate but has Dice 0",
    }
    order = np.argsort(-train.images_with_class)
    details["Most common classes (images)"] = ", ".join(
        f"{names[c]} {train.images_with_class[c]}" for c in order[:8])
    rare = [c for c in np.argsort(train.images_with_class) if train.images_with_class[c] > 0][:6]
    details["Rarest classes (images)"] = ", ".join(f"{names[c]} {train.images_with_class[c]}" for c in rare)
    missing = [names[c] for c in range(len(names)) if train.images_with_class[c] == 0]
    if missing:
        details["Classes never seen in XTrain"] = ", ".join(missing)
    return ToolResult(message=f"TrayDataset: {stats[0].images} training and {stats[1].images} test images, "
                      f"{len(names)} classes, {share[5:].sum():.1%} food pixels.",
                      image=image, details=details, preview_only=True)


# --------------------------------------------------------------------------- training


DEVICE_OPTIONS = ("Automatic (GPU if available)", "CPU")


class UnetTrainTool(ForensicsTool):
    tool_id = "unet_train"
    title = "U-Net: Train Segmentation Model"
    category = CATEGORY
    description = (
        "Set 11.1: trains the notebook's U-Net on TrayDataset in the app (food vs not food, 3 classes or "
        "all 43), keeps the epoch with the best validation Dice and saves it to models/unet with its passport."
    )
    requires_image = False

    def ask(self, parent: tk.Misc) -> um.TrainSettings | None:
        device = um.default_device()
        sizes = list(IMAGE_SIZES)
        speed = {"cuda": "a few seconds", "mps": "about 15 s on an M2", "cpu": "about 70 s"}[device]
        fields = [
            Field("root", "Dataset folder", "dir", find_dataset_root() or ""),
            Field("task", "Task", "choice", 0, [make_task(k).label for k in TASK_KEYS]),
            Field("loss", "Loss (Exercise 3)", "choice", 0, [label for label, _ in LOSS_OPTIONS]),
            Field("skip", "Skip connections (untick for Exercise 2)", "check", True),
            Field("base", "Base channels (16 → 482,737 parameters)", "int", 16, minimum=4, maximum=64),
            Field("size", "Image size (height × width)", "choice", 1, sizes),
            Field("epochs", "Epochs", "int", 15, minimum=1, maximum=500),
            Field("batch", "Batch size", "int", 16, minimum=1, maximum=256),
            Field("lr", "Learning rate (Adam)", "float", 1e-3, minimum=1e-6, maximum=1.0),
            Field("val", "Validation fraction of XTrain", "float", 0.15, minimum=0.05, maximum=0.5),
            Field("seed", "Random seed (42 = the notebook's validation images)", "int", 42, minimum=0, maximum=2**31 - 1),
            Field("by_tray", "Validation = whole trays (XTrain is 17 trays × 73 photos; the notebook's random split "
                  "puts photos of the same tray in both sets)", "check", False),
            Field("augment", "Random horizontal flip (image and mask)", "check", True),
            Field("device", "Device", "choice", 0, DEVICE_OPTIONS),
            Field("name", "Model name (empty = automatic)", "str", ""),
            Field("note", f"Device: {um.device_label(device)}. One epoch at 128 × 208 takes {speed}; "
                  "Cancel stops after the current batch and still saves the best epoch so far.", "note"),
        ]

        def check(values: dict[str, Any]) -> str | None:
            if not is_dataset_root(values["root"]):
                return "Choose the TrayDataset folder (with XTrain and yTrain)."
            return None

        values = ask_form(parent, self.title, fields, check=check)
        if values is None:
            return None
        settings = um.TrainSettings(
            root=values["root"], task=TASK_KEYS[values["task"]], size=IMAGE_SIZES[sizes[values["size"]]],
            epochs=values["epochs"], batch=values["batch"], lr=values["lr"], loss=LOSS_OPTIONS[values["loss"]][1],
            skip=values["skip"], base=values["base"], augment=values["augment"], val_fraction=values["val"],
            seed=values["seed"], split_by_tray=values["by_tray"], device="" if values["device"] == 0 else "cpu",
            name=values["name"],
        )
        name = um.safe_name(settings.name or settings.default_name())
        target = um.UNET_DIR / name
        if target.exists() and not messagebox.askyesno(
            self.title, f"{display_path(target)} already exists. Replace it with the new model?", parent=parent,
        ):
            return None
        return settings

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        um.require_torch()
        settings = self.ask(parent)
        if settings is None:
            return None

        def task(report, cancelled):
            return um.train(settings, report, cancelled)

        try:
            outcome = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
        return training_result(outcome)


def training_result(outcome: um.TrainOutcome) -> ToolResult:
    passport = outcome.passport
    worst = []
    if outcome.extra.get("val_split") is not None:
        split = outcome.extra["val_split"]
        gts = passport.task.map(split.masks)
        worst = worst_cases(outcome.val, split, outcome.extra["val_masks"], gts)
    image = render_passport(passport, worst=worst)
    details: dict[str, object] = {"Operation": "U-Net training"}
    details.update(passport_details(passport))
    details.update(evaluation_rows(outcome.val, "Validation", passport))
    if outcome.test is not None:
        details["Test: Dice (as in the notebook)"] = (
            f"{outcome.test.mean_dice:.3f} ± {np.std(outcome.test.dice):.3f} (n = {len(outcome.test.names)} images)")
    message = (f"Saved {display_path(passport.folder)}: best epoch {outcome.best_epoch}, validation Dice "
               f"{outcome.val.mean_dice:.3f}" + (f", test Dice {outcome.test.mean_dice:.3f}" if outcome.test else "")
               + (" (stopped early)" if outcome.stopped_early else "") + ".")
    return ToolResult(message=message, image=image, details=details, preview_only=True)


# --------------------------------------------------------------------------- segmentation


@dataclass(frozen=True)
class SegmentSettings:
    passport: um.UnetPassport
    source: str  # "working", "random_val", or a test image name
    threshold: float = 0.5
    output: int = 0
    seed: int | None = None


def segment(passport: um.UnetPassport, image: Image.Image, image_path: Path | None, settings: SegmentSettings,
            segment_fn=None, val_names: set[str] | None = None) -> ToolResult:
    """Segment one image and compare it with its mask when there is one (no Tk).

    ``segment_fn(image) -> (mask, confidence, network_input)`` defaults to the saved network.
    """
    names = passport.class_names
    rgb = to_rgb(image)
    started = time.perf_counter()
    if segment_fn is None:
        mask, confidence, small = um.segment_image(passport, rgb, settings.threshold)
    else:
        mask, confidence, small = segment_fn(rgb)
    seconds = time.perf_counter() - started
    root = Path(passport.data.get("dataset", {}).get("root", "")) or None
    mask_path = find_mask_for(image_path, root)
    gt = None
    if mask_path is not None:
        source_mask = read_mask(mask_path)
        if source_mask.shape == mask.shape:
            gt = passport.task.map(source_mask)
    name = image_path.name if image_path is not None else "working image"
    seen = ""
    if image_path is not None and image_path.parent.name == "XTrain":
        seen = " (validation image, unseen)" if val_names is not None and name in val_names else \
            " (TRAINING image: the model has seen it)" if val_names is not None else ""
    elif image_path is not None and image_path.parent.name == "XTest":
        seen = " (test image, unseen)"
    image_array = np.array(rgb)
    details: dict[str, object] = {
        "Operation": "U-Net segmentation",
        "Model": passport.name,
        "Task": passport.data["task"]["label"],
        "Image": f"{name}{seen}, {rgb.width}×{rgb.height}",
        "Network input": f"resized to {passport.size[1]}×{passport.size[0]}; the probabilities are resized back "
        "(bilinear) before the threshold / argmax",
    }
    if passport.binary:
        details["Threshold"] = f"P(food) > {settings.threshold:.2f}"
    details["Time"] = f"{seconds * 1000:.0f} ms"
    details.update(_percent_rows(mask, names, "Predicted"))
    title = f"{passport.name} on {name}{seen}"
    if gt is not None:
        k = len(names)
        dice = binary_dice(mask, gt) if passport.binary else None
        evaluation = evaluate_masks([mask], [gt], names, [name], settings.threshold if passport.binary else None)
        details["Ground truth"] = display_path(mask_path)
        details["Dice"] = f"{evaluation.dice[0]:.4f}" + (" (food)" if passport.binary else " (mean over the classes present)")
        details["IoU"] = f"{evaluation.iou[0]:.4f}"
        details["Pixel accuracy"] = f"{evaluation.pixel_accuracy:.4f}"
        if not passport.binary:
            class_dice = dice_scores(mask, gt, k)
            class_iou = iou_scores(mask, gt, k)
            for c in range(1, k):
                if not np.isnan(class_dice[c]):
                    details[f"Dice {names[c]}"] = f"{class_dice[c]:.3f} (IoU {class_iou[c]:.3f})"
        errors = {"TP": (mask > 0) & (mask == gt), "FP": (mask > 0) & (gt == 0), "FN": (mask == 0) & (gt > 0)}
        details["Pixels TP / FP / FN"] = " / ".join(f"{int(v.sum()):,}" for v in errors.values())
        title += f": Dice {evaluation.dice[0]:.3f}, IoU {evaluation.iou[0]:.3f}"
        message = f"{name}: Dice {evaluation.dice[0]:.3f}, IoU {evaluation.iou[0]:.3f}" + (
            f" (food {dice:.3f})" if dice is not None else "") + "."
    else:
        details["Ground truth"] = "no mask found (only for images of TrayDataset: XTrain/x.jpg → yTrain/x.png)"
        message = f"{name}: segmented with {passport.name}."
    if settings.output == 1:
        from .unet_render import overlay

        output = Image.fromarray(overlay(image_array, mask, len(names)))
    elif settings.output == 2:
        from .unet_render import colorize

        output = Image.fromarray(colorize(mask, len(names)))
    else:
        output = render_segmentation(image_array, mask, confidence, names, gt,
                                     settings.threshold if passport.binary else None, title, small)
    return ToolResult(message=message, image=output, details=details, preview_only=True)


def validation_names(passport: um.UnetPassport) -> set[str]:
    root = Path(passport.data.get("dataset", {}).get("root", ""))
    pairs = split_pairs(root, "train") if is_dataset_root(root) else []
    if not pairs:
        return set()
    dataset = passport.data.get("dataset", {})
    names = [image.name for image, _ in pairs]
    _, val_idx = split_indices(names, dataset.get("val_fraction", 0.15), dataset.get("seed", 42),
                               dataset.get("split_by_tray", False))
    return {names[i] for i in val_idx}


class UnetSegmentTool(ForensicsTool):
    tool_id = "unet_segment"
    title = "U-Net: Segment Image"
    category = CATEGORY
    description = (
        "Set 11.1: loads a trained U-Net and segments the working image, a test image or a random validation "
        "image. Shows the original, the mask and the overlay (as in the slides) and, for dataset images, the "
        "ground truth, the errors and the Dice."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        um.require_torch()
        models = _models_or_raise()
        root = Path(models[0].data.get("dataset", {}).get("root", "")) if models else None
        test_names = [p.name for p, _ in split_pairs(root, "test")] if root and is_dataset_root(root) else []
        sources: list[tuple[str, str]] = []
        if document.current is not None:
            sources.append((f"Working image ({document.path.name if document.path else 'unsaved'})", "working"))
        sources += [(f"Test image {n}", n) for n in test_names]
        sources.append(("Random validation image (not used for training)", "random_val"))
        fields = [
            Field("model", "Model", "choice", 0, [m.label for m in models]),
            Field("source", "Image", "choice", 0, [label for label, _ in sources]),
            Field("threshold", "Threshold on P(food) (binary models only)", "float", 0.5, minimum=0.0, maximum=1.0),
            Field("output", "Show", "choice", 0, OUTPUT_OPTIONS),
        ]
        values = ask_form(parent, self.title, fields)
        if values is None:
            return None
        passport = models[values["model"]]
        settings = SegmentSettings(passport, sources[values["source"]][1], values["threshold"], values["output"])
        dataset_root = Path(passport.data.get("dataset", {}).get("root", ""))

        def task(report, cancelled):
            report(0, 2, f"Loading {passport.name}…")
            val_names = validation_names(passport)
            if settings.source == "working":
                assert document.current is not None
                image, path = document.current, document.path
            else:
                if settings.source == "random_val":
                    if not val_names:
                        raise FileNotFoundError(f"TrayDataset not found at {dataset_root}.")
                    path = dataset_root / "XTrain" / random.Random(settings.seed).choice(sorted(val_names))
                else:
                    path = dataset_root / "XTest" / settings.source
                image = read_image(path)
            report(1, 2, f"Segmenting {path.name if path else 'the working image'}…")
            return segment(passport, image, path, settings, val_names=val_names)

        try:
            return run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None


# --------------------------------------------------------------------------- evaluation


class UnetEvaluateTool(ForensicsTool):
    tool_id = "unet_evaluate"
    title = "U-Net: Model Passport & Evaluation"
    category = CATEGORY
    description = (
        "Set 11: the passport of a trained U-Net (curves, settings) and its evaluation on the validation or "
        "test split: Dice vs threshold, Dice vs IoU, per-class Dice, confusion matrix and the worst images."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        models = _models_or_raise()
        splits = (("Validation (186 images, chosen by seed)", "val"), ("Test (XTest, 8 images)", "test"))
        fields = [
            Field("model", "Model", "choice", 0, [m.label for m in models]),
            Field("rerun", "Re-run the model on a split (needs PyTorch); otherwise show the saved passport", "check",
                  um.torch_available()),
            Field("split", "Split", "choice", 0, [label for label, _ in splits]),
            Field("threshold", "Threshold on P(food) (binary models only)", "float", 0.5, minimum=0.0, maximum=1.0),
            Field("worst", "Worst images to show (Exercise 5)", "int", 3, minimum=0, maximum=6),
        ]
        values = ask_form(parent, self.title, fields)
        if values is None:
            return None
        passport = models[values["model"]]
        if not values["rerun"]:
            details = {"Operation": "U-Net passport"}
            details.update(passport_details(passport))
            return ToolResult(message=f"Passport of {passport.name}.", image=render_passport(passport),
                              details=details, preview_only=True)
        um.require_torch()
        split_key = splits[values["split"]][1]

        def task(report, cancelled):
            evaluation, split, prediction = um.evaluate_passport(
                passport, split_key, threshold=values["threshold"], report=report, cancelled=cancelled)
            return evaluation, split, prediction

        try:
            evaluation, split, prediction = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
        return evaluation_result(passport, evaluation, split, prediction.masks, split_key, values["worst"])


def evaluation_result(passport: um.UnetPassport, evaluation: SplitEvaluation, split: LoadedSplit,
                      masks: np.ndarray, split_key: str, n_worst: int = 3) -> ToolResult:
    label = "validation" if split_key == "val" else "test"
    gts = passport.task.map(split.masks)
    worst = worst_cases(evaluation, split, masks, gts, n_worst) if n_worst else []
    image = render_passport(passport, evaluation, worst, label)
    details: dict[str, object] = {"Operation": f"U-Net evaluation ({label})"}
    details.update(evaluation_rows(evaluation, label.capitalize(), passport))
    details.update(passport_details(passport))
    return ToolResult(message=f"{passport.name} on {label}: Dice {evaluation.mean_dice:.3f}, IoU {evaluation.mean_iou:.3f}.",
                      image=image, details=details, preview_only=True)


# --------------------------------------------------------------------------- comparison


class UnetCompareTool(ForensicsTool):
    tool_id = "unet_compare"
    title = "U-Net: Compare Models"
    category = CATEGORY
    description = (
        "Set 11 exercises 2, 3 and 6: compares saved U-Nets (with/without skip connections, BCE / Dice / both, "
        "binary / 3 classes): validation curves, Dice vs threshold and the validation and test Dice and IoU."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        models = _models_or_raise()
        fields = [Field(f"m{i}", m.label, "check", i < 6) for i, m in enumerate(models)]
        values = ask_form(parent, self.title, fields, intro="Models to compare (from their passports):",
                          check=lambda v: None if any(v.values()) else "Choose at least one model.")
        if values is None:
            return None
        chosen = [m for i, m in enumerate(models) if values[f"m{i}"]]
        return comparison_result(chosen)


def comparison_result(passports: list[um.UnetPassport]) -> ToolResult:
    details: dict[str, object] = {"Operation": "U-Net model comparison"}
    for p in passports:
        training = p.data.get("training", {})
        architecture = p.data.get("architecture", {})
        val, test = p.split("val"), p.split("test")
        details[p.name] = (
            f"{p.data['task']['key']}, {training.get('loss')}, skip {'yes' if architecture.get('skip_connections', True) else 'no'}, "
            f"{architecture.get('parameters', 0):,} params, best epoch {training.get('best_epoch')}: "
            f"val Dice {val.get('dice_mean', float('nan')):.3f} / IoU {val.get('iou_mean', float('nan')):.3f}"
            + (f", test Dice {test['dice_mean']:.3f} / IoU {test['iou_mean']:.3f}" if test else "")
            + f", {training.get('seconds', 0) / 60:.1f} min"
        )
    best = max(passports, key=lambda p: p.split("val").get("dice_mean") or 0)
    details["Best on validation"] = best.name
    details["Note"] = ("Each model's training loss is in its own units (BCE, Dice or both), so compare the "
                       "validation Dice, not the loss values. The models with different tasks measure different "
                       "things (food only vs mean over the classes).")
    return ToolResult(message=f"Compared {len(passports)} U-Net models; best validation Dice: {best.name}.",
                      image=render_comparison(passports), details=details, preview_only=True)
