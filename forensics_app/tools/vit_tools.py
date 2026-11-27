"""Set 10 extra tools: the Vision Transformer notebook (patches, SPT, training, evaluation).

* ``VitPatchesTool``   the notebook's patch visualisation and the shifted patch tokens (no PyTorch)
* ``VitTrainTool``     train the ViT (patches or SPT, optional LSA) on CIFAR-100 / CIFAR-10 in the app
* ``VitClassifyTool``  classify a test image or the working image, with attention rollout
* ``VitCompareTool``   passport of one model, or several models compared (confusion matrix,
                       classification report, image-by-image agreement)
"""

from __future__ import annotations

import random
import tkinter as tk
from tkinter import messagebox
from typing import Any

import numpy as np

from forensics_app.core import ImageDocument
from forensics_app.paths import display_path
from . import vit_data as vd
from . import vit_model as vm
from .base import ForensicsTool, ToolResult
from .cnn_features import to_rgb
from .dialogs import TaskCancelled, run_with_progress
from .unet_model import default_device, device_label, require_torch
from .vit_render import render_patches, render_vit_classification, render_vit_comparison, render_vit_passport
from .yolo_dialogs import Field, ask_form

CATEGORY = "Vision Transformers (ViT)"


def _models_or_raise() -> list[vm.VitPassport]:
    models = vm.list_models()
    if not models:
        raise ValueError(vm.NO_MODELS)
    return models


def _downloaded_datasets() -> list[str]:
    return [k for k in vd.DATASET_KEYS if vd.is_downloaded(k)]


# --------------------------------------------------------------------------- patches


def patches_result(image: np.ndarray, title: str, image_size: int, patch: int, spt: bool) -> ToolResult:
    tokens, values = vd.tokens_per_image(image_size, patch, False)
    _, spt_values = vd.tokens_per_image(image_size, patch, True)
    details: dict[str, object] = {
        "Operation": "ViT patches and shifted patch tokens",
        "Image": title,
        "Resized to": f"{image_size}×{image_size} (the notebook's Resizing layer)",
        "Patches": f"{image_size // patch}×{image_size // patch} = {tokens} patches of {patch}×{patch} px",
        "Values per patch": f"{patch}·{patch}·3 = {values} (flattened, then a Dense layer projects them to 64)",
        "Sequence length": f"{tokens} tokens: self-attention compares {tokens}² = {tokens ** 2:,} pairs per head",
        "Position embedding": f"a learnt 64-D vector per position (0…{tokens - 1}) is added to each token, otherwise "
                              "the order of the patches would be lost",
    }
    if spt:
        details["SPT shifts"] = f"left-up, left-down, right-up, right-down by {patch // 2} px (half a patch)"
        details["SPT values per token"] = f"{patch}·{patch}·15 = {spt_values} (5 images × 3 channels), LayerNorm, Dense 64"
        details["Why SPT"] = ("a plain patch token only sees its 6×6 pixels; with the shifts each token also contains "
                              "the half-patch around it, a locality bias that helps ViTs on small datasets")
    if image_size % patch:
        details["Note"] = f"{image_size} is not a multiple of {patch}: the last {image_size % patch} px are dropped"
    image_out = render_patches(image, image_size, patch, title, spt)
    return ToolResult(message=f"{tokens} patches of {patch}×{patch} from a {image_size}×{image_size} image.",
                      image=image_out, details=details, preview_only=True)


class VitPatchesTool(ForensicsTool):
    tool_id = "vit_patches"
    title = "ViT: Patches & Shifted Patch Tokens"
    category = CATEGORY
    description = (
        "Set 10 extra notebook: how a Vision Transformer turns an image into tokens. Shows the 72×72 image cut "
        "into 6×6 patches and the Shifted Patch Tokenization (the 4 diagonal shifts and what one token contains)."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        datasets = _downloaded_datasets()
        sources = (["Working image"] if document.current is not None else []) + [
            f"Random test image from {vd.DATASETS[k].label}" for k in datasets]
        if not sources:
            raise ValueError("Open an image, or train a ViT once so that CIFAR is downloaded.")
        fields = [
            Field("source", "Image", "choice", 0, sources),
            Field("size", "Resize to (image_size)", "int", 72, minimum=8, maximum=512),
            Field("patch", "Patch size", "int", 6, minimum=2, maximum=64),
            Field("spt", "Show the shifted patch tokenization (SPT exercise)", "check", True),
        ]
        values = ask_form(parent, self.title, fields)
        if values is None:
            return None
        if document.current is not None and values["source"] == 0:
            image = np.asarray(to_rgb(document.current))
            title = document.path.name if document.path else "working image"
        else:
            key = datasets[values["source"] - (1 if document.current is not None else 0)]
            data = vd.load_cifar(key)
            i = random.randrange(len(data.x_test))
            image = data.x_test[i]
            title = f"test image {i}: {data.class_names[data.y_test[i]]}"
        return patches_result(image, title, values["size"], values["patch"], values["spt"])


# --------------------------------------------------------------------------- training


TOKENIZERS = (("Patches (notebook ViT)", False, False), ("Shifted Patch Tokenization (exercise)", True, False),
              ("SPT + Locality Self-Attention (full SPT paper)", True, True))


class VitTrainTool(ForensicsTool):
    tool_id = "vit_train"
    title = "ViT: Train Classifier"
    category = CATEGORY
    description = (
        "Set 10 extra notebook: trains the Vision Transformer (or its SPT version) on CIFAR-100 in the app, "
        "keeps the epoch with the best validation accuracy and saves it to models/vit with its passport."
    )
    requires_image = False

    def ask(self, parent: tk.Misc) -> vm.VitSettings | None:
        device = default_device()
        fields = [
            Field("dataset", "Dataset", "choice", 0, [vd.DATASETS[k].label + ("" if vd.is_downloaded(k) else
                  f" — download {vd.DATASETS[k].size_mb} MB") for k in vd.DATASET_KEYS]),
            Field("tokens", "Tokenizer", "choice", 0, [label for label, _, _ in TOKENIZERS]),
            Field("epochs", "Epochs", "int", 10, minimum=1, maximum=300),
            Field("batch", "Batch size", "int", 256, minimum=8, maximum=1024),
            Field("micro", "Micro-batch (gradient accumulation; 0 = whole batch)", "int",
                  64 if device == "mps" else 0, minimum=0, maximum=1024,
                  help="Splits each batch to fit in the GPU memory; the gradients are added up, so it is still "
                       "one optimiser step per batch."),
            Field("lr", "Learning rate (AdamW)", "float", 1e-3, minimum=1e-6, maximum=1.0),
            Field("wd", "Weight decay", "float", 1e-4, minimum=0.0, maximum=1.0),
            Field("layers", "Transformer layers", "int", 8, minimum=1, maximum=24),
            Field("heads", "Attention heads", "int", 4, minimum=1, maximum=16),
            Field("size", "Image size", "int", 72, minimum=16, maximum=224),
            Field("patch", "Patch size", "int", 6, minimum=2, maximum=32),
            Field("name", "Model name (empty = automatic)", "str", ""),
            Field("note", f"Device: {device_label(device)}. With the notebook settings one CIFAR epoch takes "
                  "about 2.5 min on an M2 (8 GB). Cancel keeps the best epoch so far.", "note"),
        ]

        def check(values: dict[str, Any]) -> str | None:
            if values["size"] % values["patch"]:
                return "The image size must be a multiple of the patch size."
            if values["micro"] and values["micro"] > values["batch"]:
                return "The micro-batch cannot be larger than the batch."
            return None

        values = ask_form(parent, self.title, fields, check=check)
        if values is None:
            return None
        _, spt, lsa = TOKENIZERS[values["tokens"]]
        settings = vm.VitSettings(
            dataset=vd.DATASET_KEYS[values["dataset"]], spt=spt, lsa=lsa, epochs=values["epochs"],
            batch=values["batch"], micro_batch=values["micro"], lr=values["lr"], weight_decay=values["wd"],
            image_size=values["size"], patch=values["patch"], heads=values["heads"], layers=values["layers"],
            name=values["name"],
        )
        source = vd.DATASETS[settings.dataset]
        if not vd.is_downloaded(settings.dataset) and not messagebox.askyesno(
            self.title, f"{source.label} is not in data/. Download it now ({source.size_mb} MB from {source.url})?",
            parent=parent,
        ):
            return None
        target = vm.VIT_DIR / vm.safe_name(settings.name or settings.default_name())
        if target.exists() and not messagebox.askyesno(self.title, f"Replace {display_path(target)}?", parent=parent):
            return None
        return settings

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        require_torch()
        settings = self.ask(parent)
        if settings is None:
            return None

        def task(report, cancelled):
            if not vd.is_downloaded(settings.dataset):
                vd.download(settings.dataset, report, cancelled)
            report(0, 1, "Loading the dataset…")
            return vm.train(settings, report, cancelled)

        try:
            outcome = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
        return passport_result(outcome.passport, "ViT training")


def passport_details(passport: vm.VitPassport) -> dict[str, object]:
    d = passport.data
    t, a, test = d.get("training", {}), d.get("architecture", {}), passport.test
    rows: dict[str, object] = {
        "Model": passport.name,
        "Model folder": display_path(passport.folder),
        "Created": d.get("created"),
        "Framework": ", ".join(f"{k} {v}" for k, v in d.get("framework", {}).items()),
        "Dataset": f"{d['dataset']['label']}: train {d['dataset']['train']:,}, validation (last 10 %) "
                   f"{d['dataset']['validation']:,}, test {d['dataset']['test']:,}",
        "Architecture": a.get("summary"),
        "Parameters": f"{a.get('parameters', 0):,}",
        "Optimizer": t.get("optimizer"),
        "Batch": f"{t.get('batch')}" + (f" (micro-batches of {t['micro_batch']})" if t.get("micro_batch") else ""),
        "Augmentation": t.get("augmentation"),
        "Epochs": f"{t.get('epochs_done')} of {t.get('epochs')}, best {t.get('best_epoch')} ({t.get('selection')})",
        "Training time": f"{t.get('seconds', 0) / 60:.1f} min on {t.get('device')}",
    }
    h = passport.history
    if h.get("val_accuracy"):
        best = int(np.argmax(h["val_accuracy"]))
        rows["Best validation accuracy / top-5"] = f"{h['val_accuracy'][best]:.2%} / {h['val_top5'][best]:.2%}"
        rows["Train accuracy at that epoch"] = f"{h['accuracy'][best]:.2%} (with augmentation and dropout on)"
    if test:
        rows["Test accuracy"] = f"{test['accuracy']:.2%}"
        rows["Test top-5 accuracy"] = f"{test['top5']:.2%}"
        rows["Test loss"] = f"{test['loss']:.4f}"
        rows["Macro / weighted F1"] = f"{test.get('macro_f1', 0):.3f} / {test.get('weighted_f1', 0):.3f}"
        names = passport.class_names
        f1 = np.array(test.get("f1", []))
        if f1.size:
            order = np.argsort(f1)
            rows["Best classes (F1)"] = ", ".join(f"{names[i]} {f1[i]:.2f}" for i in order[::-1][:5])
            rows["Worst classes (F1)"] = ", ".join(f"{names[i]} {f1[i]:.2f}" for i in order[:5])
            confusion = np.array(test["confusion"])
            off = confusion.copy()
            np.fill_diagonal(off, 0)
            pairs = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:5], off.shape))[0]
            rows["Most frequent confusions"] = ", ".join(f"{names[r]}→{names[c]} {off[r, c]}" for r, c in pairs)
    return {k: v for k, v in rows.items() if v is not None}


def passport_result(passport: vm.VitPassport, operation: str = "ViT passport") -> ToolResult:
    details: dict[str, object] = {"Operation": operation}
    details.update(passport_details(passport))
    test = passport.test
    message = f"{passport.name}: test accuracy {test.get('accuracy', 0):.1%}, top-5 {test.get('top5', 0):.1%}."
    return ToolResult(message=message, image=render_vit_passport(passport), details=details, preview_only=True)


# --------------------------------------------------------------------------- classify


def classification_result(passport: vm.VitPassport, image: np.ndarray, title: str, true_label: int | None,
                          prediction: vm.VitPrediction) -> ToolResult:
    names = passport.class_names
    probs = prediction.probs
    order = np.argsort(-probs)
    best = int(order[0])
    details: dict[str, object] = {
        "Operation": "ViT classification",
        "Model": passport.name,
        "Image": title,
        "Network input": "centre square resized to 32×32 (CIFAR size), then normalised and resized to "
                         f"{passport.settings.image_size}×{passport.settings.image_size} inside the model",
        "Prediction": f"{names[best]} ({probs[best]:.1%})",
    }
    if true_label is not None:
        details["True class"] = names[true_label]
        details["Correct"] = "yes" if best == true_label else (
            f"no (true class is #{int(np.where(order == true_label)[0][0]) + 1} with {probs[true_label]:.1%})")
    for rank, i in enumerate(order[:5], start=1):
        details[f"Top {rank}"] = f"{names[i]} {probs[i]:.3f}"
    if prediction.rollout is not None:
        details["Attention rollout"] = ("attention of each layer (mean of the heads) + identity for the skip "
                                        "connection, multiplied over the layers; averaged over all tokens because this "
                                        "ViT has no class token (the head flattens every token)")
    details["Time"] = f"{prediction.seconds * 1000:.0f} ms"
    figure = render_vit_classification(image, prediction.network_input, probs, names, prediction.rollout, true_label,
                                       f"{passport.name}: {title}", passport.settings.image_size)
    message = f"ViT: {names[best]} ({probs[best]:.1%})" + (
        "" if true_label is None else f", true class {names[true_label]}") + "."
    return ToolResult(message=message, image=figure, details=details, preview_only=True)


class VitClassifyTool(ForensicsTool):
    tool_id = "vit_classify"
    title = "ViT: Classify Image"
    category = CATEGORY
    description = (
        "Set 10 extra notebook: classifies a CIFAR test image or the working image with a trained ViT, shows the "
        "top classes and an attention-rollout map of the patches the transformer used."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        require_torch()
        models = _models_or_raise()
        sources = ["Random test image (true class known)"] + (["Working image"] if document.current is not None else [])
        fields = [
            Field("model", "Model", "choice", 0, [m.label for m in models]),
            Field("source", "Image", "choice", 0, sources),
            Field("index", "Test image index (-1 = random)", "int", -1, minimum=-1, maximum=59999),
            Field("rollout", "Attention rollout map", "check", True),
        ]
        values = ask_form(parent, self.title, fields)
        if values is None:
            return None
        passport = models[values["model"]]

        def task(report, cancelled):
            report(0, 2, f"Loading {passport.name}…")
            if values["source"] == 0:
                data = vd.load_cifar(passport.data["dataset"]["key"])
                i = values["index"] if 0 <= values["index"] < len(data.x_test) else random.randrange(len(data.x_test))
                image, true_label, title = data.x_test[i], int(data.y_test[i]), f"test image {i}"
                prediction = vm.classify(passport, image, values["rollout"])
            else:
                rgb = to_rgb(document.current)
                image, true_label = np.asarray(rgb), None
                title = document.path.name if document.path else "working image"
                prediction = vm.classify(passport, rgb, values["rollout"])
            report(1, 2, "Drawing…")
            return classification_result(passport, image, title, true_label, prediction)

        try:
            return run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None


# --------------------------------------------------------------------------- passport / compare


def comparison_result(passports: list[vm.VitPassport]) -> ToolResult:
    if len(passports) == 1:
        return passport_result(passports[0])
    details: dict[str, object] = {"Operation": "ViT model comparison"}
    for p in passports:
        t, test = p.data.get("training", {}), p.test
        details[p.name] = (f"{p.data['architecture'].get('tokenizer')}/{p.data['architecture'].get('attention')}, "
                           f"{p.data['architecture'].get('parameters', 0):,} params: test accuracy "
                           f"{test.get('accuracy', 0):.2%}, top-5 {test.get('top5', 0):.2%}, macro F1 "
                           f"{test.get('macro_f1', 0):.3f}, best epoch {t.get('best_epoch')}, "
                           f"{t.get('seconds', 0) / 60:.1f} min")
    a, b = passports[0], passports[1]
    pa, pb = np.array(a.test.get("predictions") or []), np.array(b.test.get("predictions") or [])
    truth = np.array(a.test.get("labels") or [])
    if pa.size and pa.size == pb.size == truth.size:
        ra, rb = pa == truth, pb == truth
        details["Image by image"] = (f"both right {np.sum(ra & rb):,}, only {a.name} {np.sum(ra & ~rb):,}, "
                                     f"only {b.name} {np.sum(~ra & rb):,}, both wrong {np.sum(~ra & ~rb):,}")
    best = max(passports, key=lambda p: p.test.get("accuracy", 0))
    return ToolResult(message=f"Compared {len(passports)} ViT models; best test accuracy: {best.name}.",
                      image=render_vit_comparison(passports), details=details, preview_only=True)


class VitCompareTool(ForensicsTool):
    tool_id = "vit_compare"
    title = "ViT: Passport & Compare Models"
    category = CATEGORY
    description = (
        "Set 10 extra notebook: one model's passport (curves, confusion matrix, classification report) or several "
        "models compared (ViT vs SPT): accuracy, top-5, F1 per class and image-by-image agreement."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        models = _models_or_raise()
        fields = [Field(f"m{i}", m.label, "check", i < 2) for i, m in enumerate(models)]
        values = ask_form(parent, self.title, fields, intro="One model shows its passport; two or more are compared "
                          "(the first two in the list are compared class by class).",
                          check=lambda v: None if any(v.values()) else "Choose at least one model.")
        if values is None:
            return None
        return comparison_result([m for i, m in enumerate(models) if values[f"m{i}"]])
