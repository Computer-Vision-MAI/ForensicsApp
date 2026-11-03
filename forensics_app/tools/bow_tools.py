"""Set 7: Bag of Visual Words tools (codebook, histogram of words, classification).

The numerical work lives in ``bow_model`` and the drawings in ``bow_render``; these
classes only ask for input, run the work and package the ToolResult.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_dialogs import ask_bow_settings, show_classify_window
from .bow_model import (
    DEFAULT_MODEL_PATH,
    BowModel,
    BowSettings,
    ImageAnalysis,
    classify,
    describe_image,
    get_model,
    set_model,
    train_bow_model,
)
from .bow_render import (
    render_classification,
    render_codebook,
    render_confusion,
    render_neighbours,
    render_word_histogram,
    stack_vertical,
)
from .dialogs import TaskCancelled, run_with_progress

CATEGORY = "Object Recognition (BoW)"
TOP_WORDS = 5
TRAIN_WARNING = "This image was used to train the model: its result is optimistic. Use a test image."
SPLIT_TEXT = {
    "train": "TRAIN image (used to fit the model: optimistic result)",
    "test": "TEST image (held out: never seen during training)",
    "external": "External image (not part of the training dataset)",
}


def model_details(model: BowModel, path: Path | None = None) -> dict[str, object]:
    """Results-panel summary of a trained model: data, vocabulary and test performance."""
    metrics = model.metrics
    counts = {name: 0 for name in model.classes}
    for label in model.train_labels:
        counts[model.classes[int(label)]] += 1
    details: dict[str, object] = {
        "Operation": "BoW codebook + chi-squared SVM",
        "Classes": ", ".join(f"{name} ({counts[name]} train)" for name in model.classes),
        "Images": f"{len(model.train_paths)} train / {len(model.test_paths)} test (stratified)",
        "Training descriptors": metrics["train_descriptors"],
        "Vocabulary size K": model.k,
        "Histogram length": len(model.classifier.train_vectors[0]),
    }
    if model.k_search:
        details["K by cross-validation"] = "  ".join(f"K={k}: {acc:.1%}" for k, acc in model.k_search)
    details["SVM C (by CV)"] = f"{model.classifier.c:g}"
    details["SVM gamma (1/mean chi2)"] = f"{model.classifier.gamma:.3f}"
    if metrics["cv_accuracy"] == metrics["cv_accuracy"]:  # not NaN
        details["Cross-validation accuracy (train)"] = f"{metrics['cv_accuracy']:.1%}"
    details["Test accuracy"] = f"{metrics['accuracy']:.1%}"
    for name, (precision, recall, support) in metrics["per_class"].items():
        details[f"Test {name}"] = f"precision {precision:.0%}, recall {recall:.0%} ({support} images)"
    confusion = metrics["confusion"]
    for row, name in enumerate(model.classes):
        details[f"Confusion {name}"] = " | ".join(f"{model.classes[col]}: {confusion[row][col]}" for col in range(len(model.classes)))
    details.update(model.settings.describe())
    details["Training time"] = f"{metrics['seconds']:.1f} s"
    if path is not None:
        details["Model saved to"] = str(path.resolve())
    return details


def analysis_details(model: BowModel, analysis: ImageAnalysis, document: ImageDocument) -> dict[str, object]:
    details: dict[str, object] = {
        "Image": document.path.name if document.path is not None else "current image",
        "Data split": SPLIT_TEXT[analysis.split],
        "Features": len(analysis.features),
        "Vocabulary size K": model.k,
    }
    if analysis.in_train:
        details["Warning"] = TRAIN_WARNING
    top = analysis.word_frequency.argsort()[::-1][:TOP_WORDS]
    for rank, word in enumerate(top, start=1):
        share = model.exemplars[word].class_share
        typical = model.classes[int(share.argmax())] if share.sum() else "-"
        details[f"Top word #{rank}"] = f"vw {word}: {analysis.word_frequency[word]:.1%} of features (typical of {typical})"
    for rank, (path, label, distance) in enumerate(analysis.neighbours, start=1):
        details[f"Similar train #{rank}"] = f"{Path(path).name} ({label}), chi2 = {distance:.3f}"
    return details


def _image_title(document: ImageDocument, analysis: ImageAnalysis) -> str:
    name = document.path.name if document.path is not None else "Current image"
    label = {"train": "TRAIN image - optimistic", "test": "test image", "external": "external image"}
    return f"{name} ({label[analysis.split]})"


def _warning(analysis: ImageAnalysis) -> str | None:
    return TRAIN_WARNING if analysis.in_train else None


class BowCodebookTool(ForensicsTool):
    tool_id = "bow_codebook"
    title = "BoW: Build Codebook"
    category = CATEGORY
    description = (
        "Choose a dataset folder (one subfolder per class): extracts features, builds the visual "
        "vocabulary with K-Means, trains a chi-squared SVM and evaluates it on held-out images."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        folder = filedialog.askdirectory(
            title="Choose the dataset folder (one subfolder per class)", parent=parent, mustexist=True
        )
        if not folder:
            return None
        settings = ask_bow_settings(parent, self.title, BowSettings())
        if settings is None:
            return None

        def task(report, cancelled):
            return train_bow_model(folder, settings, report, cancelled)

        try:
            model = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
        set_model(model)  # usable in this session even if saving fails below
        message = (
            f"Codebook of {model.k} visual words built from {len(model.train_paths)} training images; "
            f"test accuracy {model.metrics['accuracy']:.1%}."
        )
        try:
            path: Path | None = model.save(DEFAULT_MODEL_PATH)
            save_error = None
        except (OSError, ValueError) as error:
            path, save_error = None, str(error)
            message += " The model could not be saved: it only lasts until the app is closed."
        details = model_details(model, path)
        if save_error is not None:
            details["Model NOT saved"] = save_error
        image = stack_vertical([render_codebook(model), render_confusion(model)])
        return ToolResult(message=message, image=image, details=details, preview_only=True)


class BowHistogramTool(ForensicsTool):
    tool_id = "bow_histogram"
    title = "BoW: Histogram of Visual Words"
    category = CATEGORY
    description = "Maps every feature of the image to its closest visual word and plots the word histogram."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        model = get_model()
        analysis = describe_image(model, document.current, document.path)
        title = _image_title(document, analysis)
        parts = [render_word_histogram(document.current, analysis, model, title, warn=analysis.in_train)]
        neighbours = render_neighbours(analysis)
        if neighbours is not None:
            parts.append(neighbours)
        message = f"Histogram of {model.k} visual words from {len(analysis.features)} features."
        if analysis.in_train:
            message += " Warning: this is a training image."
        return ToolResult(
            message=message,
            image=stack_vertical(parts),
            details={"Operation": "BoW histogram", **analysis_details(model, analysis, document)},
            preview_only=True,
        )


class BowClassifyTool(ForensicsTool):
    tool_id = "bow_classify"
    title = "BoW: Classify Image"
    category = CATEGORY
    description = "Shows the histogram of visual words; press Classify to run the trained SVM and see p(class)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        model = get_model()
        analysis = describe_image(model, document.current, document.path)
        title = _image_title(document, analysis)
        histogram = render_word_histogram(document.current, analysis, model, title, warn=analysis.in_train)

        def run_classifier() -> list[tuple[str, float]]:
            return classify(model, analysis)

        if parent is None:  # tests and scripts: no window, classify directly
            ranked = run_classifier()
        else:
            ranked = show_classify_window(parent, self.title, histogram, run_classifier, _warning(analysis))
        if ranked is None:
            return None

        best, probability = ranked[0]
        image_details = analysis_details(model, analysis, document)
        details: dict[str, object] = {
            "Operation": "BoW classification",
            "Data split": image_details.pop("Data split"),
            "Prediction": best,
        }
        details.update({f"p({name})": f"{p:.3f}" for name, p in ranked})
        details.update(image_details)
        message = f"p({best}) = {probability:.2f}"
        if analysis.in_train:
            message += " (training image: optimistic)"
        classification = render_classification(document.current, analysis, ranked, _warning(analysis))
        image = stack_vertical([classification, histogram])
        return ToolResult(message=message, image=image, details=details, preview_only=True)
