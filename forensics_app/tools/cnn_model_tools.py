"""Set 8 tools: classify an image with a trained CNN, and project its features in 2-D.

* "CNN: Classify Image" (Set 8.2) loads a model trained in the notebook, shows its
  passport and classifies a saved test image, the current image or a file. The result
  figure shows the image, the exact input the network received, a Grad-CAM heatmap of
  where the network found the predicted class, the class probabilities, and the
  training history and test confusion matrix from the passport.
* "CNN: Feature Projection" (Set 8.3) loads the test-set feature vectors saved by the
  notebook and projects them onto 2-D with t-SNE and/or UMAP, one colour per class.

The computation lives in ``cnn_model``; this module only holds the dialogs, the figures
and the text for the results panel.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import display_path
from .base import ForensicsTool, ToolResult
from .bow_render import figure_to_image
from .cnn_features import to_rgb
from .cnn_model import (
    KERAS_INSTALL_HINT,
    PROJECTION_METHODS,
    Classification,
    FeatureSet,
    ModelPassport,
    classify,
    explain,
    input_to_image,
    keras_available,
    list_models,
    list_test_samples,
    load_network,
    load_test_features,
    no_models_message,
    project,
    separation_score,
    subsample,
    true_class_from_path,
    umap_available,
)
from .dialogs import TaskCancelled, load_image_file, run_with_progress

CATEGORY = "Deep Learning (CNN)"  # sidebar section of both tools
MAX_BARS = 10  # probability bars shown (all of them for CIFAR-10)
MAX_PLOT_SIDE = 900  # the input image is shrunk to this size in the figure, for speed
GRADCAM_SIDE = 320  # the region under the Grad-CAM heatmap is shown at this size
DEFAULT_MAX_POINTS = 2000  # test images projected by default (same subset as the notebook)
DEFAULT_PERPLEXITY = 30.0  # t-SNE default (scikit-learn and the notebook)
DEFAULT_NEIGHBOURS = 15  # UMAP default (umap-learn and the notebook)
# Colours of the classification figure
CORRECT_COLOR = "#2e9e4f"  # prediction matches the true class
WRONG_COLOR = "#d1495b"  # prediction differs from the true class
NEUTRAL_COLOR = "#2a7ab0"  # true class unknown; also the "train" curves
TRUE_CLASS_COLOR = "#f0a202"  # the true class when the prediction is wrong; also "validation" curves


def _available_models() -> list[ModelPassport]:
    """The trained models in models/cnn; raises a ValueError explaining how to get one if there are none."""
    models, problems = list_models()
    if not models:
        raise ValueError(no_models_message(problems))
    return models


# --------------------------------------------------------------------------- results panel text


def _format_count(value: Any) -> str:
    """'45000' -> '45,000'; anything that is not a number is shown as it is."""
    return f"{value:,}" if isinstance(value, int) else str(value)


def passport_details(passport: ModelPassport) -> dict[str, object]:
    """The model passport as rows of the results panel (missing optional fields are left out)."""
    data = passport.data
    dataset = data.get("dataset", {})
    training = data.get("training", {})
    architecture = data.get("architecture", {})
    evaluation = data.get("evaluation", {})
    framework = data.get("framework", {})
    history = passport.history
    names = passport.class_names

    samples = [dataset.get(key) for key in ("train_samples", "validation_samples", "test_samples")]
    # Every row is None when its data is missing from the passport; those rows are dropped below
    rows: dict[str, object | None] = {
        "Model": passport.title,
        "Description": data.get("description"),
        "Model folder": display_path(passport.folder),
        "Created": data.get("created"),
        "Framework": ", ".join(f"{name} {version}" for name, version in framework.items()) or None,
        "Architecture": architecture.get("summary"),
        "Parameters": _format_count(architecture["parameters"]) if "parameters" in architecture else None,
        "Feature layer": (
            f"{architecture['feature_layer']} ({architecture.get('feature_size', '?')} values)"
            if "feature_layer" in architecture else None
        ),
        "Dataset": " — ".join(str(v) for v in (dataset.get("name"), dataset.get("source")) if v) or None,
        "Samples (train / validation / test)": (
            " / ".join(_format_count(v) if v is not None else "?" for v in samples) if any(samples) else None
        ),
        "Classes": f"{len(names)}: {', '.join(names)}",
        "Preprocessing": passport.preprocessing.describe(),
        "Optimizer": training.get("optimizer"),
        "Loss": training.get("loss"),
        "Epochs / batch size": (
            f"{training.get('epochs', '?')} / {training.get('batch_size', '?')}" if training else None
        ),
        "Training time": f"{training['seconds']:.0f} s" if "seconds" in training else None,
        "Final train accuracy": f"{history['accuracy'][-1]:.1%}" if history.get("accuracy") else None,
        "Final validation accuracy": f"{history['val_accuracy'][-1]:.1%}" if history.get("val_accuracy") else None,
        "Test accuracy": f"{passport.test_accuracy:.1%}" if passport.test_accuracy is not None else None,
        "Test loss": f"{evaluation['test_loss']:.3f}" if "test_loss" in evaluation else None,
    }
    return {key: value for key, value in rows.items() if value is not None}


def classification_details(
    result: Classification,
    passport: ModelPassport,
    image: Image.Image,
    source_name: str,
    true_class: str | None,
    gradcam_note: str | None = None,
) -> dict[str, object]:
    """Prediction first, then where the image came from and Grad-CAM, then the whole passport."""
    height, width = passport.preprocessing.input_size
    details: dict[str, object] = {
        "Prediction": f"{result.predicted} ({result.confidence:.1%})",
        "Ground truth": true_class if true_class is not None else "unknown (not a saved test image)",
    }
    if true_class is not None:
        details["Correct?"] = "Yes" if true_class == result.predicted else "No"
        details[f"p({true_class})"] = f"{result.probability(true_class):.1%}"
    details["Top 3"] = " · ".join(f"{name} {p:.0%}" for name, p in result.ranking[:3])
    details["Image"] = f"{source_name} ({image.width} × {image.height} px)"
    left, top, right, bottom = result.crop_box
    details["Network input"] = f"region ({left}, {top})–({right}, {bottom}) resized to {width} × {height} px"
    if gradcam_note is not None:
        details["Grad-CAM"] = gradcam_note
    details.update(passport_details(passport))
    return details


def projection_details(
    passport: ModelPassport,
    feature_set: FeatureSet,
    total: int,
    scores: dict[str, float | None],
    settings: dict[str, Any],
) -> dict[str, object]:
    """Numbers behind the projection figure, and how to read it."""
    labels, predictions = feature_set.labels, feature_set.predictions
    details: dict[str, object] = {
        "Model": passport.title,
        "Feature vectors": f"{len(feature_set):,} of {total:,} test images, {feature_set.features.shape[1]} values each",
        "Before projecting": "PCA to at most 50 dimensions (removes noise, speeds up t-SNE)",
    }
    for method, score in scores.items():
        if method == "tsne":
            details["t-SNE"] = f"perplexity {settings['perplexity']:g}, random seed 0"
        else:
            details["UMAP"] = f"n_neighbors {settings['n_neighbors']}, min_dist 0.1, random seed 0"
        if score is not None:
            details[f"Class separation {PROJECTION_METHODS[method]}"] = f"silhouette {score:.2f} (1 = compact, separate clusters)"
    if predictions is not None:
        wrong = int(np.sum(predictions != labels))
        details["Network accuracy on these images"] = f"{1 - wrong / len(labels):.1%} ({wrong} misclassified)"
    names = passport.class_names
    for index, name in enumerate(names):
        details[f"Points of {name}"] = int(np.sum(labels == index))
    details["Reading the plot"] = (
        "Each point is one test image, coloured by its true class. Images the network finds "
        "similar end up close together, so well-separated clusters mean the features "
        "distinguish those classes; overlapping colours show the classes it confuses."
    )
    return details


# --------------------------------------------------------------------------- figures


def class_colors(n: int) -> np.ndarray:
    """One distinct colour per class (RGBA rows): tab10 up to 10 classes, tab20 up to 20."""
    if n <= 10:
        return colormaps["tab10"](np.arange(n))
    if n <= 20:
        return colormaps["tab20"](np.arange(n))
    return colormaps["turbo"](np.linspace(0.05, 0.95, n))


def _no_data(ax, text: str) -> None:
    """Placeholder for a panel whose data is not in the passport."""
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, color="#666666", wrap=True)
    ax.set_axis_off()


def _plot_history(ax, passport: ModelPassport, metric: str, title: str) -> None:
    """Training and validation curve of one metric, with the test result as a dashed line."""
    history = passport.history
    if not history.get(metric):
        _no_data(ax, f"No '{metric}' history in the passport")
        return
    epochs = np.arange(1, len(history[metric]) + 1)
    ax.plot(epochs, history[metric], color=NEUTRAL_COLOR, label="train")
    if history.get(f"val_{metric}"):
        ax.plot(epochs[: len(history[f"val_{metric}"])], history[f"val_{metric}"], color=TRUE_CLASS_COLOR, label="validation")
    test_value = passport.data.get("evaluation", {}).get(f"test_{metric}")
    if test_value is not None:
        ax.axhline(test_value, color="#555555", linestyle="--", linewidth=1, label=f"test ({test_value:.3g})")
    # Transfer learning has two phases (frozen base, then fine-tuning): mark where each starts
    start = 0
    for phase in passport.data.get("training", {}).get("phases", [])[:-1]:
        start += int(phase.get("epochs", 0))
        ax.axvline(start + 0.5, color="#999999", linestyle=":", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _plot_confusion(ax, passport: ModelPassport) -> None:
    """Test confusion matrix of the passport, as the percentage of each true class (row)."""
    matrix = passport.confusion
    names = passport.class_names
    if matrix is None or matrix.shape != (len(names), len(names)):
        _no_data(ax, "No test confusion matrix in the passport")
        return
    rows = matrix.sum(axis=1, keepdims=True)
    # Each row as a percentage of its true class (an empty row stays 0 instead of dividing by 0)
    percent = np.divide(matrix, rows, out=np.zeros(matrix.shape, dtype=float), where=rows > 0) * 100
    ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    if len(names) <= 12:  # numbers stay readable up to about a dozen classes
        for (row, col), value in np.ndenumerate(percent):
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if value > 50 else "black")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title("Test confusion matrix (% of each true class)", fontsize=10)


def gradcam_overlay(image: Image.Image, box: tuple[int, int, int, int], heatmap: np.ndarray,
                    side: int = GRADCAM_SIDE, alpha: float = 0.45) -> Image.Image:
    """The region the network saw, enlarged to ``side`` px, with the Grad-CAM heatmap blended on top.

    The heatmap has the resolution of the convolutional layer (e.g. 30 x 30); it is
    stretched smoothly over the region (the same region the network input came from) and
    coloured with the 'jet' colour map: red = strongest evidence, blue = none.
    """
    region = to_rgb(image).crop(box)
    scale = side / max(region.size)  # enlarges a 32 x 32 test image, shrinks a big photo
    size = (max(1, round(region.width * scale)), max(1, round(region.height * scale)))
    region = region.resize(size, Image.Resampling.BILINEAR)
    # A float32 array becomes a 32-bit float ("F") image, which PIL can resize smoothly
    smooth = Image.fromarray(np.asarray(heatmap, dtype=np.float32)).resize(size, Image.Resampling.BILINEAR)
    colours = colormaps["jet"](np.clip(np.asarray(smooth), 0.0, 1.0))[..., :3] * 255.0  # RGBA -> RGB, 0..255
    blended = (1.0 - alpha) * np.asarray(region, dtype=np.float64) + alpha * colours  # alpha blending
    return Image.fromarray(np.clip(np.round(blended), 0, 255).astype(np.uint8))


def render_classification(
    image: Image.Image,
    result: Classification,
    passport: ModelPassport,
    true_class: str | None,
    heatmap: np.ndarray | None = None,
) -> Image.Image:
    """Top row: image, network input, Grad-CAM (if given), probabilities. Bottom row: the passport.

    The bottom row shows the passport's training history and test confusion matrix. The
    grid has 12 columns so that the top row can hold 3 or 4 panels and the bottom row 3.
    """
    fig = Figure(figsize=(14, 8.6), dpi=100)
    grid = fig.add_gridspec(2, 12, height_ratios=(1.0, 0.9), hspace=0.45, wspace=1.2)
    # one empty column before the probabilities leaves room for the class names
    panels = [grid[0, 0:4], grid[0, 4:7], grid[0, 8:12]] if heatmap is None else [
        grid[0, 0:3], grid[0, 3:5], grid[0, 5:8], grid[0, 9:12]]

    # 1. The image as given, with the region the network sees
    rgb = to_rgb(image)
    shrink = min(1.0, MAX_PLOT_SIDE / max(rgb.size))  # big photos are drawn smaller; the box is scaled too
    shown = rgb.resize((max(1, round(rgb.width * shrink)), max(1, round(rgb.height * shrink)))) if shrink < 1 else rgb
    ax = fig.add_subplot(panels[0])
    ax.imshow(shown)
    left, top, right, bottom = result.crop_box
    cropped = (right - left, bottom - top) != rgb.size  # no red box when the network sees the whole image
    if cropped:  # -0.5: imshow puts pixel centres on integer coordinates
        ax.add_patch(Rectangle((left * shrink - 0.5, top * shrink - 0.5), (right - left) * shrink, (bottom - top) * shrink,
                               fill=False, edgecolor="red", linewidth=2))
    ax.set_title(f"Image ({rgb.width} × {rgb.height} px)" + ("\nred box: region given to the network" if cropped else ""),
                 fontsize=10)
    ax.set_axis_off()

    # 2. What the network receives, after the passport's preprocessing (pixels shown as squares)
    height, width = passport.preprocessing.input_size
    ax = fig.add_subplot(panels[1])
    ax.imshow(input_to_image(result.network_input, passport.preprocessing), interpolation="nearest")
    ax.set_title(f"Network input ({width} × {height} px)", fontsize=10)
    ax.set_axis_off()

    # 2b. Grad-CAM: where the evidence for the predicted class is
    if heatmap is not None:
        ax = fig.add_subplot(panels[2])
        ax.imshow(gradcam_overlay(image, result.crop_box, heatmap))
        rows, cols = heatmap.shape
        ax.set_title(f"Grad-CAM for '{result.predicted}'\nred = strongest evidence ({cols} × {rows} map)", fontsize=10)
        ax.set_axis_off()

    # 3. Probability of every class, most likely on top
    ranking = result.ranking[:MAX_BARS][::-1]  # reversed: barh draws the first item at the bottom
    names = [name for name, _ in ranking]
    values = [p for _, p in ranking]
    colors = ["#c9d6e3"] * len(ranking)  # light grey for the other classes
    predicted_color = NEUTRAL_COLOR if true_class is None else (CORRECT_COLOR if true_class == result.predicted else WRONG_COLOR)
    colors[-1] = predicted_color  # the last bar (drawn on top) is the prediction
    if true_class is not None and true_class != result.predicted and true_class in names:
        colors[names.index(true_class)] = TRUE_CLASS_COLOR
    ax = fig.add_subplot(panels[-1])
    ax.barh(range(len(ranking)), values, color=colors)
    for y, value in enumerate(values):
        if value > 0.85:  # a long bar: write the value inside its end, in white
            ax.text(value - 0.02, y, f"{value:.1%}", va="center", ha="right", fontsize=8, color="white")
        else:  # otherwise just after the bar
            ax.text(value + 0.02, y, f"{value:.1%}", va="center", fontsize=8)
    labels = [f"{name}  (true)" if name == true_class else name for name in names]
    ax.set_yticks(range(len(ranking)), labels, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability (softmax output)")
    ax.set_title("Class probabilities", fontsize=10)

    # 4. The passport: how the model was trained and how it did on the test set
    _plot_history(fig.add_subplot(grid[1, 0:4]), passport, "loss", "Passport: loss per epoch")
    _plot_history(fig.add_subplot(grid[1, 4:8]), passport, "accuracy", "Passport: accuracy per epoch")
    _plot_confusion(fig.add_subplot(grid[1, 8:12]), passport)

    verdict = ""
    if true_class is not None:
        verdict = f"   |   true class: {true_class} ({'correct' if true_class == result.predicted else 'wrong'})"
    fig.suptitle(f"{passport.title}:  {result.predicted} ({result.confidence:.1%}){verdict}", fontsize=13,
                 color=predicted_color)
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.98)
    return figure_to_image(fig)


def render_projection(
    embeddings: dict[str, np.ndarray],
    feature_set: FeatureSet,
    passport: ModelPassport,
    scores: dict[str, float | None],
    mark_errors: bool,
) -> Image.Image:
    """One scatter plot per method, every test image coloured by its true class."""
    names = passport.class_names
    colors = class_colors(len(names))
    labels = feature_set.labels
    wrong = None  # boolean mask of the test images the network got wrong
    if mark_errors and feature_set.predictions is not None:
        wrong = feature_set.predictions != labels

    panels = len(embeddings)
    fig = Figure(figsize=(6.3 * panels + 1.9, 6.4), dpi=100)
    for index, (method, points) in enumerate(embeddings.items(), start=1):
        ax = fig.add_subplot(1, panels, index)
        for label, name in enumerate(names):
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], s=7, color=colors[label], alpha=0.75, label=name, linewidths=0)
        if wrong is not None and wrong.any():
            ax.scatter(points[wrong, 0], points[wrong, 1], s=9, marker="x", color="black", alpha=0.45,
                       linewidths=0.6, label="misclassified")
        method_name = PROJECTION_METHODS[method]
        score = scores.get(method)
        ax.set_title(method_name + (f"   (silhouette {score:.2f})" if score is not None else ""), fontsize=11)
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")
        ax.set_xticks([])  # t-SNE / UMAP coordinates have no meaning in themselves
        ax.set_yticks([])
    # One shared legend on the right (the classes are the same in every panel)
    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center right", fontsize=9, markerscale=2.5, frameon=False)
    fig.suptitle(
        f"{passport.title}: {len(feature_set):,} test images, {feature_set.features.shape[1]}-D features projected to 2-D",
        fontsize=12,
    )
    # Keep about 1.7 inches on the right for the legend, whatever the number of panels
    fig.subplots_adjust(left=0.04, right=1 - 1.7 / (6.3 * panels + 1.9), top=0.88, bottom=0.07, wspace=0.12)
    return figure_to_image(fig)


# --------------------------------------------------------------------------- dialogs


class CnnClassifyDialog(simpledialog.Dialog):
    """Choose the trained model and the image to classify."""

    def __init__(self, parent: tk.Misc, models: list[ModelPassport], current_name: str | None) -> None:
        """``current_name`` is the name of the image open in the app, or None when there is none."""
        self._models = models
        self._current_name = current_name
        self._samples: list[tuple[Path, str]] = []  # (file, true class) of the selected model
        # The choices, read by the tool after the dialog closes
        self.model: ModelPassport = models[0]
        self.source = "sample"  # "sample", "current" or "browse"
        self.sample: Path | None = None  # the chosen test image when source == "sample"
        self.gradcam = True  # compute and draw Grad-CAM
        self.user_confirmed = False  # stays False when the user cancels
        super().__init__(parent, "Classify an image with a trained CNN (Set 8.2)")

    def body(self, master: tk.Frame) -> tk.Widget:
        """Model selector, then the image source."""
        model_frame = ttk.LabelFrame(master, text="Trained model (models/cnn)", padding=8)
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._model_combo = ttk.Combobox(
            model_frame, values=[model.label for model in self._models], state="readonly", width=60
        )
        self._model_combo.current(0)
        self._model_combo.grid(row=0, column=0, sticky="w")
        self._model_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_model_changed())

        source_frame = ttk.LabelFrame(master, text="Image to classify", padding=8)
        source_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self._source_var = tk.StringVar(master, value="sample")
        ttk.Radiobutton(
            source_frame, text="Test image saved with the model (true class known):", value="sample",
            variable=self._source_var, command=self._on_source_changed,
        ).grid(row=0, column=0, sticky="w")
        self._sample_combo = ttk.Combobox(source_frame, state="readonly", width=48)
        self._sample_combo.grid(row=1, column=0, sticky="w", padx=18, pady=(2, 6))
        if self._current_name is not None:
            ttk.Radiobutton(
                source_frame, text=f"Current image in the app ({self._current_name})", value="current",
                variable=self._source_var, command=self._on_source_changed,
            ).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Radiobutton(
            source_frame, text="Open an image from disk…", value="browse",
            variable=self._source_var, command=self._on_source_changed,
        ).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(
            source_frame,
            text="Tip: a photo from Face Analysis › Webcam Photo Capture becomes the current image.\n"
                 "The network sees the centred region with its input shape (a square for CIFAR-10).",
            font="TkSmallCaptionFont", foreground="#555555", justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(6, 0))

        explain_frame = ttk.LabelFrame(master, text="Explanation", padding=8)
        explain_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self._gradcam_var = tk.BooleanVar(master, value=True)
        ttk.Checkbutton(
            explain_frame, text="Show Grad-CAM: where in the image the network found the predicted class",
            variable=self._gradcam_var,
        ).grid(row=0, column=0, sticky="w")

        self._on_model_changed()
        return self._model_combo

    def _on_model_changed(self) -> None:
        """Fill the test-image list with the samples saved next to the selected model."""
        self.model = self._models[self._model_combo.current()]
        self._samples = list_test_samples(self.model)
        self._sample_combo.configure(values=[f"{label} — {path.name}" for path, label in self._samples])
        if self._samples:
            self._sample_combo.current(0)
        else:
            self._sample_combo.set("(no test_samples folder for this model)")
            if self._source_var.get() == "sample":
                self._source_var.set("current" if self._current_name is not None else "browse")
        self._on_source_changed()

    def _on_source_changed(self) -> None:
        """The test-image list is only active when that source is selected."""
        active = self._source_var.get() == "sample" and bool(self._samples)
        self._sample_combo.configure(state="readonly" if active else "disabled")

    def validate(self) -> bool:
        """Store the choices; a saved test image must exist when that source is chosen."""
        self.source = self._source_var.get()
        self.gradcam = self._gradcam_var.get()
        if self.source == "sample":
            if not self._samples:
                messagebox.showerror("No test images", "This model has no saved test images.", parent=self)
                return False
            self.sample = self._samples[self._sample_combo.current()][0]
        return True

    def apply(self) -> None:
        """Called by simpledialog after a successful validate(): the user pressed OK."""
        self.user_confirmed = True


class CnnProjectionDialog(simpledialog.Dialog):
    """Choose the model whose test features are projected, the method and its parameters."""

    def __init__(self, parent: tk.Misc, models: list[ModelPassport], umap_installed: bool) -> None:
        """``models`` are the models with saved test features; UMAP is offered only if installed."""
        self._models = models
        self._umap_installed = umap_installed
        # The choices, read by the tool after the dialog closes
        self.model: ModelPassport = models[0]
        self.methods: list[str] = ["tsne"]
        self.max_points = DEFAULT_MAX_POINTS
        self.perplexity = DEFAULT_PERPLEXITY
        self.n_neighbors = DEFAULT_NEIGHBOURS
        self.mark_errors = True
        self.user_confirmed = False
        super().__init__(parent, "Feature projection of the test set (Set 8.3)")

    def body(self, master: tk.Frame) -> tk.Widget:
        """Model, method, and the parameters of each method."""
        model_frame = ttk.LabelFrame(master, text="Trained model with saved test features", padding=8)
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._model_combo = ttk.Combobox(
            model_frame, values=[model.label for model in self._models], state="readonly", width=60
        )
        self._model_combo.current(0)
        self._model_combo.grid(row=0, column=0, sticky="w")

        method_frame = ttk.LabelFrame(master, text="Projection method", padding=8)
        method_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self._method_var = tk.StringVar(master, value="both" if self._umap_installed else "tsne")
        umap_state = "normal" if self._umap_installed else "disabled"
        ttk.Radiobutton(method_frame, text="t-SNE", value="tsne", variable=self._method_var).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(method_frame, text="UMAP", value="umap", variable=self._method_var,
                        state=umap_state).grid(row=0, column=1, sticky="w", padx=12)
        ttk.Radiobutton(method_frame, text="Both, side by side", value="both", variable=self._method_var,
                        state=umap_state).grid(row=0, column=2, sticky="w")
        if not self._umap_installed:
            ttk.Label(method_frame, text="UMAP needs the optional package umap-learn (requirements-keras.txt).",
                      font="TkSmallCaptionFont", foreground="#555555").grid(row=1, column=0, columnspan=3, sticky="w")

        settings = ttk.LabelFrame(master, text="Parameters", padding=8)
        settings.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self._points_var = tk.StringVar(master, value=str(DEFAULT_MAX_POINTS))
        self._perplexity_var = tk.StringVar(master, value=f"{DEFAULT_PERPLEXITY:g}")
        self._neighbors_var = tk.StringVar(master, value=str(DEFAULT_NEIGHBOURS))
        self._mark_var = tk.BooleanVar(master, value=True)
        rows = (
            ("Test images to project (random subset):", self._points_var),
            ("t-SNE perplexity (≈ neighbours per point):", self._perplexity_var),
            ("UMAP n_neighbors:", self._neighbors_var),
        )
        for row, (text, variable) in enumerate(rows):
            ttk.Label(settings, text=text).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(settings, textvariable=variable, width=10).grid(row=row, column=1, sticky="w", padx=(8, 0))
        ttk.Checkbutton(settings, text="Mark the test images the network misclassified (×)",
                        variable=self._mark_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        return self._model_combo

    def validate(self) -> bool:
        """Check the numbers and store every choice."""
        try:
            max_points = int(self._points_var.get())
            perplexity = float(self._perplexity_var.get())
            n_neighbors = int(self._neighbors_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Enter whole numbers (perplexity may be decimal).", parent=self)
            return False
        if not 50 <= max_points <= 20000:
            messagebox.showerror("Invalid value", "Use between 50 and 20000 test images.", parent=self)
            return False
        if not 2 <= perplexity <= 100 or not 2 <= n_neighbors <= 200:
            messagebox.showerror("Invalid value", "Perplexity must be 2–100 and n_neighbors 2–200.", parent=self)
            return False
        self.model = self._models[self._model_combo.current()]
        method = self._method_var.get()
        self.methods = ["tsne", "umap"] if method == "both" else [method]
        self.max_points, self.perplexity, self.n_neighbors = max_points, perplexity, n_neighbors
        self.mark_errors = self._mark_var.get()
        return True

    def apply(self) -> None:
        """Called by simpledialog after a successful validate(): the user pressed OK."""
        self.user_confirmed = True


# --------------------------------------------------------------------------- tools


class CnnClassifyTool(ForensicsTool):
    """Set 8.2: load a trained CNN, show its passport and classify one image."""

    tool_id = "cnn_classify"
    title = "CNN: Classify Image (Set 8.2)"
    category = CATEGORY
    description = (
        "Loads a CNN trained in the Set 8 notebook, applies the preprocessing from its passport "
        "and shows the class probabilities, a Grad-CAM heatmap and the model passport."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Ask for the model and image, classify it on a worker thread and draw the result."""
        # Fail early, before the dialog, when there is nothing to load or no way to load it
        models = _available_models()
        if not keras_available():
            raise ValueError(KERAS_INSTALL_HINT)
        current_name = None  # offered as a source only when an image is open in the app
        if document.current is not None:
            current_name = document.path.name if document.path is not None else "unsaved image"
        dialog = CnnClassifyDialog(parent, models, current_name)
        if not dialog.user_confirmed:
            return None
        passport = dialog.model

        # 1. The image, and its true class when it is a test image saved by the notebook
        if dialog.source == "sample":
            path = dialog.sample
            with Image.open(path) as opened:
                image = opened.copy()
        elif dialog.source == "browse":
            loaded = load_image_file(parent, title="Open an image to classify")
            if loaded is None:
                return None
            path, image = loaded
        else:
            path, image = document.path, document.current
        true_class = true_class_from_path(path, passport.class_names)
        source_name = path.name if path is not None else "current image"

        # 2. Loading TensorFlow and the network takes seconds: do it behind a progress window
        # It runs on a worker thread: it must not touch Tkinter, only report progress
        def task(report, cancelled) -> tuple[Classification, np.ndarray | None, str | None]:
            steps = 3 if dialog.gradcam else 2
            report(0, steps, "Loading the trained network (TensorFlow is slow to start the first time)…")
            network = load_network(passport)
            if cancelled():
                raise TaskCancelled()
            report(1, steps, "Classifying the image…")
            result = classify(image, passport, network)
            heatmap, note = None, None
            if dialog.gradcam:
                if cancelled():
                    raise TaskCancelled()
                report(2, steps, "Computing Grad-CAM…")
                try:
                    heatmap = explain(network, result, passport)
                    rows, cols = heatmap.shape
                    layer = getattr(network, "gradcam_layer", "?")
                    note = f"class '{result.predicted}', layer '{layer}' ({cols} × {rows} maps); red = strongest evidence"
                except Exception as error:  # the explanation is extra: it must never hide the prediction
                    note = f"not available ({error})"
            return result, heatmap, note

        try:
            result, heatmap, gradcam_note = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None

        verdict = ""
        if true_class is not None:
            verdict = f" True class: {true_class} ({'correct' if true_class == result.predicted else 'wrong'})."
        return ToolResult(
            message=f"{passport.title}: {source_name} is '{result.predicted}' ({result.confidence:.1%}).{verdict}",
            image=render_classification(image, result, passport, true_class, heatmap),
            details=classification_details(result, passport, image, source_name, true_class, gradcam_note),
            preview_only=True,  # an analysis figure: the working image is left as it was
        )


class CnnProjectionTool(ForensicsTool):
    """Set 8.3: project the saved test-set feature vectors onto 2-D with t-SNE and/or UMAP."""

    tool_id = "cnn_projection"
    title = "CNN: Feature Projection t-SNE / UMAP (Set 8.3)"
    category = CATEGORY
    description = (
        "Loads the test-set feature vectors saved by the Set 8 notebook and shows them in 2-D "
        "with t-SNE and UMAP, one colour per class."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Ask for the model and method, project on a worker thread and draw the scatter plots."""
        # Only models whose folder has test_features.npz can be projected
        models = [model for model in _available_models() if model.features_path.exists()]
        if not models:
            raise ValueError(
                "None of the trained models has saved test features (test_features.npz). Run the "
                "Set 8.3 section of the notebook and copy the model folder again."
            )
        dialog = CnnProjectionDialog(parent, models, umap_available())
        if not dialog.user_confirmed:
            return None
        passport = dialog.model
        methods = dialog.methods

        # Worker thread: load the features, then one projection per chosen method
        def task(report, cancelled) -> tuple[int, FeatureSet, dict[str, np.ndarray]]:
            steps = len(methods) + 1
            report(0, steps, "Loading the saved test features…")
            features = load_test_features(passport)
            sample = subsample(features, dialog.max_points)  # t-SNE is slow with many points
            embeddings = {}
            for step, method in enumerate(methods, start=1):
                if cancelled():
                    raise TaskCancelled()
                report(step, steps, f"Projecting {len(sample):,} feature vectors with {PROJECTION_METHODS[method]}…")
                embeddings[method] = project(
                    sample.features, method, perplexity=dialog.perplexity, n_neighbors=dialog.n_neighbors
                )
            return len(features), sample, embeddings

        try:
            total, sample, embeddings = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None

        # One number per projection: how well the classes separate (silhouette)
        scores = {method: separation_score(points, sample.labels) for method, points in embeddings.items()}
        settings = {"perplexity": dialog.perplexity, "n_neighbors": dialog.n_neighbors}
        names = " and ".join(PROJECTION_METHODS[method] for method in methods)
        return ToolResult(
            message=f"{passport.title}: {len(sample):,} test feature vectors projected with {names}.",
            image=render_projection(embeddings, sample, passport, scores, dialog.mark_errors),
            details=projection_details(passport, sample, total, scores, settings),
            preview_only=True,  # an analysis figure: the working image is left as it was
        )
