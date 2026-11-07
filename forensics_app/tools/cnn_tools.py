"""Set 7 bonus tool: compare the BoW model with CNN (ResNet-18) features on the same split."""

from __future__ import annotations

import tkinter as tk

from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_model import classify, describe_image, get_model
from .bow_render import figure_to_image
from .bow_tools import CATEGORY
from .cnn_features import (
    INSTALL_HINT,
    Comparison,
    EmbedFn,
    classify_with_cnn,
    compare_with_cnn,
    load_resnet18,
    torch_available,
)
from .dialogs import TaskCancelled, run_with_progress

MAX_DISAGREEMENTS = 15
IMAGENET_NOTE = (
    "ImageNet (the CNN's pre-training data) already contains dalmatians, butterflies and "
    "wallabies, so the CNN starts with very relevant knowledge (transfer learning)."
)


def _confusion_panel(ax, matrix: np.ndarray, classes: list[str], title: str, cmap: str) -> None:
    ax.imshow(matrix, cmap=cmap)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", fontsize=11,
                color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)


def render_comparison(comparison: Comparison) -> Image.Image:
    """Both confusion matrices on the same test images, and the recall of every class."""
    classes = comparison.classes
    bow_acc = comparison.bow_metrics["accuracy"]
    cnn_acc = comparison.cnn_metrics["accuracy"]
    fig = Figure(figsize=(14, 4.6), dpi=100)
    fig.suptitle(
        f"Same {len(comparison.test_paths)} test images:  hand-crafted BoW {bow_acc:.1%}  vs  "
        f"CNN features (ResNet-18) {cnn_acc:.1%}",
        fontsize=12,
    )
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.4), wspace=0.45)
    _confusion_panel(fig.add_subplot(grid[0]), np.asarray(comparison.bow_metrics["confusion"]), classes,
                     f"BoW (SIFT + K-Means + chi2 SVM)\n{comparison.bow_dimensions} values per image", "Blues")
    _confusion_panel(fig.add_subplot(grid[1]), np.asarray(comparison.cnn_metrics["confusion"]), classes,
                     f"CNN (ResNet-18 + linear SVM)\n{comparison.cnn_dimensions} values per image", "Oranges")
    ax = fig.add_subplot(grid[2])
    x = np.arange(len(classes))
    bow = [comparison.bow_metrics["per_class"][name][1] for name in classes]
    cnn = [comparison.cnn_metrics["per_class"][name][1] for name in classes]
    ax.bar(x - 0.2, bow, 0.4, label=f"BoW ({bow_acc:.0%})", color="#2a7ab0")
    ax.bar(x + 0.2, cnn, 0.4, label=f"CNN ({cnn_acc:.0%})", color="#e07b28")
    for offset, values in ((-0.2, bow), (0.2, cnn)):
        for xi, value in zip(x, values):
            ax.text(xi + offset, value + 0.02, f"{value:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, classes, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall (test)")
    ax.set_title("Correctly recognised, per class", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.06, right=0.98)
    return figure_to_image(fig)


def comparison_details(comparison: Comparison) -> dict[str, object]:
    bow, cnn = comparison.bow_metrics, comparison.cnn_metrics
    right_bow = comparison.bow_predicted == comparison.test_labels
    right_cnn = comparison.cnn_predicted == comparison.test_labels
    details: dict[str, object] = {
        "Operation": "BoW vs CNN features (same train/test split)",
        "Test images": len(comparison.test_paths),
        "BoW test accuracy": f"{bow['accuracy']:.1%}",
        "CNN test accuracy": f"{cnn['accuracy']:.1%}",
        "BoW cross-validation (train)": f"{comparison.bow_cv_accuracy:.1%}",
        "CNN cross-validation (train)": f"{comparison.cnn_cv_accuracy:.1%}",
        "Both right": int(np.sum(right_bow & right_cnn)),
        "Only BoW right": int(np.sum(right_bow & ~right_cnn)),
        "Only CNN right": int(np.sum(~right_bow & right_cnn)),
        "Both wrong": int(np.sum(~right_bow & ~right_cnn)),
    }
    for name in comparison.classes:
        details[f"Recall {name}"] = f"BoW {bow['per_class'][name][1]:.0%} | CNN {cnn['per_class'][name][1]:.0%}"
    details["BoW representation"] = f"{comparison.bow_dimensions} values (visual-word histogram with pyramid)"
    details["CNN representation"] = f"{comparison.cnn_dimensions} values (ResNet-18 global average pool)"
    details["CNN classifier"] = f"linear SVM on L2-normalised features, C = {comparison.cnn_c:g} (by CV)"
    details["CNN time"] = f"{comparison.cnn_seconds:.1f} s (features + training)"
    for index, (name, true, bow_label, cnn_label) in enumerate(comparison.disagreements()[:MAX_DISAGREEMENTS], start=1):
        details[f"Disagreement #{index}"] = f"{name}: true {true} | BoW {bow_label} | CNN {cnn_label}"
    details["Note"] = IMAGENET_NOTE
    return details


def current_image_details(document: ImageDocument, comparison: Comparison, embed_fn: EmbedFn) -> dict[str, object]:
    """Both predictions for the open image, so the two methods can be compared on it."""
    assert document.current is not None
    model = get_model()
    analysis = describe_image(model, document.current, document.path)
    if len(analysis.features):
        best, p = classify(model, analysis)[0]
        bow_text = f"p({best}) = {p:.2f}"
    else:
        bow_text = "no local features, cannot classify"
    best, p = classify_with_cnn(comparison.classifier, embed_fn, document.current, comparison.classes)[0]
    name = document.path.name if document.path is not None else "current image"
    return {"Current image": f"{name} ({analysis.split})", "Current image BoW": bow_text, "Current image CNN": f"p({best}) = {p:.2f}"}


class CnnCompareTool(ForensicsTool):
    tool_id = "cnn_compare"
    title = "Bonus: Compare BoW vs CNN"
    category = CATEGORY
    description = (
        "Trains an SVM on ResNet-18 features with the same train/test split as the BoW model "
        "and compares both on the test images (hand-crafted vs deep features)."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if not torch_available():
            raise ValueError(INSTALL_HINT)
        model = get_model()

        def task(report, cancelled):
            report(0, 1, "Loading ResNet-18 (about 45 MB are downloaded the first time)")
            embed_fn = load_resnet18()
            return embed_fn, compare_with_cnn(model, embed_fn, report, cancelled)

        try:
            embed_fn, comparison = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
        details = comparison_details(comparison)
        if document.current is not None:
            details.update(current_image_details(document, comparison, embed_fn))
        bow_acc = comparison.bow_metrics["accuracy"]
        cnn_acc = comparison.cnn_metrics["accuracy"]
        return ToolResult(
            message=f"Same {len(comparison.test_paths)} test images: BoW {bow_acc:.1%} vs CNN {cnn_acc:.1%}.",
            image=render_comparison(comparison),
            details=details,
            preview_only=True,
        )
