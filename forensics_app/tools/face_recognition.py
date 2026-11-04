"""Face recognition tool corresponding to Set 6.3.

Loads the saved Eigenfaces model, computes projection weights w for train and test images,
finds the m nearest neighbors using Euclidean distance, and classifies the identity
using Majority Voting, Distance-Weighted Voting, or Class Centroids.
Also supports FisherFaces (LDA dimensionality reduction).
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import load_image_file
from .face_recognition_core import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    FaceModel,
    classify_face,
    load_face_model,
    make_recognition_gallery,
    prepare_lfw_dataset,
    project_face_image,
    train_eigenfaces_model,
)

STRATEGY_OPTIONS = {
    "majority": "Majority voting (m-NN)",
    "distance_weighted": "Distance-weighted voting (1/d)",
    "centroid": "Class centroids (Distance to class mean)",
}
STRATEGY_LABELS = {v: k for k, v in STRATEGY_OPTIONS.items()}


class FaceRecognitionDialog(simpledialog.Dialog):
    """Menu dialog configuring face recognition parameters, styled after matching_dialog.py."""

    def __init__(
        self,
        parent: tk.Misc,
        has_current_image: bool,
        test_images: list[dict[str, Any]],
        max_components: int = 100,
    ) -> None:
        self._has_current_image = has_current_image
        self._test_images = test_images
        self._max_components = max_components

        # Default settings
        self.method: str = "eigenfaces"  # 'eigenfaces' or 'fisherfaces'
        self.k: int = 30
        self.m: int = 3
        self.strategy: str = "majority"
        self.source: str = "test_set" if test_images else ("current" if has_current_image else "browse")
        self.selected_test_file: str | None = test_images[0]["path"] if test_images else None
        self.user_confirmed = False

        super().__init__(parent, "Face Recognition Settings (Set 6.3)")

    def body(self, master: tk.Frame) -> tk.Widget:
        # 1. Feature Representation Frame
        subspace_frame = ttk.LabelFrame(master, text="Feature Representation", padding=8)
        subspace_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self._method_var = tk.StringVar(master, value="eigenfaces")
        ttk.Radiobutton(
            subspace_frame,
            text="EigenFaces (PCA basis)",
            value="eigenfaces",
            variable=self._method_var,
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Radiobutton(
            subspace_frame,
            text="FisherFaces (PCA + LDA subspace)",
            value="fisherfaces",
            variable=self._method_var,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(subspace_frame, text=f"Number of eigenfaces (k ≤ {self._max_components}):").grid(
            row=1, column=0, sticky="w", pady=(6, 2)
        )
        self._k_var = tk.StringVar(master, value=str(min(30, self._max_components)))
        self._k_entry = ttk.Entry(subspace_frame, textvariable=self._k_var, width=10)
        self._k_entry.grid(row=1, column=1, sticky="w", pady=(6, 2))

        ttk.Label(
            subspace_frame,
            text=f"* k is bounded by the {self._max_components} components saved in the Eigenfaces Builder step.",
            font="TkSmallCaptionFont",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(1, 2))

        ttk.Label(
            subspace_frame,
            text="* FisherFaces (LDA): subspace dimension is automatically fixed to C - 1 = 6 (no tuning needed).",
            font="TkSmallCaptionFont",
            foreground="#2b5797",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # 2. Matching & Classification Frame
        matching_frame = ttk.LabelFrame(master, text="Matching & Classification Strategy", padding=8)
        matching_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(matching_frame, text="Closest training faces to display (m):").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self._m_var = tk.StringVar(master, value="3")
        self._m_entry = ttk.Entry(matching_frame, textvariable=self._m_var, width=10)
        self._m_entry.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(matching_frame, text="Decision strategy:").grid(
            row=1, column=0, sticky="w", pady=(6, 2)
        )
        self._strategy_cb = ttk.Combobox(
            matching_frame,
            values=list(STRATEGY_OPTIONS.values()),
            state="readonly",
            width=36,
        )
        self._strategy_cb.current(0)
        self._strategy_cb.grid(row=1, column=1, sticky="w", pady=(6, 2))

        # 3. Test Image Source Frame
        source_frame = ttk.LabelFrame(master, text="Test Image Source", padding=8)
        source_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self._source_var = tk.StringVar(master, value=self.source)

        row_idx = 0
        if self._test_images:
            ttk.Radiobutton(
                source_frame,
                text="Pick from saved test folder (LFW test set):",
                value="test_set",
                variable=self._source_var,
                command=self._on_source_changed,
            ).grid(row=row_idx, column=0, columnspan=2, sticky="w")
            row_idx += 1

            # Dropdown with sample test images
            test_display_names = [
                f"{idx + 1}. {item['name']} ({Path(item['path']).name})"
                for idx, item in enumerate(self._test_images[:40])
            ]
            self._test_combo = ttk.Combobox(
                source_frame, values=test_display_names, state="readonly", width=42
            )
            self._test_combo.current(0)
            self._test_combo.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=18, pady=(2, 6))
            row_idx += 1

        ttk.Radiobutton(
            source_frame,
            text="Open image from disk (file chooser)...",
            value="browse",
            variable=self._source_var,
            command=self._on_source_changed,
        ).grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=2)
        row_idx += 1

        if self._has_current_image:
            ttk.Radiobutton(
                source_frame,
                text="Use current working image in document",
                value="current",
                variable=self._source_var,
                command=self._on_source_changed,
            ).grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=2)

        self._on_source_changed()
        return self._k_entry

    def _on_source_changed(self) -> None:
        if hasattr(self, "_test_combo"):
            is_test_set = self._source_var.get() == "test_set"
            self._test_combo.configure(state="readonly" if is_test_set else "disabled")

    def validate(self) -> bool:
        try:
            k = int(self._k_var.get().strip())
            m = int(self._m_var.get().strip())
            if k < 1 or k > self._max_components:
                raise ValueError(f"k must be between 1 and {self._max_components}.")
            if m < 1 or m > 10:
                raise ValueError("m must be between 1 and 10.")
            self.k = k
            self.m = m
            self.method = self._method_var.get()
            self.strategy = STRATEGY_LABELS[self._strategy_cb.get()]
            self.source = self._source_var.get()

            if self.source == "test_set" and hasattr(self, "_test_combo"):
                selected_idx = self._test_combo.current()
                if 0 <= selected_idx < len(self._test_images):
                    self.selected_test_file = self._test_images[selected_idx]["path"]

            return True
        except ValueError as err:
            messagebox.showerror("Invalid Setting", str(err), parent=self)
            return False

    def apply(self) -> None:
        self.user_confirmed = True


class FaceRecognitionTool(ForensicsTool):
    """Face recognition tool (Set 6.3) using Eigenfaces (PCA) and Fisherfaces (LDA)."""

    tool_id = "face_recognition"
    title = "Face Recognition"
    category = "Face Analysis"
    description = "Classify faces by projecting onto eigenfaces (w vectors) and computing Euclidean distances."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        # 1. Ensure dataset and model exist
        meta = prepare_lfw_dataset(DEFAULT_DATA_DIR)
        model = load_face_model(DEFAULT_MODEL_DIR)
        if model is None:
            # Auto-train default model if absent
            model = train_eigenfaces_model(
                n_components=100, data_dir=DEFAULT_DATA_DIR, model_dir=DEFAULT_MODEL_DIR
            )

        test_items = meta.get("test", [])
        dialog = FaceRecognitionDialog(
            parent=parent,
            has_current_image=document.is_loaded,
            test_images=test_items,
            max_components=len(model.eigenfaces),
        )

        if not dialog.user_confirmed:
            return None

        k = dialog.k
        m = dialog.m
        strategy = dialog.strategy
        use_lda = dialog.method == "fisherfaces"
        source = dialog.source

        # 2. Acquire test image and ground truth (if known)
        true_name: str | None = None
        source_path: Path | None = None

        if source == "test_set" and dialog.selected_test_file:
            source_path = Path(dialog.selected_test_file)
            with Image.open(source_path) as img:
                test_image = img.copy()
            # Extract ground truth name from filename or metadata
            for item in test_items:
                if Path(item["path"]).resolve() == source_path.resolve():
                    true_name = item["name"]
                    break

        elif source == "browse":
            loaded = load_image_file(parent, title="Open Test Face Image")
            if loaded is None:
                return None
            source_path, test_image = loaded
            # Check if user picked an image from the test set folder
            for item in test_items:
                if Path(item["path"]).name == source_path.name:
                    true_name = item["name"]
                    break

        else:
            if not document.is_loaded or document.current is None:
                messagebox.showinfo("No image", "Open an image first.", parent=parent)
                return None
            test_image = document.current
            if document.path:
                source_path = document.path
                for item in test_items:
                    if Path(item["path"]).name == source_path.name:
                        true_name = item["name"]
                        break

        # 3. Project test image onto the subspace to obtain vector w
        w_test = project_face_image(test_image, model, k=k)

        # 4. Classify test image and retrieve top m closest training images
        res = classify_face(
            test_w=w_test,
            model=model,
            k=k,
            m=m,
            strategy=strategy,
            use_lda=use_lda,
        )

        pred_name = res["predicted_name"]
        top_matches = res["top_matches"]
        strategy_info = res["strategy_info"]

        # 5. Build side-by-side gallery visual output
        gallery_image = make_recognition_gallery(
            test_image=test_image,
            top_matches=top_matches,
            predicted_name=pred_name,
            strategy=strategy_info["type"],
            true_name=true_name,
        )

        # 6. Prepare detailed results dictionary
        is_match = (true_name is not None) and (pred_name == true_name)
        verdict_str = (
            "CORRECT MATCH"
            if is_match
            else ("MISMATCH" if true_name is not None else "Unknown Ground Truth")
        )

        method_str = "FisherFaces (PCA + LDA)" if use_lda else "EigenFaces (PCA)"
        k_var = float(np.sum(model.explained_variance_ratio[:k]) * 100.0)
        details: dict[str, Any] = {
            "Predicted Identity": pred_name,
            "Classification Result": verdict_str,
            "Representation": method_str,
            "Eigenfaces Used (k)": f"{k} ({k_var:.2f}% explained variance)",
            "Model Components Available": len(model.eigenfaces),
            "Neighbors Examined (m)": m,
            "Decision Rule": strategy_info["type"],
        }
        if true_name is not None:
            details["Ground Truth Name"] = true_name
            details["Match Correct?"] = "Yes" if is_match else "No"

        for match in top_matches:
            r = match["rank"]
            d_val = match["distance"]
            details[f"Top-{r} Match"] = f"{match['name']} (distance = {d_val:.2f})"

        min_d = top_matches[0]["distance"]
        status_msg = (
            f"Recognized as '{pred_name}' [{method_str}, {strategy_info['type']}, min d={min_d:.2f}]. "
            f"Result: {verdict_str}."
        )

        # Also initialize the working image if empty so user can inspect
        if document.current is None:
            document.original = gallery_image.copy()
            document.current = gallery_image.copy()

        return ToolResult(
            message=status_msg,
            image=gallery_image,
            details=details,
            preview_only=True,
        )
