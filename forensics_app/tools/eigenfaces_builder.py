"""Eigenface builder tool corresponding to Set 6.2.

Trains the PCA model on the face dataset, exports global and per-class mean faces
to disk, and visualizes the first K eigenfaces in a grid format.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np

from forensics_app.core import ImageDocument
from forensics_app.paths import display_path
from .base import ForensicsTool, ToolResult
from .face_recognition_core import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    FaceModel,
    load_face_model,
    make_eigenfaces_grid,
    prepare_lfw_dataset,
    train_eigenfaces_model,
)


class EigenfacesDialog(simpledialog.Dialog):
    """Configuration dialog for the Eigenface Builder tool."""

    def __init__(
        self,
        parent: tk.Misc,
        model_exists: bool,
        existing_components: int | None = None,
        existing_variance: float | None = None,
    ) -> None:
        """Show the builder options, pre-filled from the model on disk if there is one."""
        self._model_exists = model_exists
        self._existing_components = existing_components
        self._existing_variance = existing_variance
        self.k: int = 16
        # If an existing model is on disk, suggest its component count; otherwise default to 150
        self.n_components: int = existing_components if existing_components is not None else 150
        self.retrain: bool = not model_exists
        self.user_confirmed = False
        super().__init__(parent, "Eigenface Builder (Set 6.2)")

    def body(self, master: tk.Frame) -> tk.Widget:
        """Build the model-status banner and the parameter fields."""
        # Information banner
        if self._existing_components is not None:
            status_text = (
                f"Model found on disk: {self._existing_components} components "
                f"({self._existing_variance:.1f}% total variance stored)."
            )
        elif self._model_exists:
            status_text = "Model found on disk (ready to visualize)."
        else:
            status_text = "No model found on disk. It will be trained on the LFW face dataset."

        info_frame = ttk.LabelFrame(master, text="Dataset & Model Status", padding=8)
        info_frame.pack(fill="x", padx=6, pady=6)
        ttk.Label(info_frame, text=status_text, foreground="#2b5797").pack(anchor="w")
        ttk.Label(
            info_frame,
            text=f"Mean faces will be saved to: {display_path(DEFAULT_MODEL_DIR / 'mean_faces')}",
            font="TkSmallCaptionFont",
            foreground="#555555",
        ).pack(anchor="w", pady=(4, 0))

        # Parameters frame
        params_frame = ttk.LabelFrame(master, text="Parameters", padding=8)
        params_frame.pack(fill="x", padx=6, pady=6)

        ttk.Label(params_frame, text="Eigenfaces to visualize (K):").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self._k_var = tk.StringVar(master, value="16")
        self._k_entry = ttk.Entry(params_frame, textvariable=self._k_var, width=10)
        self._k_entry.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(params_frame, text="Total PCA components:").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self._comp_var = tk.StringVar(master, value=str(self.n_components))
        self._comp_entry = ttk.Entry(params_frame, textvariable=self._comp_var, width=10)
        self._comp_entry.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(
            params_frame,
            text="* Max theoretical components: 965 (100% variance). 100 comps ≈ 91.3%, 150 ≈ 94.5%, 383 ≈ 99%.",
            font="TkSmallCaptionFont",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(1, 4))

        self._retrain_var = tk.BooleanVar(master, value=self.retrain)
        ttk.Checkbutton(
            params_frame,
            text="Force re-train model and re-save mean faces",
            variable=self._retrain_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 2))

        return self._k_entry

    def validate(self) -> bool:
        """Check K and the component count; show an error and return False if invalid."""
        try:
            k = int(self._k_var.get().strip())
            comp = int(self._comp_var.get().strip())
            max_possible = 965  # N_train - 1 (for 966 training images)
            if k < 1 or k > 200:
                raise ValueError("K must be between 1 and 200.")
            if comp < 10 or comp > max_possible:
                raise ValueError(
                    f"Total PCA components must be between 10 and {max_possible} "
                    f"(the maximum rank of the centered training matrix)."
                )
            if k > comp:
                raise ValueError(f"K ({k}) cannot be greater than total components ({comp}).")
            self.k = k
            self.n_components = comp
            self.retrain = bool(self._retrain_var.get())
            return True
        except ValueError as err:
            messagebox.showerror("Invalid Input", str(err), parent=self)
            return False

    def apply(self) -> None:
        """Mark the dialog as confirmed."""
        self.user_confirmed = True


class EigenfacesBuilderTool(ForensicsTool):
    """Tool to build the Eigenfaces model and visualize the top K eigenfaces."""

    tool_id = "eigenfaces_builder"
    title = "Eigenface Builder"
    category = "Face Analysis"
    description = "Train PCA eigenfaces model, export mean faces, and view the top K eigenfaces."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Load or train the Eigenfaces model and return a grid of the top K eigenfaces."""
        # Check whether a trained model currently exists
        existing_model = load_face_model(DEFAULT_MODEL_DIR)
        existing_comps = len(existing_model.eigenfaces) if existing_model else None
        existing_var = (
            float(np.sum(existing_model.explained_variance_ratio) * 100.0)
            if existing_model
            else None
        )

        dialog = EigenfacesDialog(
            parent,
            model_exists=(existing_model is not None),
            existing_components=existing_comps,
            existing_variance=existing_var,
        )

        if not dialog.user_confirmed:
            return None

        k = dialog.k
        n_comp = dialog.n_components
        retrain = dialog.retrain

        # Train or load model:
        # Retrain if user checked Force re-train, if no model exists, OR if the stored model
        # has fewer components than requested (e.g. 100 on disk, but 150 requested).
        need_retrain = retrain or existing_model is None or len(existing_model.eigenfaces) < n_comp
        if need_retrain:
            model = train_eigenfaces_model(
                n_components=n_comp,
                data_dir=DEFAULT_DATA_DIR,
                model_dir=DEFAULT_MODEL_DIR,
            )
        else:
            model = existing_model

        # Build square grid visualization of the first K eigenfaces
        grid_image = make_eigenfaces_grid(model.eigenfaces, k=k)

        # Distinguish the three variance metrics:
        # 1. Top K: the variance of the eigenfaces currently displayed on screen.
        # 2. Active Subspace: the variance of the n_components requested by the user.
        # 3. Total Stored: the overall variance of all components stored in the model on disk.
        active_n = min(n_comp, len(model.eigenfaces))
        top_k_var = float(np.sum(model.explained_variance_ratio[:k]) * 100.0)
        active_var = float(np.sum(model.explained_variance_ratio[:active_n]) * 100.0)
        total_stored_var = float(np.sum(model.explained_variance_ratio) * 100.0)
        total_stored_n = len(model.eigenfaces)

        mean_faces_path = DEFAULT_MODEL_DIR / "mean_faces"

        details = {
            "Visualized Eigenfaces (K)": f"{k}  ({top_k_var:.2f}% variance)",
            "Active Subspace Components": f"{active_n}  ({active_var:.2f}% variance)",
            "Total Components Stored": f"{total_stored_n}  ({total_stored_var:.2f}% variance)",
            "Training Faces": len(model.X_train_pca),
            "Classes": len(model.target_names),
            "Mean Faces Directory": display_path(mean_faces_path),
            "Image Dimensions": f"{model.img_shape[1]} × {model.img_shape[0]} px",
        }

        # If no image was loaded in the document, initialize it with the grid preview
        if document.current is None:
            document.original = grid_image.copy()
            document.current = grid_image.copy()

        message = (
            f"Eigenfaces model ready! Visualizing top {k} eigenfaces ({top_k_var:.1f}% variance). "
            f"Active subspace: {active_n} components ({active_var:.1f}% variance). "
            f"Model on disk: {total_stored_n} components ({total_stored_var:.1f}% variance)."
        )

        return ToolResult(
            message=message,
            image=grid_image,
            details=details,
            preview_only=True,
        )
