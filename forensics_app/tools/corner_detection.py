"""Corner detection tool supporting Harris and FAST algorithms from skimage.feature."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice
from .image_utils import render_feature_overlay, to_grayscale_array

CORNER_METHODS = ("Harris", "FAST")


def detect_corners(
    img: Image.Image,
    method: str = "harris",
    threshold: float = 0.1,
    min_distance: int = 1,
    fast_n: int = 9,
) -> np.ndarray:
    """Detect corner feature coordinates in an image using Harris or FAST.

    Parameter meanings:
    -------------------
    - method: Corner detection algorithm to use ('harris' or 'fast').
        * 'harris': Computes the auto-correlation matrix of image gradients and detects
          corners where both eigenvalues are large (Noble/Harris response).
        * 'fast': Features from Accelerated Segment Test. Tests 16 pixels on a circle
          around candidate pixels to quickly find high-contrast corner points.
    - threshold: Relative peak threshold in [0.0, 1.0] passed to feature.corner_peaks
      as `threshold_rel`. Minimum intensity of accepted peaks, calculated as
      max(response) * threshold. A higher value keeps only the most prominent peaks,
      reducing false positives/noise.
    - min_distance: Minimal allowed distance separating adjacent peaks in pixels.

    Args:
        img: Input PIL Image in any mode.
        method: 'harris' or 'fast'.
        threshold: Relative peak threshold in [0.0, 1.0].
        min_distance: Minimum distance separating peaks in pixels.

    Returns:
        (N, 2) numpy array of (row, col) coordinates of detected corner peaks.
    """
    method_lower = method.lower()
    if method_lower not in ("harris", "fast"):
        raise ValueError(
            f"Unknown corner detection method '{method}'. Must be 'harris' or 'fast'."
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"Threshold must be between 0.0 and 1.0 (got {threshold})."
        )

    gray_arr = to_grayscale_array(img)

    if method_lower == "harris":
        response = feature.corner_harris(gray_arr)
    else:  # "fast"
        response = feature.corner_fast(gray_arr, n=fast_n)

    if threshold <= 0.0:
        threshold_rel = 0.0
    elif threshold >= 1.0:
        # skimage.feature.peak_local_max uses strict inequality `image > threshold_rel * max(image)`.
        # When threshold_rel=1.0, no pixel can be strictly greater than max(image), returning 0 peaks.
        # np.nextafter(1.0, 0.0) yields the largest float strictly less than 1.0, retaining the maximum peaks.
        threshold_rel = np.nextafter(1.0, 0.0)
    else:
        threshold_rel = threshold

    coords = feature.corner_peaks(
        response,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )
    return coords


def apply_corner_detection(
    img: Image.Image,
    method: str = "harris",
    threshold: float = 0.1,
    superimpose: bool = False,
    alpha: float = 0.6,
    marker_radius: int = 0,
    min_distance: int = 1,
    fast_n: int = 9,
) -> Image.Image:
    """Detect corners and visualize them as a binary mask or superimposed red overlay.

    Args:
        img: Input PIL Image in any mode.
        method: 'harris' or 'fast'.
        threshold: Relative peak threshold in [0.0, 1.0].
        superimpose: True to superimpose in red over original image; False for binary mask.
        alpha: Blending weight in [0.0, 1.0] when superimposing.
        marker_radius: Pixel radius for drawing each corner (0 for single pixel).
        min_distance: Minimum distance separating peaks in pixels.
        fast_n: Number of consecutive pixels on the circle for FAST (default 9).

    Returns:
        PIL Image: mode 'L' (binary mask) or mode 'RGB' (superimposed overlay).
    """
    coords = detect_corners(
        img=img,
        method=method,
        threshold=threshold,
        min_distance=min_distance,
        fast_n=fast_n,
    )
    return render_feature_overlay(
        img=img,
        coords=coords,
        superimpose=superimpose,
        alpha=alpha,
        marker_radius=marker_radius,
    )


class CornerDetectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner Detection"
    category = "Feature Detection"
    description = (
        "Detects corner features using Harris or FAST algorithms. "
        "Visualize corners as a binary mask or superimpose them over the original image."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask corner method (Harris vs FAST)
        method_idx = ask_choice(
            parent,
            self.title,
            "Select corner detection method:",
            CORNER_METHODS,
        )
        if method_idx is None:
            return None
        method = CORNER_METHODS[method_idx].lower()

        # 2. Ask peak threshold
        threshold = simpledialog.askfloat(
            self.title,
            "Enter Peak Threshold (relative to maximum peak, 0.0 - 1.0):",
            initialvalue=0.1,
            minvalue=0.0,
            maxvalue=1.0,
            parent=parent,
        )
        if threshold is None:
            return None

        # 3. Solicit Output Style
        superimpose = messagebox.askyesno(
            "Output Style",
            "Do you want to superimpose the corners over the original image?\n\n"
            "(Select 'Yes' for a ponderated RGB overlay, 'No' for a binary corner mask)",
            parent=parent,
        )

        alpha = 0.6
        if superimpose:
            alpha = simpledialog.askfloat(
                "Superimpose Weight",
                "Enter Blending Weight (0-1):",
                initialvalue=0.6,
                minvalue=0.0,
                maxvalue=1.0,
                parent=parent,
            )
            if alpha is None:
                return None

        try:
            coords = detect_corners(document.current, method=method, threshold=threshold)
            out_img = render_feature_overlay(
                document.current, coords, superimpose=superimpose, alpha=alpha
            )
        except (ValueError, RuntimeError) as error:
            messagebox.showerror("Processing Error", str(error), parent=parent)
            return None

        msg = (
            f"{method.upper()} corners superimposed successfully ({len(coords)} detected)."
            if superimpose
            else f"{method.upper()} corner mask generated successfully ({len(coords)} detected)."
        )

        return ToolResult(
            image=out_img,
            message=msg,
            details={
                "Operation": "Corner Detection",
                "Method": method.upper(),
                "Threshold": threshold,
                "Superimposed": superimpose,
                "Alpha": alpha if superimpose else "N/A",
                "Corners Detected": len(coords),
            },
        )
