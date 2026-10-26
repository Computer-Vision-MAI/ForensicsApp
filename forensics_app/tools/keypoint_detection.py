"""Keypoint detection and descriptor extraction tool using ORB and SIFT from skimage.feature."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice
from .image_utils import render_feature_overlay, render_keypoint_overlay, to_grayscale_array

KEYPOINT_METHODS = ("ORB", "SIFT")


def extract_keypoints(
    img: Image.Image,
    method: str = "orb",
    n_keypoints: int = 200,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma: float = 1.6,
    return_meta: bool = False,
) -> tuple[np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Extract keypoints and descriptors using ORB or SIFT algorithms.

    Parameter meanings:
    -------------------
    - method: Feature detector algorithm ('orb' or 'sift').

    [ORB Parameters]
    - n_keypoints: Maximum number of keypoints to return.
        ORB detects candidate points using the FAST algorithm and scores them using
        the Harris corner measure. If more than n_keypoints are detected, only the top
        n_keypoints with the highest Harris response are retained.

    [SIFT Parameters]
    - n_octaves: Maximum number of octave layers in the Gaussian scale space pyramid.
        With every octave, the image spatial resolution is halved and the Gaussian scale
        is doubled. Allows detecting features invariant to large scale transformations.
    - n_scales: Number of scale sublevels sampled per octave.
        Controls the scale sampling density when finding local extrema in the
        Difference-of-Gaussian (DoG) scale-space.
    - sigma (sigma_min in skimage): The initial Gaussian blur level (standard deviation)
        applied to the seed image before constructing the scale-space octave pyramid.
        Higher sigma values smooth the finest scale more heavily.

    Args:
        img: Input PIL Image in any mode.
        method: 'orb' or 'sift'.
        n_keypoints: (ORB only) Maximum keypoints to detect/extract.
        n_octaves: (SIFT only) Maximum octaves in scale pyramid.
        n_scales: (SIFT only) Scales per octave.
        sigma: (SIFT only) Base Gaussian standard deviation.
        return_meta: If True, also return (scales, orientations) extracted alongside keypoints.

    Returns:
        If return_meta is False (default):
            (keypoints, descriptors)
        If return_meta is True:
            (keypoints, descriptors, scales, orientations)
        - keypoints: (N, 2) numpy ndarray of (row, col) coordinates.
        - descriptors: (N, D) numpy ndarray of descriptors (binary for ORB, float for SIFT),
          or None if no keypoints were detected.
        - scales: (N,) numpy ndarray of scale values (sigmas for SIFT, pyramid factors for ORB).
        - orientations: (N,) numpy ndarray of angles in radians.
    """
    method_lower = method.lower()
    gray_arr = to_grayscale_array(img)

    if method_lower == "orb":
        if n_keypoints <= 0:
            raise ValueError(f"n_keypoints must be positive (got {n_keypoints}).")
        extractor = feature.ORB(n_keypoints=n_keypoints)
    elif method_lower == "sift":
        if n_octaves <= 0:
            raise ValueError(f"n_octaves must be positive (got {n_octaves}).")
        if n_scales <= 0:
            raise ValueError(f"n_scales must be positive (got {n_scales}).")
        if sigma < 1.0:
            raise ValueError(
                f"sigma must be at least 1.0 (got {sigma}). SIFT scale-space pyramid requires sigma >= 1.0."
            )
        extractor = feature.SIFT(
            n_octaves=n_octaves,
            n_scales=n_scales,
            sigma_min=sigma,
        )
    else:
        raise ValueError(
            f"Unknown keypoint method '{method}'. Must be 'orb' or 'sift'."
        )

    try:
        extractor.detect_and_extract(gray_arr)
        keypoints = extractor.keypoints
        descriptors = extractor.descriptors
        if keypoints is None:
            keypoints = np.empty((0, 2), dtype=np.float64)
        if return_meta:
            if method_lower == "orb":
                scales = extractor.scales
            else:
                scales = getattr(extractor, "sigmas", extractor.scales)
            orientations = extractor.orientations
            if scales is None:
                scales = np.empty(0, dtype=np.float64)
            if orientations is None:
                orientations = np.empty(0, dtype=np.float64)
            return keypoints, descriptors, scales, orientations
        return keypoints, descriptors
    except RuntimeError as error:
        # Both ORB and SIFT raise RuntimeError when no features are found in uniform/low-contrast images
        if "found no features" in str(error):
            empty_kp = np.empty((0, 2), dtype=np.float64)
            if return_meta:
                return empty_kp, None, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
            return empty_kp, None
        raise


def scale_keypoint_radii(
    scales: np.ndarray | None,
    method: str = "orb",
    orb_radius_multiplier: float = 4.0,
) -> np.ndarray | None:
    """Convert detector scale values to physical pixel radii for visualization."""
    if scales is None:
        return None
    if method.lower() == "orb":
        return scales * orb_radius_multiplier
    return scales


def convert_keypoint_orientations(
    orientations: np.ndarray | None,
    method: str = "orb",
) -> np.ndarray | None:
    """Convert detector-specific orientation angles to Cartesian image coordinates (where 0 rad is right)."""
    if orientations is None:
        return None
    if method.lower() == "sift":
        return (np.pi / 2.0) - orientations
    return orientations


def apply_keypoint_detection(
    img: Image.Image,
    method: str = "orb",
    n_keypoints: int = 200,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma: float = 1.6,
    superimpose: bool = False,
    alpha: float = 0.6,
    marker_radius: int = 0,
    *,
    draw_rich_keypoints: bool = False,
) -> Image.Image:
    """Detect keypoints and visualize them as a binary mask or superimposed red overlay.

    Args:
        img: Input PIL Image in any mode.
        method: 'orb' or 'sift'.
        n_keypoints: (ORB only) Maximum number of keypoints.
        n_octaves: (SIFT only) Maximum octaves.
        n_scales: (SIFT only) Scales per octave.
        sigma: (SIFT only) Base Gaussian standard deviation.
        superimpose: True for ponderated RGB overlay; False for binary keypoint mask.
        alpha: Blending weight in [0.0, 1.0] when superimposing.
        marker_radius: Pixel radius for drawing simple keypoints if draw_rich_keypoints is False.
        draw_rich_keypoints: If True, draw scale circumference circles and orientation rays.

    Returns:
        PIL Image: mode 'L' (binary mask) or mode 'RGB'/'RGBA' (superimposed overlay).
    """
    keypoints, _, scales, orientations = extract_keypoints(
        img=img,
        method=method,
        n_keypoints=n_keypoints,
        n_octaves=n_octaves,
        n_scales=n_scales,
        sigma=sigma,
        return_meta=True,
    )
    if draw_rich_keypoints:
        vis_scales = scale_keypoint_radii(scales, method=method)
        vis_orientations = convert_keypoint_orientations(orientations, method=method)
        return render_keypoint_overlay(
            img=img,
            coords=keypoints,
            scales=vis_scales,
            orientations=vis_orientations,
            superimpose=superimpose,
            alpha=alpha,
        )
    return render_feature_overlay(
        img=img,
        coords=keypoints,
        superimpose=superimpose,
        alpha=alpha,
        marker_radius=marker_radius,
    )


class KeypointDetectionTool(ForensicsTool):
    tool_id = "keypoint_detection"
    title = "Keypoint Detection (ORB / SIFT)"
    category = "Feature Detection"
    description = (
        "Extracts keypoints and descriptors using ORB or SIFT algorithms. "
        "Visualize keypoints as a binary mask or superimpose them over the original image."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask detector / descriptor choice
        method_idx = ask_choice(
            parent,
            self.title,
            "Select keypoint detector and descriptor:",
            KEYPOINT_METHODS,
        )
        if method_idx is None:
            return None
        method = KEYPOINT_METHODS[method_idx].lower()

        # 2. Ask configurable parameters based on chosen algorithm
        n_keypoints = 200
        n_octaves = 8
        n_scales = 3
        sigma = 1.6

        if method == "orb":
            # n_keypoints: Maximum number of keypoints to return
            n_keypoints = simpledialog.askinteger(
                "ORB Parameters",
                "Number of keypoints (n_keypoints):\n"
                "(Maximum number of best keypoints to retain)",
                initialvalue=200,
                minvalue=1,
                maxvalue=100000,
                parent=parent,
            )
            if n_keypoints is None:
                return None
        else:  # SIFT
            # n_octaves: Number of octave layers in the scale-space pyramid
            n_octaves = simpledialog.askinteger(
                "SIFT Parameters",
                "Number of octaves (n_octaves):\n"
                "(Maximum octave pyramid layers, image size halves per octave)",
                initialvalue=8,
                minvalue=1,
                maxvalue=30,
                parent=parent,
            )
            if n_octaves is None:
                return None

            # n_scales: Number of scale sublevels sampled per octave
            n_scales = simpledialog.askinteger(
                "SIFT Parameters",
                "Number of scales per octave (n_scales):\n"
                "(Intermediate scale sampling levels for DoG extrema)",
                initialvalue=3,
                minvalue=1,
                maxvalue=20,
                parent=parent,
            )
            if n_scales is None:
                return None

            # sigma: Initial Gaussian blur level (sigma_min)
            sigma = simpledialog.askfloat(
                "SIFT Parameters",
                "Initial Gaussian blur sigma (sigma_min):\n"
                "(Base scale blur applied to the seed image, minimum 1.0)",
                initialvalue=1.6,
                minvalue=1.0,
                maxvalue=10.0,
                parent=parent,
            )
            if sigma is None:
                return None

        # 3. Solicit Output Style
        superimpose = messagebox.askyesno(
            "Output Style",
            "Do you want to superimpose the keypoints over the original image?\n\n"
            "(Select 'Yes' for a ponderated RGB overlay, 'No' for a binary keypoint mask)",
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
            keypoints, descriptors, scales, orientations = extract_keypoints(
                img=document.current,
                method=method,
                n_keypoints=n_keypoints,
                n_octaves=n_octaves,
                n_scales=n_scales,
                sigma=sigma,
                return_meta=True,
            )
            # Visualize rich keypoints with scale circumference and orientation ray
            vis_scales = scale_keypoint_radii(scales, method=method)
            vis_orientations = convert_keypoint_orientations(orientations, method=method)
            out_img = render_keypoint_overlay(
                img=document.current,
                coords=keypoints,
                scales=vis_scales,
                orientations=vis_orientations,
                superimpose=superimpose,
                alpha=alpha,
            )
        except (ValueError, RuntimeError) as error:
            messagebox.showerror("Processing Error", str(error), parent=parent)
            return None

        msg = (
            f"{method.upper()} keypoints superimposed successfully ({len(keypoints)} detected)."
            if superimpose
            else f"{method.upper()} keypoint mask generated successfully ({len(keypoints)} detected)."
        )

        details = {
            "Operation": "Keypoint Detection",
            "Method": method.upper(),
            "Superimposed": superimpose,
            "Alpha": alpha if superimpose else "N/A",
            "Keypoints Detected": len(keypoints),
        }
        if method == "orb":
            details["n_keypoints"] = n_keypoints
        else:
            details["n_octaves"] = n_octaves
            details["n_scales"] = n_scales
            details["sigma"] = sigma

        return ToolResult(
            image=out_img,
            message=msg,
            details=details,
        )
