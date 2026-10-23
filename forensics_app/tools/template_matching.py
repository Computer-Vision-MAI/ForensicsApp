from __future__ import annotations
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import feature, img_as_float, transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import OPEN_TYPES, ask_choice, load_image_file
from .image_utils import is_grayscale, normalize_high_depth_image, render_mask_overlay, to_grayscale_array


RESIZE_OPTIONS = (
    "No resizing (keep template as-is)",
    "Rescale (% of base image)",
    "Resize (fixed height and width in pixels)",
)

# Re-export to_grayscale_array for backwards compatibility and test imports
to_unit_gray_array = to_grayscale_array


def to_unit_color_array(img: Image.Image) -> np.ndarray:
    """Convert an image to a 3D float RGB array (H, W, 3) in [0, 1]."""
    mode = img.mode
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    if is_high_depth:
        gray = normalize_high_depth_image(img)
        return np.stack([gray, gray, gray], axis=-1)
    if mode in ("RGBA", "RGBa"):
        arr = np.array(img.convert("RGBA"))
        return img_as_float(arr[..., :3])
    return img_as_float(np.array(img.convert("RGB")))


def prepare_matched_arrays(
    image: Image.Image, template: Image.Image
) -> tuple[np.ndarray, np.ndarray]:
    """Convert image and template into matching representations.

    If either is grayscale, both are converted to 2D grayscale arrays.
    If both are color, both are converted to 3D RGB arrays.
    """
    orig_gray = is_grayscale(image)
    tmpl_gray = is_grayscale(template)

    if orig_gray or tmpl_gray:
        return to_unit_gray_array(image), to_unit_gray_array(template)
    return to_unit_color_array(image), to_unit_color_array(template)


def resize_template(
    template: Image.Image, base_size: tuple[int, int], size: float | tuple[int, int] | None
) -> Image.Image:
    """Resize template based on percentage of base image or fixed size."""
    if size is None:
        return template

    base_w, base_h = base_size
    tmpl_w, tmpl_h = template.size

    if isinstance(size, (int, float)):
        pct = float(size)
        if pct <= 0.0 or pct > 100.0:
            raise ValueError(f"Size percentage must be in the range (0, 100], got {pct}.")
        scale = (pct / 100.0) * min(base_w / tmpl_w, base_h / tmpl_h)
        new_w = max(1, int(round(tmpl_w * scale)))
        new_h = max(1, int(round(tmpl_h * scale)))
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        new_w, new_h = int(size[0]), int(size[1])
        if new_w <= 0 or new_h <= 0:
            raise ValueError("Template dimensions must be positive integers.")
    else:
        raise ValueError(f"Invalid size parameter: {size}")

    if (new_w, new_h) == (tmpl_w, tmpl_h):
        return template
    return template.resize((new_w, new_h), Image.Resampling.LANCZOS)


def rotate_template_array(template_arr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate template array anticlockwise by angle in degrees."""
    normalized_angle = angle % 360.0
    if np.isclose(normalized_angle, 0.0) or np.isclose(normalized_angle, 360.0):
        return template_arr

    # Exact quarter-turn rotation for multiples of 90 degrees
    rem = normalized_angle % 90.0
    if np.isclose(rem, 0.0) or np.isclose(rem, 90.0):
        k = int(round(normalized_angle / 90.0)) % 4
        if k == 0:
            return template_arr
        return np.ascontiguousarray(np.rot90(template_arr, k=k, axes=(0, 1)))

    # For other angles, resize canvas so rotation preserves template content
    return transform.rotate(template_arr, angle, resize=True)


def render_template_result(
    image: Image.Image,
    matches: np.ndarray,
    superimpose: bool,
    alpha: float,
) -> Image.Image:
    """Render matches as either a binary mask or a red superimposed overlay."""
    return render_mask_overlay(image, matches, superimpose=superimpose, alpha=alpha)


def apply_template_matching(
    image: Image.Image,
    template: Image.Image,
    size: float | tuple[int, int] | None = None,
    rotation: float = 0.0,
    threshold: float = 0.8,
    superimpose: bool = False,
    alpha: float = 0.6,
) -> Image.Image:
    """Detect template matches and return a binary mask or superimposed image."""
    base_w, base_h = image.size
    if base_w == 0 or base_h == 0:
        raise ValueError("Image dimensions must be non-zero.")

    # 1. Resize template
    resized_tmpl = resize_template(template, (base_w, base_h), size)

    tmpl_w, tmpl_h = resized_tmpl.size
    if tmpl_w > base_w or tmpl_h > base_h:
        raise ValueError(
            f"Template dimensions ({tmpl_w}x{tmpl_h}) cannot exceed "
            f"image dimensions ({base_w}x{base_h})."
        )

    # 2. Prepare type-matched arrays
    img_arr, tmpl_arr = prepare_matched_arrays(image, resized_tmpl)

    # 3. Rotate template array
    tmpl_arr = rotate_template_array(tmpl_arr, rotation)

    rot_tmpl_h, rot_tmpl_w = tmpl_arr.shape[:2]
    if rot_tmpl_w > base_w or rot_tmpl_h > base_h:
        raise ValueError(
            f"Template dimensions ({rot_tmpl_w}x{rot_tmpl_h}) cannot exceed "
            f"image dimensions ({base_w}x{base_h})."
        )

    # 4. Normalized cross-correlation
    corr = feature.match_template(img_arr, tmpl_arr, pad_input=True)
    if corr.ndim == 3:
        # For 3D multichannel images with pad_input=True, skimage pads all 3 axes.
        # The center slice along the 3rd axis (index 1) represents the unpadded,
        # fully aligned 3D cross-correlation across all RGB channels jointly.
        corr_2d = corr[..., corr.shape[-1] // 2]
    else:
        corr_2d = corr

    corr_2d = np.nan_to_num(corr_2d, nan=-1.0)
    matches = corr_2d >= threshold

    # 5. Render output
    return render_template_result(image, matches, superimpose, alpha)


class TemplateMatchingTool(ForensicsTool):
    tool_id = "template_matching"
    title = "Template Matching"
    category = "Feature Detection"
    description = (
        "Matches a template image against the current image using normalized "
        "cross-correlation with configurable size, rotation, and threshold."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Open template image
        tmpl_data = self.load_image(parent, title="Open template image")
        if not tmpl_data:
            return None
        tmpl_path, tmpl_img = tmpl_data

        base_w, base_h = document.current.size
        tmpl_w, tmpl_h = tmpl_img.size

        # 2. Ask resizing option: No resizing, Rescale (%), or Resize (H and W)
        choice_idx = ask_choice(
            parent,
            "Template Resizing",
            "Choose template sizing option:",
            RESIZE_OPTIONS,
        )
        if choice_idx is None:
            return None

        size: float | tuple[int, int] | None = None
        if choice_idx == 0:
            size = None
        elif choice_idx == 1:
            scale_pct = simpledialog.askfloat(
                "Template Rescale",
                "Enter template scale (% of base image, 1-100):",
                initialvalue=20.0,
                minvalue=1.0,
                maxvalue=100.0,
                parent=parent,
            )
            if scale_pct is None:
                return None
            size = scale_pct
        elif choice_idx == 2:
            target_h = simpledialog.askfloat(
                "Template Height",
                f"Enter template height in pixels (1-{base_h}):",
                initialvalue=float(min(tmpl_h, base_h)),
                minvalue=1.0,
                maxvalue=float(base_h),
                parent=parent,
            )
            if target_h is None:
                return None

            target_w = simpledialog.askfloat(
                "Template Width",
                f"Enter template width in pixels (1-{base_w}):",
                initialvalue=float(min(tmpl_w, base_w)),
                minvalue=1.0,
                maxvalue=float(base_w),
                parent=parent,
            )
            if target_w is None:
                return None

            size = (int(round(target_w)), int(round(target_h)))

        # 3. Ask rotation (degrees anticlockwise 0-359)
        rotation = simpledialog.askfloat(
            "Template Rotation",
            "Enter rotation angle in degrees (0-359, anticlockwise):",
            initialvalue=0.0,
            minvalue=0.0,
            maxvalue=359.0,
            parent=parent,
        )
        if rotation is None:
            return None

        # 4. Ask threshold (-1.0 to 1.0)
        threshold = simpledialog.askfloat(
            "Template Matching Threshold",
            "Enter correlation threshold (-1.0 to 1.0):",
            initialvalue=0.8,
            minvalue=-1.0,
            maxvalue=1.0,
            parent=parent,
        )
        if threshold is None:
            return None

        # 5. Superimposability & Alpha weight
        superimpose = messagebox.askyesno(
            "Output Style",
            "Do you want to superimpose the match over the original image?\n\n"
            "(Select 'Yes' for a red overlay on the original, 'No' for a thresholded mask)",
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

        # 6. Apply template matching
        try:
            out_img = apply_template_matching(
                image=document.current,
                template=tmpl_img,
                size=size,
                rotation=rotation,
                threshold=threshold,
                superimpose=superimpose,
                alpha=alpha,
            )
        except ValueError as error:
            messagebox.showerror("Processing Error", str(error), parent=parent)
            return None

        msg = (
            "Template matching result superimposed successfully."
            if superimpose
            else "Template matching mask generated successfully."
        )

        return ToolResult(
            image=out_img,
            message=msg,
            details={
                "Operation": "Template Matching",
                "Template": tmpl_path.name,
                "Sizing": RESIZE_OPTIONS[choice_idx],
                "Size Parameter": (
                    "Original"
                    if size is None
                    else f"{size:g}%"
                    if isinstance(size, (int, float))
                    else f"{size[0]}x{size[1]}"
                ),
                "Rotation (deg)": rotation,
                "Threshold": threshold,
                "Superimposed": superimpose,
                "Alpha": alpha if superimpose else "N/A",
            },
        )

    def load_image(
        self, parent: tk.Misc, title: str = "Open template image"
    ) -> tuple[Path, Image.Image] | None:
        """Open a dialog to load a template image and return it."""
        return load_image_file(parent, title=title)
