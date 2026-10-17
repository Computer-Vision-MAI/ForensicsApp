from __future__ import annotations
import warnings
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import exposure, color, img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import OPEN_TYPES, load_image_file
from .image_utils import extract_target_channel, image_to_unit_array


# The three ways of measuring how bright a pixel is. They give different numbers
# for the same color, so the base and the reference must always use the same one
LIGHTNESS = "lightness"  # L* of LAB
LUMA = "luma"  # Y of YCbCr, and the gray of mode L
VALUE = "value"  # V of HSV: the largest of red, green and blue

# The modes that are matched without leaving their color space: which intensity
# they hold and in which channel
NATIVE_INTENSITY = {"LAB": (LIGHTNESS, 0), "YCbCr": (LUMA, 0), "HSV": (VALUE, 2)}


def reference_intensity(ref_image: Image.Image, kind: str) -> np.ndarray:
    """Return the ``kind`` of intensity (LIGHTNESS, LUMA or VALUE) of the reference, in [0, 1].

    Raises ValueError if the reference mode is not supported.
    """
    if ref_image.mode in NATIVE_INTENSITY:
        native_kind, index = NATIVE_INTENSITY[ref_image.mode]
        if native_kind == kind:
            # The reference already stores this intensity: read it as it is
            return img_as_float(np.array(ref_image))[..., index]
        # It stores another one (e.g. V when L* is needed): compute it from the colors
        ref_image = ref_image.convert("RGB")

    if kind == VALUE:
        pixels, _ = image_to_unit_array(ref_image)
        # A gray image has a single channel, which is already its value
        return pixels if pixels.ndim == 2 else pixels.max(axis=-1)
    target_ref, _, _ = extract_target_channel(ref_image, as_grayscale=kind == LUMA)
    return target_ref


def apply_histogram_match(base_image: Image.Image, ref_image: Image.Image) -> Image.Image:
    """Matches the histogram of the base image to the reference image, natively handling LAB/HSV/YCbCr."""
    mode = base_image.mode
    
    # 1. Native Branch: Process LAB, HSV and YCbCr directly, without leaving their color space
    if mode in NATIVE_INTENSITY:
        float_base = img_as_float(np.array(base_image))
        
        # Luminance (L) is index 0 in LAB. Value (V) is index 2 in HSV. Luma (Y) is index 0 in YCbCr.
        kind, target_idx = NATIVE_INTENSITY[mode]
        target_base = float_base[..., target_idx]

        # The reference is measured with the same intensity as the base channel
        target_ref = reference_intensity(ref_image, kind)
        
        # Apply the matching algorithm
        float_base[..., target_idx] = exposure.match_histograms(target_base, target_ref)
        
        # Output natively back to the original format
        final_8bit = img_as_ubyte(float_base)
        return Image.fromarray(final_8bit, mode=mode)

    # 2. Standard Branch: RGB, Grayscale, CMYK, and High-Depth modes
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    is_grayscale = mode in ("L", "LA", "1") or is_high_depth
    
    # Extract the base channel using the shared utility: gray for a grayscale
    # image, L* for a color one. The reference is measured the same way
    target_base, lab_base, alpha_base = extract_target_channel(base_image, as_grayscale=is_grayscale)
    target_ref = reference_intensity(ref_image, LUMA if is_grayscale else LIGHTNESS)

    # Apply the matching algorithm
    matched = exposure.match_histograms(target_base, target_ref)

    # 3. Reconstruct the image and pack into 8-bits
    if is_grayscale:
        final_8bit = img_as_ubyte(matched)
        
        output_mode = "LA" if alpha_base is not None else "L"
    else:
        lab_base[..., 0] = matched * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enhanced_rgb = color.lab2rgb(lab_base)
        final_8bit = img_as_ubyte(enhanced_rgb)
        output_mode = "RGBA" if alpha_base is not None else "RGB"

    # 4. Reassemble the Alpha channel if it existed
    if alpha_base is not None:
        final_array = np.dstack((final_8bit, alpha_base))
    else:
        final_array = final_8bit

    return Image.fromarray(final_array, mode=output_mode)


class HistogramMatchTool(ForensicsTool):
    tool_id = "histogram_match"
    title = "Match Histogram"
    category = "Enhancement"
    description = "Modifies the image contrast and brightness to match the histogram of a selected reference image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Ask the user for the reference image
        ref_data = self.load_reference_image(parent, title="Select Reference Image for Histogram Matching")
        if not ref_data:
            return None  # The user canceled the selection
        
        ref_path, ref_img = ref_data

        # 2. Apply the mathematical logic and catch processing errors safely
        try:
            output = apply_histogram_match(document.current, ref_img)
        except ValueError as error:
            messagebox.showerror("Processing Error", str(error), parent=parent)
            return None
  
        # 3. Return the result
        return ToolResult(
            image=output,
            message="Histogram matched successfully.",
            details={
                "Operation": "Histogram Match",
                "Reference File": ref_path.name,
                "Original Mode": document.current.mode,
                "Output Mode": output.mode,
            },
        )

    def load_reference_image(self, parent: tk.Misc, title: str) -> tuple[Path, Image.Image] | None:
        """Open a dialog to load the reference image without quantizing or resizing it."""
        return load_image_file(parent, title=title)