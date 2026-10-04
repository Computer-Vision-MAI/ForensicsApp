from __future__ import annotations
import warnings
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import exposure, color, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .image_utils import extract_target_channel

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


import warnings
import numpy as np
from PIL import Image
from skimage import exposure, color, img_as_float, img_as_ubyte
from .image_utils import extract_target_channel

def apply_histogram_match(base_image: Image.Image, ref_image: Image.Image) -> Image.Image:
    """Matches the histogram of the base image to the reference image, natively handling HSV/YCbCr."""
    mode = base_image.mode
    
    # 1. Native Branch: Process HSV and YCbCr directly without LAB conversion
    if mode in ("LAB", "HSV", "YCbCr"):
        float_base = img_as_float(np.array(base_image))
        
        # Luminance (L) is index 0 in LAB. Value (V) is index 2 in HSV. Luma (Y) is index 0 in YCbCr.
        target_idx = 0 if (mode == "LAB" or mode == "YCbCr") else 2
        target_base = float_base[..., target_idx]
        
        # We only need the intensity distribution from the reference image, 
        # so we extract it as a 2D grayscale array regardless of its original mode.
        target_ref, _, _ = extract_target_channel(ref_image, as_grayscale=True)
        
        # Apply the matching algorithm
        float_base[..., target_idx] = exposure.match_histograms(target_base, target_ref)
        
        # Output natively back to the original format
        final_8bit = img_as_ubyte(float_base)
        return Image.fromarray(final_8bit, mode=mode)

    # 2. Standard Branch: RGB, Grayscale, CMYK, and High-Depth modes
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    is_grayscale = mode in ("L", "LA", "1") or is_high_depth
    
    # Extract channels for both images using the shared utility
    target_base, lab_base, alpha_base = extract_target_channel(base_image, as_grayscale=is_grayscale)
    target_ref, _, _ = extract_target_channel(ref_image, as_grayscale=is_grayscale)

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

        # 2. Apply the mathematical logic
        output = apply_histogram_match(document.current, ref_img)
  
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
        filename = filedialog.askopenfilename(title=title, filetypes=OPEN_TYPES, parent=parent)
        if not filename:
            return None
        
        try:
            source = Path(filename)
            with Image.open(source) as image:
                # Copy into memory so the file lock is released immediately
                loaded = image.copy() 
            return source, loaded
    
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=parent)
            return None