import tkinter as tk
from tkinter import simpledialog
import numpy as np
from PIL import Image
import warnings

# Importamos las herramientas de scikit-image mostradas en la referencia
from skimage import exposure, color, img_as_float, img_as_ubyte

from .dialogs import ask_choice

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

import warnings
import numpy as np
from PIL import Image
from skimage import exposure, color, img_as_ubyte
from .image_utils import extract_target_channel


def apply_contrast_enhancement(img: Image.Image, method: str, clip_percent: int) -> Image.Image:
    """Enhances contrast by protecting colors in LAB and preserving range in high-depth modes."""
    mode = img.mode
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    is_grayscale = mode in ("L", "LA", "1") or is_high_depth
    
    # 1. Extract the target channel using the shared utility
    target_channel, lab_image, alpha_channel = extract_target_channel(img, as_grayscale=is_grayscale)

    # 2. Apply the selected mathematical algorithm
    if method == "percentile":
        p_low, p_high = np.percentile(target_channel, (clip_percent, 100 - clip_percent))
        enhanced = exposure.rescale_intensity(
            target_channel, 
            in_range=(p_low, p_high), 
            out_range=(0, 1)
        )
    elif method == "equalize":
        enhanced = exposure.equalize_hist(target_channel)
    elif method == "adaptive":
        enhanced = exposure.equalize_adapthist(target_channel, clip_limit=0.03)
    else:
        raise ValueError(f"Unsupported contrast enhancement method: '{method}'")

    # 3. Reconstruct the image and pack into 8-bits
    if is_grayscale:
        final_8bit = img_as_ubyte(enhanced)
        output_mode = "LA" if alpha_channel is not None else "L"
    else:
        lab_image[..., 0] = enhanced * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enhanced_rgb = color.lab2rgb(lab_image)
        final_8bit = img_as_ubyte(enhanced_rgb)
        output_mode = "RGBA" if alpha_channel is not None else "RGB"

    # 4. Reassemble the Alpha channel if it existed
    if alpha_channel is not None:
        final_array = np.dstack((final_8bit, alpha_channel))
    else:
        final_array = final_8bit

    out_image = Image.fromarray(final_array, mode=output_mode)
    return out_image

class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast Enhancement"
    category = "Enhancement"
    description = "Normalizes image contrast using multiple algorithms (Percentile, Equalization, CLAHE)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        
        # 1. Ask the user to select the contrast enhancement method
        options = ("Percentile Stretching", "Histogram Equalization", "Adaptive (CLAHE)")
        methods = ("percentile", "equalize", "adaptive")
        
        choice_idx = ask_choice(
            parent, 
            title="Contrast Enhancement", 
            prompt="Select Enhancement Method:", 
            options=options
        )
        
        if choice_idx is None:
            return None  # User canceled the method selection dialog
            
        method = methods[choice_idx]
        percent = 0
        
        # 2. If it's percentile, chain a second dialog for the percentage
        if method == "percentile":
            percent = simpledialog.askinteger(
                "Clip Percentage",
                "Enter clip percentage (0-10):",
                initialvalue=2,
                minvalue=0,
                maxvalue=10,
                parent=parent
            )
            if percent is None:
                return None  # User canceled the percentage input dialog
        
        # 3. Apply the mathematical transformation
        out_image = apply_contrast_enhancement(document.current, method, percent)
        
        return ToolResult(
            image=out_image,
            message=f"Contrast enhanced successfully using {method}.",
            details={
                "Operation": "Contrast Enhancement", 
                "Method": method.capitalize(),
                "Clip Percentage": f"{percent}%" if method == "percentile" else "N/A",
                "Original Mode": document.current.mode,
            }
        )