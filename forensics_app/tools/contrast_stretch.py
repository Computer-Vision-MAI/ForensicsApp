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
from skimage import exposure, color, img_as_float, img_as_ubyte

def apply_contrast_enhancement(img: Image.Image, method: str, clip_percent: int) -> Image.Image:
    """Enhances contrast by protecting colors in LAB and preserving range in high-depth modes."""
    mode = img.mode
    
    # 1. Detect image categories
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    is_grayscale = mode in ("L", "LA", "1") or is_high_depth
    
    # 2. Extract alpha channel and normalize working array to float [0, 1]
    has_alpha = False
    alpha_channel = None
    float_img = None
    
    if mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        has_alpha = True
        if is_grayscale:
            img = img.convert("LA")
            img_array = np.array(img)
            alpha_channel = img_array[..., 1]
            float_img = img_as_float(img_array[..., 0])
        else:
            img = img.convert("RGBA")
            img_array = np.array(img)
            alpha_channel = img_array[..., 3]
            float_img = img_as_float(img_array[..., :3])
    else:
        if is_high_depth:
            # Load raw data without Pillow clipping to 255
            raw_array = np.asarray(img, dtype=np.float64)
            
            # 1. Identify valid pixels so corrupted data doesn't skew the true min/max
            valid_mask = np.isfinite(raw_array)
            
            if np.any(valid_mask):
                c_min = np.min(raw_array[valid_mask])
                c_max = np.max(raw_array[valid_mask])
                
                # 2. Neutralize non-finite values safely using the true boundaries
                # NaNs and -Infs become the minimum (renders as black), +Infs become the maximum (white)
                raw_array = np.nan_to_num(raw_array, nan=c_min, posinf=c_max, neginf=c_min)
                
                # 3. Normalize manually to [0, 1]
                if c_max > c_min:
                    float_img = (raw_array - c_min) / (c_max - c_min)
                else:
                    float_img = np.zeros_like(raw_array)
            else:
                # If the image is entirely corrupted (all NaNs/Infs), output a black safe image
                float_img = np.zeros_like(raw_array)
        elif is_grayscale:
            img = img.convert("L")
            float_img = img_as_float(np.array(img))
        else:
            img = img.convert("RGB")
            float_img = img_as_float(np.array(img))

    # 3. Prepare target channel for scikit-image
    if is_grayscale:
        target_channel = float_img
    else:
        lab_image = color.rgb2lab(float_img)
        target_channel = lab_image[..., 0] / 100.0

    # 4. Apply the selected mathematical algorithm
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

    # 5. Reconstruct the image and pack into 8-bits
    if is_grayscale:
        final_8bit = img_as_ubyte(enhanced)
        output_mode = "LA" if has_alpha else "L"
    else:
        lab_image[..., 0] = enhanced * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enhanced_rgb = color.lab2rgb(lab_image)
        final_8bit = img_as_ubyte(enhanced_rgb)
        output_mode = "RGBA" if has_alpha else "RGB"

    # 6. Reassemble the Alpha channel if it existed
    if has_alpha:
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