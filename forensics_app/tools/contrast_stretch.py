import numpy as np
from PIL import Image
import tkinter as tk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult



import numpy as np
from PIL import Image
import tkinter as tk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def apply_contrast_stretch(img: Image.Image) -> Image.Image:
    """Normalizes the image to span the full 0-255 intensity range independently per channel."""
    mode = img.mode
    
    # Expand palette colors while retaining any transparency.
    if mode in ("P", "PA"):
        mode = "RGBA" if mode == "PA" or "transparency" in img.info else "RGB"
        img = img.convert(mode)
        
    # Convert to float32 to prevent integer overflow during math operations
    img_array = np.array(img).astype(np.float32)
    
    # Separate the color channels from the Alpha channel
    if mode in ("RGBA", "LA"):
        color_channels = img_array[..., :-1]
        alpha_channel = img_array[..., -1:]
    else:
        color_channels = img_array
        alpha_channel = None

    # Calculate minimum and maximum values for each channel
    c_min = np.min(color_channels, axis=(0, 1), keepdims=True)
    c_max = np.max(color_channels, axis=(0, 1), keepdims=True)
    
    # Prevent division by zero if an image is completely flat
    denominator = c_max - c_min
    denominator[denominator == 0] = 1 
    
    # Apply the contrast stretching formula
    stretched = (color_channels - c_min) / denominator * 255.0
    stretched = np.clip(stretched, 0, 255)
    
    # Reattach the Alpha channel if it existed
    if alpha_channel is not None:
        final_array = np.concatenate([stretched, alpha_channel], axis=-1)
    else:
        final_array = stretched
        
    output_mode = "L" if mode in ("I", "F") else mode
    return Image.fromarray(final_array.astype(np.uint8), mode=output_mode)


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast Stretching"
    category = "Enhancement"
    description = "Normalizes the image to span the full 0-255 intensity range."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        
        out_image = apply_contrast_stretch(document.current)
        
        return ToolResult(
            image=out_image,
            message="Contrast stretched successfully.",
            details={
                "Operation": "Contrast Stretch", 
                "Original Mode": document.current.mode,
                "Output Mode": out_image.mode
            }
        )
