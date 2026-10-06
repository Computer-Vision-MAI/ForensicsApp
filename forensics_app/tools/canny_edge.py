import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .image_utils import normalize_high_depth_image

def apply_canny_edge(
    img: Image.Image,
    sigma: float,
    low_threshold: float,
    high_threshold: float,
    superimpose: bool = False,
    alpha: float = 0.6,
) -> Image.Image:
    """Detect edges and return a binary mask or an overlay on the source colors."""
    mode = img.mode
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    is_gray = mode in ("L", "I", "F") or mode.startswith("I;16")
    
    if is_high_depth:
        gray_arr = (normalize_high_depth_image(img) * 255.0).astype(np.uint8)
    else:
        gray_arr = np.array(img.convert("L"))

    edges = feature.canny(
        gray_arr,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    if superimpose:
        # If the image is high-depth, we need to convert it to a standard RGB array for superimposition.
        # But we use the normlized grayscale to ensure we don't lose the high-depth information in the edge detection process.
        if is_high_depth:
            rgb_arr = np.stack((gray_arr,) * 3, axis=-1).astype(np.float32)
        else:
            rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
        edge_color = np.array([255, 0, 0], dtype=np.float32)
        rgb_arr[edges] = rgb_arr[edges] * (1.0 - alpha) + edge_color * alpha
        return Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8))

    return Image.fromarray((edges * 255).astype(np.uint8))


class CannyEdgeTool(ForensicsTool):
    tool_id = "canny_edge"
    title = "Canny Edge Detector"
    category = "Edge Detection"
    description = "Detects edges in the image using the Canny edge detection algorithm. " \
    "Visualize edges as a binary mask or superimpose them over the original image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> 'ToolResult | None':
        assert document.current is not None

        sigma = simpledialog.askfloat(
            "Canny Edge Detector",
            "Enter Sigma (Gaussian filter standard deviation):",
            initialvalue=1.0,
            minvalue=0.0,
            maxvalue=10.0,
            parent=parent
        )
        if sigma is None:
            return None

        low_thresh = simpledialog.askfloat(
            "Canny Edge Detector",
            "Enter Low Threshold (0-255):",
            initialvalue=10.0,
            minvalue=0.0,
            maxvalue=255.0,
            parent=parent
        )
        if low_thresh is None:
            return None

        high_thresh = simpledialog.askfloat(
            "Canny Edge Detector",
            "Enter High Threshold (0-255):",
            initialvalue=50.0,
            minvalue=0.0,
            maxvalue=255.0,
            parent=parent
        )
        if high_thresh is None:
            return None

        # 3. Solicit Output Style
        superimpose = messagebox.askyesno(
            "Output Style",
            "Do you want to superimpose the edges over the original image?\n\n"
            "(Select 'Yes' for a ponderated RGB overlay, 'No' for a binary edge mask)",
            parent=parent
        )

        alpha = 0.6
        if superimpose:
            alpha = simpledialog.askfloat(
                "Superimpose Weight",
                "Enter Blending Weight (0-1):",
                initialvalue=0.6,
                minvalue=0.0,
                maxvalue=1.0,
                parent=parent
            )
            if alpha is None:
                return None

        try:
            out_img = apply_canny_edge(
                document.current, sigma, low_thresh, high_thresh, superimpose, alpha
            )
        except ValueError as error:
            messagebox.showerror("Processing Error", str(error), parent=parent)
            return None

        msg = (
            "Canny edges superimposed successfully."
            if superimpose else "Canny edge mask generated successfully."
        )

        # 6. Return the ToolResult
        return ToolResult(
            image=out_img,
            message=msg,
            details={
                "Operation": "Canny Edge Detection",
                "Sigma": sigma,
                "Low Threshold": low_thresh,
                "High Threshold": high_thresh,
                "Superimposed": superimpose,
                "Alpha": alpha if superimpose else "N/A"
                
            }
        )
