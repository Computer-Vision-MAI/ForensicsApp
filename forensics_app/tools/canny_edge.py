import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

class CannyEdgeTool(ForensicsTool):
    tool_id = "canny_edge"
    title = "Canny Edge Detector"
    category = "Edge Detection"
    description = "Detects edges in the image using the Canny edge detection algorithm. " \
    "Visualize edges as a binary mask or superimpose them over the original image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> 'ToolResult | None':
        assert document.current is not None

        # 1. Convert to grayscale mimicking the base tool
        # (Using 'L' mode ensures an 8-bit [0, 255] baseline for standard Canny operations)
        gray_img = document.current.convert("L")
        gray_arr = np.array(gray_img)

        # 2. Solicit Canny parameters consecutively
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

        # 4. Apply the Canny Edge Detector safely
        try:
            edges = feature.canny(
                gray_arr, 
                sigma=sigma, 
                low_threshold=low_thresh, 
                high_threshold=high_thresh
            )
        except ValueError as error:
            messagebox.showerror("Processing Error", str(error), parent=parent)
            return None

        # 5. Format the output based on user choice
        if superimpose:
            # Convert grayscale base to a 3-channel RGB array
            rgb_arr = np.stack((gray_arr,) * 3, axis=-1).astype(np.float32)
            
            # Define edge color (Pure Red) and blending weight (alpha)
            edge_color = np.array([255, 0, 0], dtype=np.float32)

            alpha = 0.6  # 60% edge color, 40% original image intensity
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

            # Perform the ponderated sum 
            rgb_arr[edges] = (rgb_arr[edges] * (1.0 - alpha)) + (edge_color * alpha)
            rgb_arr[~edges] = (rgb_arr[~edges] * (1.0 - alpha))
            
            out_img = Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8), mode="RGB")
            msg = "Canny edges superimposed successfully."
        else:
            # Create a strict binary mask [0, 255]
            mask_arr = (edges * 255).astype(np.uint8)
            out_img = Image.fromarray(mask_arr, mode="L")
            msg = "Canny edge mask generated successfully."

        # 6. Return the ToolResult
        return ToolResult(
            image=out_img,
            message=msg,
            details={
                "Operation": "Canny Edge Detection",
                "Sigma": sigma,
                "Low Threshold": low_thresh,
                "High Threshold": high_thresh,
                "Superimposed": superimpose
            }
        )