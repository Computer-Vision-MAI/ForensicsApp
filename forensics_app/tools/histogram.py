import numpy as np
from PIL import Image

# Use the Agg backend directly to prevent conflicts with Tkinter's mainloop
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import tkinter as tk
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def calculate_histograms(image: Image.Image) -> dict[str, np.ndarray]:
    """Return 256-bin counts per channel for supported 8-bit image modes.

    Binary images use intensities 0 and 255; palette images use RGBA colors.
    Other modes are rejected rather than flattened or silently clipped.
    """
    if image.mode == "1":
        image = image.convert("L")
    elif image.mode in ("P", "PA"):
        image = image.convert("RGBA")

    channels = {
        "L": ("Intensity",),
        "LA": ("Intensity", "Alpha"),
        "RGB": ("Red", "Green", "Blue"),
        "RGBA": ("Red", "Green", "Blue", "Alpha"),
        "CMYK": ("Cyan", "Magenta", "Yellow", "Black"),
    }
    if image.mode not in channels:
        raise ValueError(f"Histogram does not support image mode {image.mode!r}.")

    img_array = np.array(image).reshape(-1, len(channels[image.mode]))
    return {
        name: np.bincount(img_array[:, index], minlength=256)
        for index, name in enumerate(channels[image.mode])
    }


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Generate Histogram"
    category = "Analysis"
    description = "Extracts and plots the color or grayscale histogram of the current image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        histograms = calculate_histograms(document.current)

        # Create an isolated Matplotlib figure
        fig = Figure(figsize=(8, 6), tight_layout=True)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        colors = {
            "Intensity": "black", "Alpha": "gray",
            "Red": "red", "Green": "green", "Blue": "blue",
            "Cyan": "cyan", "Magenta": "magenta", "Yellow": "yellow", "Black": "black",
        }
        for name, counts in histograms.items():
            ax.plot(np.arange(256), counts, color=colors[name], alpha=0.7, label=name)
        ax.legend(loc="upper right")

        # Configure chart aesthetics
        ax.set_title("Image Histogram")
        ax.set_xlabel("Pixel Value (0 - 255)")
        ax.set_ylabel("Frequency (Pixel Count)")
        ax.grid(axis='y', alpha=0.3)

        # Render the figure to a raw RGBA buffer
        canvas.draw()
        rgba_buffer = canvas.buffer_rgba()

        # Convert the buffer directly to a PIL Image
        hist_image = Image.frombuffer(
            "RGBA",
            canvas.get_width_height(),
            rgba_buffer,
            "raw",
            "RGBA",
            0,
            1
        )

        return ToolResult(
            image=hist_image,
            preview_only=True,
            message="Histogram generated successfully.",
            details={
                "Operation": "Histogram extraction",
                "Original Mode": document.current.mode,
                "Channels Plotted": len(histograms)
            },
        )
