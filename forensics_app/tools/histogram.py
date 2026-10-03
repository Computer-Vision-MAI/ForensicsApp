import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

# Mapa comprensivo de canales para modos de Pillow
CHANNEL_NAMES = {
    "1": ("Binary",),
    "L": ("Intensity",),
    "I": ("Intensity (32-bit integer)",),
    "F": ("Intensity (32-bit float)",),
    "I;16": ("Intensity (16-bit)",),
    "I;16L": ("Intensity (16-bit)",),
    "I;16B": ("Intensity (16-bit)",),
    "I;16N": ("Intensity (16-bit)",),
    "P": ("Palette index",),
    "PA": ("Palette index", "Alpha"),
    "RGB": ("Red", "Green", "Blue"),
    "CMYK": ("Cyan", "Magenta", "Yellow", "Black"),
    "YCbCr": ("Luma (Y)", "Blue-difference (Cb)", "Red-difference (Cr)"),
    "LAB": ("Lightness (L)", "Green-red (a)", "Blue-yellow (b)"),
    "HSV": ("Hue", "Saturation", "Value"),
    "RGBA": ("Red", "Green", "Blue", "Alpha"),
    "RGBX": ("Red", "Green", "Blue", "Padding"),
    "RGBa": ("Red (premultiplied)", "Green (premultiplied)", "Blue (premultiplied)", "Alpha"),
    "LA": ("Intensity", "Alpha"),
    "La": ("Intensity (premultiplied)", "Alpha"),
}

def get_channel_color(name: str, index: int) -> str:
    """Assigns a visual color to the plot line based on the channel's name semantics."""
    name_lower = name.lower()
    if "red" in name_lower and "difference" not in name_lower: return "red"
    if "green" in name_lower and "red" not in name_lower: return "green"
    if "blue" in name_lower and "difference" not in name_lower: return "blue"
    if "alpha" in name_lower or "padding" in name_lower: return "gray"
    if "cyan" in name_lower: return "cyan"
    if "magenta" in name_lower: return "magenta"
    if "yellow" in name_lower: return "y"
    if "hue" in name_lower: return "purple"
    if "saturation" in name_lower: return "orange"
    if any(k in name_lower for k in ("intensity", "luma", "lightness", "value", "black", "binary")): return "black"
    
    # Fallback palette for unknown exotic channels
    return [f"tab:{c}" for c in ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]][index % 10]


def calculate_histograms(image: Image.Image, bins: int = 256) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return counts and bin edges per channel for almost any image mode.
    
    Uses np.histogram to support continuous ranges (Float), 16/32-bit depths, 
    and customizable bin quantities.
    """
    original_mode = image.mode

    # Convert binary to grayscale and palettes to RGBA to get meaningful color distributions
    if original_mode == "1":
        image = image.convert("L")
    elif original_mode in ("P", "PA"):
        image = image.convert("RGBA")

    channels = CHANNEL_NAMES.get(image.mode)
    if not channels:
        raise ValueError(f"Histogram does not support image mode {image.mode!r}.")

    img_array = np.array(image)
    
    # Flatten spatial dimensions while keeping channel separation
    if img_array.ndim == 2:
        img_array = img_array.reshape(-1, 1)
    else:
        img_array = img_array.reshape(-1, img_array.shape[-1])

    # Standard 8-bit formats should rigidly measure 0-255 unless scaled
    is_standard_8bit = original_mode not in ("I", "F") and not original_mode.startswith("I;16")
    
    results = {}
    for index, name in enumerate(channels):
        channel_data = img_array[:, index]
        
        # np.histogram handles floats, negatives, and large ints naturally
        if is_standard_8bit:
            counts, edges = np.histogram(channel_data, bins=bins, range=(0, 256))
        else:
            counts, edges = np.histogram(channel_data, bins=bins)
            
        results[name] = (counts, edges)

    return results


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Generate Histogram"
    category = "Analysis"
    description = "Extracts and plots the histogram of the image across all supported modes and depths."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        mode = document.current.mode
        
        # Determine a smart default for the dialog prompt
        default_bins = 256
        if mode.startswith("I;16"):
            default_bins = 1024  # 16-bit has 65536 values, 1024 bins offers a good overview
        
        # Open a minimal dialog to let the user override the resolution
        bins = simpledialog.askinteger(
            "Histogram Resolution",
            f"Enter number of bins to calculate for mode '{mode}':",
            initialvalue=default_bins,
            minvalue=2,
            maxvalue=65536,
            parent=parent
        )
        
        if not bins:
            return None  # Action canceled by user

        # Perform calculations
        histograms = calculate_histograms(document.current, bins=bins)

        # Plotting
        fig = Figure(figsize=(8, 6), tight_layout=True)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        x_min, x_max = float('inf'), float('-inf')

        for index, (name, (counts, edges)) in enumerate(histograms.items()):
            # Calculate midpoints of bins to draw smooth lines
            bin_centers = (edges[:-1] + edges[1:]) / 2
            
            color = get_channel_color(name, index)
            ax.plot(bin_centers, counts, color=color, alpha=0.7, label=name)
            
            # Track global min/max to properly bound the X axis
            x_min = min(x_min, edges[0])
            x_max = max(x_max, edges[-1])

        ax.legend(loc="upper right")
        
        # Clamp X axis exactly to the data range (crucial for floats and 16-bit)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim(bottom=0)

        # Configure aesthetics dynamically
        ax.set_title(f"Image Histogram ({mode}) - {bins} Bins")
        ax.set_xlabel("Pixel Value / Intensity")
        ax.set_ylabel("Frequency (Pixel Count)")
        ax.grid(axis='y', alpha=0.3)

        canvas.draw()
        hist_image = Image.frombuffer(
            "RGBA", canvas.get_width_height(), canvas.buffer_rgba(), "raw", "RGBA", 0, 1
        )

        return ToolResult(
            image=hist_image,
            preview_only=True,
            message=f"Histogram with {bins} bins generated successfully.",
            details={
                "Operation": "Histogram extraction",
                "Original Mode": mode,
                "Bins": bins,
                "Channels Plotted": len(histograms),
                "Data Range": f"[{x_min:.1f}, {x_max:.1f}]"
            },
        )