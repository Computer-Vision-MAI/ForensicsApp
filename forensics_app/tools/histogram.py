import numpy as np
from PIL import Image

# Use the Agg backend directly to prevent conflicts with Tkinter's mainloop
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import tkinter as tk
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Generate Histogram"
    category = "Analysis"
    description = "Extracts and plots the color or grayscale histogram of the current image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        
        current_image = document.current
        mode = current_image.mode

        # Convert palette images to RGBA to get accurate color channels
        if mode in ("P", "PA"):
            current_image = current_image.convert("RGBA")
            
        img_array = np.array(current_image)

        # Create an isolated Matplotlib figure
        fig = Figure(figsize=(8, 6), tight_layout=True)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Plot logic based on image mode
        if mode in ("RGB", "RGBA"):
            channel_names = ("Red", "Green", "Blue", "Alpha")
            colors = ("red", "green", "blue", "gray")
            
            # Iterate through available channels (3 for RGB, 4 for RGBA)
            for i in range(img_array.shape[-1]):
                channel_data = img_array[..., i].flatten()
                ax.plot(
                    np.arange(256),
                    np.histogram(channel_data, bins=256, range=(0, 256))[0],
                    color=colors[i],
                    alpha=0.7,
                    label=channel_names[i]
                )
        
            
            ax.legend(loc="upper right")
            
        else:
            # Handle Grayscale ("L") or Binary ("1")
            channel_data = img_array.flatten()
            ax.plot(
                np.arange(256),
                np.histogram(channel_data, bins=256, range=(0, 256))[0], 
                color="black", 
                alpha=0.7, 
                label="Intensity",
                
            )
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
            message="Histogram generated successfully.",
            details={
                "Operation": "Histogram extraction",
                "Original Mode": document.current.mode,
                "Channels Plotted": img_array.ndim if len(img_array.shape) > 2 else 1
            },
        )