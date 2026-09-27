from __future__ import annotations
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.messagebox import askyesno

import numpy as np

from PIL import Image
from PIL import UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

class MaskTool(ForensicsTool):
    tool_id = "mask"
    title = "Apply mask"
    category = "Filtering"
    description = "Mask an image with another image and potentially apply a texture to the mask."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        # 1. Load the mask without forcing conversion yet so it retains its source-channel semantics
        mask_data = self.load_image(parent, title="Open mask image", base_image=None)
        if not mask_data:
            return None  # The user canceled the selection
        mask_path, mask_img = mask_data

        # 2. Ask for the texture
        with_texture = askyesno(
            "Texture", 
            "Do you want to apply a specific texture to the mask?",
            parent=parent
        )
        
        texture_img = None
        if with_texture:
            # Load the texture image in the same mode/palette as the current image
            texture_data = self.load_image(parent, title="Open texture image", base_image=document.current)
            if not texture_data:
                return None  # The user canceled when asked for the texture
            texture_path, texture_img = texture_data

        # 3. Apply the boolean logic
        output = self.apply_mask(document.current, mask_img, texture_img)
  
        # 4. Return the result
        return ToolResult(
            image=output,
            message="Mask applied successfully.",
            details={
                "Operation": "Apply mask",
                "Mask used": mask_path.name,
                "Texture applied": texture_path.name if with_texture else "No Texture",
                "Output mode": output.mode,
            },
        )

    def load_image(self, parent: tk.Misc, title: str, base_image: Image.Image | None = None) -> tuple[Path, Image.Image] | None:
        """Open a dialog to load a generic image and return it."""
        filename = filedialog.askopenfilename(title=title, filetypes=OPEN_TYPES, parent=parent)
        if not filename:
            return None
        
        try:
            source = Path(filename)
            with Image.open(source) as image:
                loaded = image.copy()
                if base_image is not None:
                    # Convert mapping to the base image's palette if P mode is used
                    if base_image.mode == "P":
                        loaded = loaded.convert("RGB").quantize(palette=base_image)
                    else:
                        loaded = loaded.convert(base_image.mode)
            return source, loaded
    
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=parent)
            return None

    def apply_mask(self, base_image: Image.Image, mask_img: Image.Image, texture_img: Image.Image | None) -> Image.Image:
        """Apply the mask (and optional texture) using a boolean array."""
        
        # Resize the mask if necessary using NEAREST to prevent selecting originally black pixels
        if mask_img.size != base_image.size:
            mask_img = mask_img.resize(base_image.size, Image.Resampling.NEAREST)

        # Convert to numpy arrays
        base_arr = np.array(base_image)
        mask_arr = np.array(mask_img)

        # Create a boolean mask: True where any channel is > 0 (reduce to one bool per pixel)
        if mask_arr.ndim > 2:
            bool_mask = np.any(mask_arr > 0, axis=-1)
        else:
            bool_mask = mask_arr > 0

        # Copy the original image to avoid modifying it directly in memory
        output_arr = base_arr.copy()

        if texture_img is not None:
            # Texture exists: resize it (keeping filter unchanged), convert to array, and apply it
            if texture_img.size != base_image.size:
                texture_img = texture_img.resize(base_image.size, Image.Resampling.LANCZOS)
            
            texture_arr = np.array(texture_img)
            
            # Replace pixels in the original image using the boolean mask
            output_arr[bool_mask] = texture_arr[bool_mask]
        else:
            # No texture: apply the colors from the original mask directly
            # Ensure mask colors match base mode before replacement
            if mask_img.mode != base_image.mode:
                if base_image.mode == "P":
                    mask_img = mask_img.convert("RGB").quantize(palette=base_image)
                else:
                    mask_img = mask_img.convert(base_image.mode)
                    
            mask_paint_arr = np.array(mask_img)
            output_arr[bool_mask] = mask_paint_arr[bool_mask]

        # Reconstruct result preserving base image's mode and palette
        out_image = Image.fromarray(output_arr, mode=base_image.mode)
        if base_image.mode == "P" and base_image.getpalette() is not None:
            out_image.putpalette(base_image.getpalette())
            
        return out_image