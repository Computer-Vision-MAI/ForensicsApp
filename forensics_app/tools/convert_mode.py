"""Convert the working image to another Pillow mode."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.core.channels import ALPHA, MODE_CHANNELS, NUMERIC_MODES
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice

TARGET_MODES = ("RGB", "RGBA", "L", "LA", "CMYK", "YCbCr", "LAB", "HSV")


def rescale_to_8bit(image: Image.Image) -> Image.Image:
    """Return a numeric image (I, F, I;16...) as 8-bit grayscale, stretched to 0-255.

    The range is taken from the finite values only; NaN and infinite pixels, and
    every pixel of a flat image, become zero.
    """
    values = np.asarray(image).astype(float)
    finite = np.isfinite(values)
    if not finite.any():
        return Image.new("L", image.size, 0)
    low, high = values[finite].min(), values[finite].max()
    if low == high:
        return Image.new("L", image.size, 0)
    scaled = np.zeros(values.shape)
    scaled[finite] = (values[finite] - low) / (high - low) * 255
    return Image.fromarray(scaled.round().astype(np.uint8))


def has_transparency(image: Image.Image) -> bool:
    """Return True if the image has an alpha channel or a palette transparency."""
    return ALPHA in MODE_CHANNELS[image.mode] or "transparency" in image.info


def convert_mode(image: Image.Image, target: str) -> Image.Image:
    """Return ``image`` converted to ``target``; alpha is dropped, never blended.

    Raises ValueError if ``target`` is not in TARGET_MODES or is the current mode.
    """
    if target not in TARGET_MODES:
        raise ValueError(
            f"Unsupported target mode {target!r}. Choose one of: {', '.join(TARGET_MODES)}."
        )
    if image.mode == target:
        raise ValueError(f"Image is already in mode {target!r}.")

    if image.mode in NUMERIC_MODES:
        image = rescale_to_8bit(image)
    # Pillow cannot convert La to other modes directly
    if image.mode == "La":
        image = image.convert("LA")
    if image.mode == target:
        return image
    try:
        return image.convert(target)
    except ValueError:
        # Some pairs (e.g. LAB to CMYK) have no direct conversion, so convert to RGB first
        return image.convert("RGB").convert(target)


class ConvertModeTool(ForensicsTool):
    tool_id = "convert_mode"
    title = "Convert mode"
    category = "Color"
    description = "Convert the working image to another color mode."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        source = document.current

        targets = tuple(mode for mode in TARGET_MODES if mode != source.mode)
        labels = tuple(f"{mode} ({', '.join(MODE_CHANNELS[mode])})" for mode in targets)

        index = ask_choice(parent, self.title, f"Convert {source.mode} to:", labels)
        if index is None:
            return None
        target = targets[index]

        output = convert_mode(source, target)

        details = {
            "Operation": self.title,
            "From": source.mode,
            "To": target,
            "Alpha": _alpha_note(source, output),
        }
        if source.mode in NUMERIC_MODES:
            details["Values"] = "rescaled to 0-255"
        return ToolResult(
            image=output,
            message=f"Converted the image from {source.mode} to {target}.",
            details=details,
        )


def _alpha_note(source: Image.Image, output: Image.Image) -> str:
    """Describe what happened to the transparency of ``source``."""
    if not has_transparency(source):
        return "none"
    return "kept" if has_transparency(output) else "discarded"
