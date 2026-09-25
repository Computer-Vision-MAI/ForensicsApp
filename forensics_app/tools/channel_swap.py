"""Channel swap tools for manipulating image channels."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

CHANNEL_LETTERS = "RGB"


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Swap channels"
    category = "Color"
    description = "Swap the channels of the working image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        if Image.getmodebase(document.current.mode) == "L":
            raise ValueError(
                f"The image is grayscale ({document.current.mode}). "
                "Use Undo or Reset to go back to the color image."
            )

        text = simpledialog.askstring(
            "Enter channel order",
            "New channel order (e.g. BGR, RGB, etc.):",
            initialvalue="BGR",
            parent=parent,
        )
        if text is None:
            return None

        order = parse_channel_order(text)

        image_array = np.asarray(document.current.convert("RGB"))
        swapped = swap_channels(image_array, order)
        output = Image.fromarray(swapped)

        new_order = "".join(CHANNEL_LETTERS[i] for i in order)
        return ToolResult(
            image=output,
            message=f"Swapped channels to {new_order}.",
            details={
                "Operation": "Channel swap",
                "Order": f"RGB -> {new_order}",
            },
        )


def parse_channel_order(text: str) -> tuple[int, int, int]:
    """Parse a channel order string like "RGB" or "BGR" into a tuple of indices.

    Case and surrounding whitespace are ignored. Raises ValueError if the text is
    not a permutation of R, G and B.
    """
    order = text.strip().upper()
    if sorted(order) != sorted(CHANNEL_LETTERS):
        raise ValueError(f"Channel order must use each of R, G and B exactly once. Got: {text!r}")
    return tuple(CHANNEL_LETTERS.index(letter) for letter in order)


def swap_channels(array: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    """Swap the channels of an image array according to the given order.

    Raises ValueError if the array is not (height, width, 3) or the order is not
    a permutation of (0, 1, 2).
    """
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Input array must be a 3D array with 3 channels.")
    if sorted(order) != [0, 1, 2]:
        raise ValueError(f"Channel order must use each of R, G and B exactly once. Got: {order}")

    return array[:, :, list(order)]
