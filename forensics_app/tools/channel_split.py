"""Split an image into its color channels."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

CHANNEL_NAMES = ("Red", "Green", "Blue")


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Split channels"
    category = "Color"
    description = "Split the working image into its individual color channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        if Image.getmodebase(document.current.mode) == "L":
            raise ValueError(
                f"The image is grayscale ({document.current.mode}). "
                "Use Undo or Reset to go back to the color image."
            )

        channel_index = simpledialog.askinteger(
            "Select channel",
            "Enter channel index (0 for Red, 1 for Green, 2 for Blue):",
            minvalue=0,
            maxvalue=2,
            parent=parent,
        )
        if channel_index is None:
            return None

        # convert() drops alpha without blending, so channel values stay the originals
        image_array = np.asarray(document.current.convert("RGB"))

        channel = extract_channel(image_array, channel_index)
        output = Image.fromarray(channel)

        name = CHANNEL_NAMES[channel_index]
        return ToolResult(
            image=output,
            message=f"Extracted the {name} channel.",
            details={
                "Operation": "Channel split",
                "Channel": name,
                **channel_statistics(channel),
            },
        )


def channel_statistics(channel: np.ndarray) -> dict[str, int | float]:
    """Return the minimum, maximum and mean (rounded to 1 decimal) of a channel."""
    return {
        "Min": int(channel.min()),
        "Max": int(channel.max()),
        "Mean": round(float(channel.mean()), 1),
    }


def extract_channel(array: np.ndarray, index: int) -> np.ndarray:
    """Return channel ``index`` (0=Red, 1=Green, 2=Blue) of an RGB array as a 2D array.

    Raises ValueError if the array is not (height, width, 3) or the index is not 0, 1, or 2.
    """

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image array with shape (height, width, 3).")
    # Explicit check: NumPy would silently accept -1 and return the Blue channel
    if index < 0 or index > 2:
        raise ValueError("Channel index must be 0 (Red), 1 (Green), or 2 (Blue).")

    return array[:, :, index]
