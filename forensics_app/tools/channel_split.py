"""Split an image into its color channels."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.core.channels import MODE_CHANNELS, to_multichannel
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Split channels"
    category = "Color"
    description = "Split the working image into its individual color channels."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        image = to_multichannel(document.current)
        names = MODE_CHANNELS[image.mode]

        channel_index = ask_choice(
            parent,
            "Split channels",
            f"Channel to extract ({image.mode}):",
            names,
        )
        if channel_index is None:
            return None

        image_array = np.asarray(image)

        channel = extract_channel(image_array, channel_index)
        output = Image.fromarray(channel)

        name = names[channel_index]
        return ToolResult(
            image=output,
            message=f"Extracted the {name} channel.",
            details={
                "Operation": "Channel split",
                "Source mode": image.mode,
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
    """Return channel ``index`` of a (height, width, channels) array as a 2D array.

    Raises ValueError if the array has no channel axis or the index is out of range.
    """
    if array.ndim != 3:
        raise ValueError("Expected a (height, width, channels) array.")
    channel_count = array.shape[2]
    # Explicit check, NumPy would silently accept -1 and return the last channel.
    if not 0 <= index < channel_count:
        raise ValueError(f"Channel index {index} must be between 0 and {channel_count - 1}.")
    return array[:, :, index]
