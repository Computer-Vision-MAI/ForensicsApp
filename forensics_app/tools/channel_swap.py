"""Channel swap tools for manipulating image channels."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.core.channels import FIXED_CHANNELS, MODE_CHANNELS, to_multichannel
from .base import ForensicsTool, ToolResult
from .dialogs import ask_channel_order


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Swap channels"
    category = "Color"
    description = "Swap the channels of the working image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        image = to_multichannel(document.current)
        names = MODE_CHANNELS[image.mode]
        movable = tuple(i for i, name in enumerate(names) if name not in FIXED_CHANNELS)
        fixed = tuple(i for i, name in enumerate(names) if name in FIXED_CHANNELS)
        if len(movable) < 2:
            raise ValueError(f"Image mode {image.mode!r} has only one color channel, "
                             "there is nothing to swap.")

        movable_names = tuple(names[i] for i in movable)
        default = tuple(reversed(range(len(movable))))
        chosen = ask_channel_order(parent, "Swap channels", movable_names, default)
        if chosen is None:
            return None

        order = tuple(movable[i] for i in chosen) + fixed

        swapped = swap_channels(np.asarray(image), order)
        output = Image.fromarray(swapped, mode=image.mode)

        before = ", ".join(movable_names)
        after = ", ".join(names[i] for i in order if i in movable)
        return ToolResult(
            image=output,
            message=f"Swapped channels to {after}.",
            details={
                "Operation": "Channel swap",
                "Source mode": image.mode,
                "Order": f"{before} -> {after}",
            },
        )


def swap_channels(array: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Reorder the channels of a (height, width, channels) array.

    Raises ValueError if the array has no channel axis or ``order``
    is not a permutation of its channel indices.
    """
    if array.ndim != 3:
        raise ValueError("Expected a (height, width, channels) array.")
    channel_count = array.shape[2]
    if sorted(order) != list(range(channel_count)):
        raise ValueError(f"Order {order} must use each channel index 0-{channel_count - 1} exactly once.")
    return array[:, :, list(order)]
