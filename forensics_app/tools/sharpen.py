"""Sharpen the working image with unsharp masking."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice
from .image_utils import image_to_unit_array, per_channel, unit_array_to_image

MANUAL = "manual"
SKIMAGE = "skimage"
DETAIL = "detail"
METHODS = (MANUAL, SKIMAGE, DETAIL)
METHOD_LABELS = (
    "Manual: original + amount x (original - blurred)",
    "skimage unsharp_mask",
    "Show only the details (original - blurred)",
)

DEFAULT_SIGMA = 2.0
DEFAULT_AMOUNT = 1.5
MAX_SIGMA = 100.0
MAX_AMOUNT = 10.0


def blur(pixels: np.ndarray, sigma: float) -> np.ndarray:
    """Return ``pixels`` smoothed with a Gaussian, each channel on its own."""
    # reflect is the border mode unsharp_mask uses, so both methods agree
    return per_channel(lambda channel: filters.gaussian(channel, sigma=sigma, mode="reflect"), pixels)


def detail_layer(pixels: np.ndarray, sigma: float) -> np.ndarray:
    """Return what the blur removes: ``pixels`` minus their blurred version (signed)."""
    return pixels - blur(pixels, sigma)


def sharpen_manual(pixels: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Return ``pixels`` plus ``amount`` times their detail layer, clipped to [0, 1]."""
    return np.clip(pixels + amount * detail_layer(pixels, sigma), 0.0, 1.0)


def sharpen_skimage(pixels: np.ndarray, radius: float, amount: float) -> np.ndarray:
    """Return ``pixels`` sharpened by skimage's unsharp_mask, each channel on its own."""
    # Not channel_axis=-1: skimage 0.26 slices the wrong axis for a negative one
    return per_channel(
        lambda channel: filters.unsharp_mask(channel, radius=radius, amount=amount), pixels
    )


def sharpen(image: Image.Image, method: str, sigma: float, amount: float = DEFAULT_AMOUNT) -> Image.Image:
    """Return ``image`` sharpened with ``method``, or its detail layer on mid-gray.

    The result is 8-bit L or RGB (LA or RGBA if the image has transparency).
    Raises ValueError if ``method`` is not in METHODS.
    """
    if method not in METHODS:
        raise ValueError(f"Unsupported sharpening method {method!r}. Choose one of: {', '.join(METHODS)}.")
    pixels, alpha = image_to_unit_array(image)
    if method == MANUAL:
        result = sharpen_manual(pixels, sigma, amount)
    elif method == SKIMAGE:
        result = sharpen_skimage(pixels, sigma, amount)
    else:
        # The details are signed, so 0 is drawn as mid-gray
        result = detail_layer(pixels, sigma) + 0.5
    return unit_array_to_image(result, alpha)


class SharpenTool(ForensicsTool):
    tool_id = "sharpen"
    title = "Sharpen"
    category = "Enhancement"
    description = "Sharpen the image with unsharp masking, or show the details it adds."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        source = document.current

        index = ask_choice(parent, self.title, "Sharpening method:", METHOD_LABELS)
        if index is None:
            return None
        method = METHODS[index]

        sigma = simpledialog.askfloat(
            self.title,
            f"Blur {'radius' if method == SKIMAGE else 'sigma'} (0-{MAX_SIGMA:g}):",
            initialvalue=DEFAULT_SIGMA,
            minvalue=0.0,
            maxvalue=MAX_SIGMA,
            parent=parent,
        )
        if sigma is None:
            return None

        details = {
            "Operation": self.title,
            "Method": METHOD_LABELS[index],
            "Radius" if method == SKIMAGE else "Sigma": sigma,
        }

        if method == DETAIL:
            output = sharpen(source, method, sigma)
            details["Mode"] = f"{source.mode} -> {output.mode}"
            # The details are an analysis view, so they do not replace the working image
            return ToolResult(
                image=output,
                message="Showing the details removed by the blur; the working image is unchanged.",
                details=details,
                preview_only=True,
            )

        amount = simpledialog.askfloat(
            self.title,
            f"Amount of detail to add (0-{MAX_AMOUNT:g}):",
            initialvalue=DEFAULT_AMOUNT,
            minvalue=0.0,
            maxvalue=MAX_AMOUNT,
            parent=parent,
        )
        if amount is None:
            return None

        output = sharpen(source, method, sigma, amount)
        details["Amount"] = amount
        details["Mode"] = f"{source.mode} -> {output.mode}"
        return ToolResult(
            image=output,
            message=f"Sharpened the image (amount {amount:g}).",
            details=details,
        )
