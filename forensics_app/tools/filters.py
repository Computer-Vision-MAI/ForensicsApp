"""Apply one of the classic filters from skimage.filters to the working image."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice
from .image_utils import image_to_unit_array, is_grayscale, per_channel, unit_array_to_image
from .sharpen import MAX_AMOUNT, MAX_SIGMA, sharpen_skimage

GAUSSIAN = "gaussian"
MEDIAN = "median"
SOBEL = "sobel"
PREWITT = "prewitt"
SCHARR = "scharr"
UNSHARP = "unsharp_mask"
OTSU = "otsu"

FILTER_LABELS = {
    GAUSSIAN: "Gaussian (smoothing / noise reduction)",
    MEDIAN: "Median (edge-preserving denoising)",
    SOBEL: "Sobel (edge detection)",
    PREWITT: "Prewitt (edge detection)",
    SCHARR: "Scharr (edge detection)",
    UNSHARP: "Unsharp mask (sharpening)",
    OTSU: "Otsu threshold (binarization)",
}
EDGE_OPERATORS = {SOBEL: filters.sobel, PREWITT: filters.prewitt, SCHARR: filters.scharr}
EDGE_SOURCE_LABELS = (
    "Grayscale: one set of edges",
    "Each color channel on its own: colored edges",
)

# The weights Pillow uses for mode L, so the result matches Convert to grayscale
LUMA_WEIGHTS = (0.299, 0.587, 0.114)

DEFAULT_SIGMA = 2.0
DEFAULT_MEDIAN_SIZE = 3
MAX_MEDIAN_SIZE = 101
DEFAULT_RADIUS = 1.0
DEFAULT_AMOUNT = 1.0


def to_gray(pixels: np.ndarray) -> np.ndarray:
    """Return an H x W x 3 RGB array as H x W luminance; a gray array is returned as is."""
    # Gray = 0.299 x red + 0.587 x green + 0.114 x blue, for every pixel
    return pixels if pixels.ndim == 2 else pixels @ np.array(LUMA_WEIGHTS)


def gaussian(pixels: np.ndarray, sigma: float) -> np.ndarray:
    """Return ``pixels`` smoothed with a Gaussian of standard deviation ``sigma``."""
    return per_channel(lambda channel: filters.gaussian(channel, sigma=sigma), pixels)


def median(pixels: np.ndarray, size: int) -> np.ndarray:
    """Return ``pixels`` with each value replaced by the median of its ``size`` x ``size`` window.

    Values are rounded to 8-bit levels first. At the borders the window only
    covers the pixels inside the image.
    """
    if size < 1:
        raise ValueError("The median window needs at least one pixel.")
    # The window: a size x size square of ones marks which neighbors are used
    footprint = np.ones((size, size), dtype=np.uint8)
    # filters.median sorts the neighbors of every pixel, which takes minutes for a
    # large window. filters.rank.median does not sort: it counts how many neighbors
    # have each of the 256 gray levels and updates that count as the window moves.
    # It is much faster, but it only accepts whole numbers from 0 to 255, so the
    # pixels go from the 0-1 scale to 0-255 here (0.2 -> 51) ...
    levels = np.round(pixels * 255.0).astype(np.uint8)
    # ... and back to the 0-1 scale after the filter (51 -> 0.2)
    return per_channel(lambda channel: filters.rank.median(channel, footprint), levels) / 255.0


def edges(pixels: np.ndarray, operator: str) -> np.ndarray:
    """Return the gradient magnitude of ``pixels`` under sobel, prewitt or scharr."""
    if operator not in EDGE_OPERATORS:
        raise ValueError(
            f"Unsupported edge operator {operator!r}. Choose one of: {', '.join(EDGE_OPERATORS)}."
        )
    return per_channel(EDGE_OPERATORS[operator], pixels)


def otsu(pixels: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the grayscale ``pixels`` binarized to 0.0 / 1.0, and the Otsu threshold used."""
    gray = to_gray(pixels)
    threshold = float(filters.threshold_otsu(gray))
    return (gray > threshold).astype(float), threshold


def apply_filter(
    image: Image.Image, name: str, grayscale: bool = False, **parameters: float
) -> tuple[Image.Image, dict[str, str]]:
    """Return ``image`` filtered with ``name``, plus what the filter measured.

    ``parameters`` are ``sigma`` for gaussian, ``size`` for median, and ``radius``
    and ``amount`` for unsharp_mask; omitted ones use the defaults. ``grayscale``
    makes the edge operators work on the gray version of a color image instead
    of on each channel. The result is 8-bit L or RGB (LA or RGBA if the image
    has transparency).
    Raises ValueError if ``name`` is not in FILTER_LABELS.
    """
    if name not in FILTER_LABELS:
        raise ValueError(f"Unsupported filter {name!r}. Choose one of: {', '.join(FILTER_LABELS)}.")
    pixels, alpha = image_to_unit_array(image)
    measured: dict[str, str] = {}

    if name == GAUSSIAN:
        result = gaussian(pixels, parameters.get("sigma", DEFAULT_SIGMA))
    elif name == MEDIAN:
        result = median(pixels, int(parameters.get("size", DEFAULT_MEDIAN_SIZE)))
    elif name in EDGE_OPERATORS:
        result = edges(to_gray(pixels) if grayscale else pixels, name)
    elif name == UNSHARP:
        result = sharpen_skimage(
            pixels,
            parameters.get("radius", DEFAULT_RADIUS),
            parameters.get("amount", DEFAULT_AMOUNT),
        )
    else:
        result, threshold = otsu(pixels)
        measured["Threshold"] = f"{threshold * 255:.1f} of 255"
    return unit_array_to_image(result, alpha), measured


class FiltersTool(ForensicsTool):
    tool_id = "filters"
    title = "Filters"
    category = "Filtering"
    description = "Apply a skimage filter: Gaussian, median, Sobel, Prewitt, Scharr, unsharp mask or Otsu."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        source = document.current

        names = tuple(FILTER_LABELS)
        index = ask_choice(parent, self.title, "Filter:", tuple(FILTER_LABELS.values()))
        if index is None:
            return None
        name = names[index]

        parameters = self._ask_parameters(parent, name, min(source.size))
        if parameters is None:
            return None

        details = {"Operation": self.title, "Filter": FILTER_LABELS[name]}
        grayscale = False
        if name in EDGE_OPERATORS and not is_grayscale(source):
            choice = ask_choice(parent, self.title, "Detect the edges on:", EDGE_SOURCE_LABELS)
            if choice is None:
                return None
            grayscale = choice == 0
            details["Edges on"] = EDGE_SOURCE_LABELS[choice]

        output, measured = apply_filter(source, name, grayscale=grayscale, **parameters)

        details.update({key.capitalize(): value for key, value in parameters.items()})
        details.update(measured)
        details["Mode"] = f"{source.mode} -> {output.mode}"
        return ToolResult(
            image=output,
            message=f"Applied the {name} filter.",
            details=details,
        )

    def _ask_parameters(self, parent: tk.Misc, name: str, shortest_side: int) -> dict[str, float] | None:
        """Ask for the parameters of filter ``name``, one dialog each; None if cancelled."""
        if name == GAUSSIAN:
            questions = [("sigma", f"Sigma (0-{MAX_SIGMA:g}):", DEFAULT_SIGMA, MAX_SIGMA)]
        elif name == MEDIAN:
            largest = min(MAX_MEDIAN_SIZE, shortest_side)
            size = simpledialog.askinteger(
                self.title,
                f"Window size in pixels (1-{largest}):",
                initialvalue=min(DEFAULT_MEDIAN_SIZE, largest),
                minvalue=1,
                maxvalue=largest,
                parent=parent,
            )
            return None if size is None else {"size": size}
        elif name == UNSHARP:
            questions = [
                ("radius", f"Blur radius (0-{MAX_SIGMA:g}):", DEFAULT_RADIUS, MAX_SIGMA),
                ("amount", f"Amount of detail to add (0-{MAX_AMOUNT:g}):", DEFAULT_AMOUNT, MAX_AMOUNT),
            ]
        else:
            questions = []

        parameters: dict[str, float] = {}
        for key, prompt, default, largest in questions:
            value = simpledialog.askfloat(
                self.title, prompt, initialvalue=default, minvalue=0.0, maxvalue=largest, parent=parent
            )
            if value is None:
                return None
            parameters[key] = value
        return parameters
