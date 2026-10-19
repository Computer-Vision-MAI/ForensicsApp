"""Find a person in the working image by comparing HOG descriptors with a template."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from forensics_app.core.channels import NUMERIC_MODES
from .base import ForensicsTool, ToolResult
from .convert_mode import rescale_to_8bit
from .dialogs import load_image_file
from .hog import HogParameters, ask_hog_parameters, compute_hog
from .template_matching import prepare_matched_arrays

# The values of the course example: 16 bins, 8 x 8 cells, 2 x 2 blocks
DEFAULT_PARAMETERS = HogParameters(orientations=16, pixels_per_cell=8, cells_per_block=2)
DEFAULT_STEP = 4
# Every window needs its own HOG and the search runs on the interface thread:
# 2000 windows take a few seconds
MAX_WINDOWS = 2000

EUCLIDEAN_COLOR = (255, 0, 0)
DOT_COLOR = (0, 0, 255)


@dataclass(frozen=True)
class Detection:
    """The window whose top-left corner is at (``top``, ``left``) and its ``score``."""

    top: int
    left: int
    score: float


def _offsets(length: int, window: int, step: int) -> range:
    """Return the start of every window along one side, including the last one that fits."""
    if step < 1:
        raise ValueError("The step must be at least one pixel.")
    if window > length:
        raise ValueError(f"The template ({window} pixels) does not fit in the image ({length} pixels).")
    # + 1 so a window that ends exactly at the border is tried too
    return range(0, length - window + 1, step)


def window_positions(
    image_shape: tuple[int, ...], window_shape: tuple[int, ...], step: int
) -> list[tuple[int, int]]:
    """Return the (row, column) of the top-left corner of every window, ``step`` pixels apart.

    Raises ValueError if the window is larger than the image or ``step`` is below 1.
    """
    rows = _offsets(image_shape[0], window_shape[0], step)
    cols = _offsets(image_shape[1], window_shape[1], step)
    return [(row, col) for row in rows for col in cols]


def smallest_step(image_shape: tuple[int, ...], window_shape: tuple[int, ...]) -> int:
    """Return the smallest step that gives at most MAX_WINDOWS windows."""
    free_rows = len(_offsets(image_shape[0], window_shape[0], 1))
    free_cols = len(_offsets(image_shape[1], window_shape[1], 1))
    step = 1
    while math.ceil(free_rows / step) * math.ceil(free_cols / step) > MAX_WINDOWS:
        step += 1
    return step


def detect_person(
    pixels: np.ndarray, template: np.ndarray, parameters: HogParameters, step: int
) -> tuple[Detection, Detection, int]:
    """Slide a template-sized window over ``pixels`` and compare its HOG with the template's.

    Returns the window closest to the template by Euclidean distance (the smallest),
    the one most similar by dot product (the largest), and how many windows were tried.
    ``pixels`` and ``template`` must both be gray (H x W) or both color (H x W x 3).
    """
    height, width = template.shape[:2]
    positions = window_positions(pixels.shape, template.shape, step)
    template_descriptor, _ = compute_hog(template, parameters)

    distances = []
    products = []
    for row, col in positions:
        window = pixels[row:row + height, col:col + width]
        descriptor, _ = compute_hog(window, parameters)
        # 0 if the window has exactly the template's gradients, larger the more they differ
        distances.append(np.linalg.norm(template_descriptor - descriptor))
        # Large when strong gradients of the window point the same way as the template's
        products.append(np.dot(descriptor, template_descriptor))

    closest = int(np.argmin(distances))
    most_similar = int(np.argmax(products))
    return (
        Detection(*positions[closest], float(distances[closest])),
        Detection(*positions[most_similar], float(products[most_similar])),
        len(positions),
    )


def draw_detections(
    image: Image.Image,
    window_size: tuple[int, int],
    detections: Sequence[tuple[Detection, tuple[int, int, int], int]],
) -> Image.Image:
    """Return an RGB copy of ``image`` with a rectangle per (detection, color, line width).

    ``window_size`` is (width, height). Lines grow inwards from the window border,
    so a wide line drawn first stays visible around a thinner one on the same window.
    """
    if image.mode in NUMERIC_MODES:
        image = rescale_to_8bit(image)
    # Pillow cannot convert La to other modes directly
    if image.mode == "La":
        image = image.convert("LA")
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = window_size
    for detection, color, line in detections:
        box = (detection.left, detection.top, detection.left + width - 1, detection.top + height - 1)
        draw.rectangle(box, outline=color, width=line)
    return canvas


class PersonDetectionTool(ForensicsTool):
    tool_id = "person_detection"
    title = "Person detection (HOG)"
    category = "Feature Detection"
    description = "Find the window most similar to a person template by comparing HOG descriptors."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        source = document.current

        loaded = load_image_file(parent, title="Open person template")
        if loaded is None:
            return None
        template_path, template_image = loaded

        pixels, template = prepare_matched_arrays(source, template_image)
        try:
            lowest_step = smallest_step(pixels.shape, template.shape)
        except ValueError as error:
            messagebox.showerror("Processing Error", str(error), parent=parent)
            return None

        parameters = ask_hog_parameters(parent, self.title, DEFAULT_PARAMETERS, min(template_image.size))
        if parameters is None:
            return None
        # A step as long as the template already leaves no gap between windows
        largest_step = max(lowest_step, max(template_image.size))
        step = simpledialog.askinteger(
            self.title,
            f"Step between windows in pixels ({lowest_step}-{largest_step}):",
            initialvalue=max(DEFAULT_STEP, lowest_step),
            minvalue=lowest_step,
            maxvalue=largest_step,
            parent=parent,
        )
        if step is None:
            return None

        euclidean, dot, windows = detect_person(pixels, template, parameters, step)
        # Euclidean first and wider, so it still shows if both pick the same window
        output = draw_detections(
            source, template_image.size, [(euclidean, EUCLIDEAN_COLOR, 4), (dot, DOT_COLOR, 2)]
        )
        same = (euclidean.top, euclidean.left) == (dot.top, dot.left)
        details = {
            "Operation": self.title,
            "Template": f"{template_path.name} ({template_image.width} x {template_image.height})",
            "Orientations": parameters.orientations,
            "Pixels per cell": f"{parameters.pixels_per_cell} x {parameters.pixels_per_cell}",
            "Cells per block": f"{parameters.cells_per_block} x {parameters.cells_per_block}",
            "Step": step,
            "Windows": windows,
            "Euclidean (red)": f"x {euclidean.left}, y {euclidean.top} | distance {euclidean.score:.3f}",
            "Dot product (blue)": f"x {dot.left}, y {dot.top} | score {dot.score:.3f}",
            "Same window": "yes" if same else "no",
        }
        return ToolResult(
            image=output,
            message=(
                f"Person found at x {euclidean.left}, y {euclidean.top} (Euclidean) "
                f"and x {dot.left}, y {dot.top} (dot product)."
            ),
            details=details,
        )
