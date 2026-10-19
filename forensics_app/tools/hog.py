"""Extract and draw the Histogram of Oriented Gradients (HOG) of the working image."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .image_utils import image_to_unit_array, unit_array_to_image

MAX_ORIENTATIONS = 36


@dataclass(frozen=True)
class HogParameters:
    """How the gradients are grouped: orientation bins, cell side and block side.

    Cells and blocks are square: ``pixels_per_cell`` pixels and ``cells_per_block``
    cells on each side.
    """

    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 3


def grid_size(shape: tuple[int, ...], parameters: HogParameters) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the (rows, columns) of cells and of blocks that fit in an image of ``shape``.

    Pixels left over at the right and bottom do not form a whole cell and are ignored.
    """
    cells = (shape[0] // parameters.pixels_per_cell, shape[1] // parameters.pixels_per_cell)
    # A block slides one cell at a time, so n cells hold n - (block side) + 1 blocks
    blocks = (cells[0] - parameters.cells_per_block + 1, cells[1] - parameters.cells_per_block + 1)
    return cells, blocks


def descriptor_length(shape: tuple[int, ...], parameters: HogParameters) -> int:
    """Return how many numbers the HOG descriptor of an image of ``shape`` has."""
    _, (block_rows, block_cols) = grid_size(shape, parameters)
    # Every block stores one histogram per cell it covers
    return block_rows * block_cols * parameters.cells_per_block**2 * parameters.orientations


def compute_hog(
    pixels: np.ndarray, parameters: HogParameters, visualize: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the HOG descriptor of an H x W or H x W x 3 array, and its drawing if ``visualize``.

    A color array uses, in every pixel, the gradient of the channel that changes most.
    Raises ValueError if the image is smaller than one block.
    """
    _, (block_rows, block_cols) = grid_size(pixels.shape, parameters)
    if block_rows < 1 or block_cols < 1:
        side = parameters.pixels_per_cell * parameters.cells_per_block
        raise ValueError(
            f"The image ({pixels.shape[1]} x {pixels.shape[0]}) is smaller than one HOG block "
            f"({side} x {side} pixels). Use smaller cells or fewer cells per block."
        )
    result = feature.hog(
        pixels,
        orientations=parameters.orientations,
        pixels_per_cell=(parameters.pixels_per_cell, parameters.pixels_per_cell),
        cells_per_block=(parameters.cells_per_block, parameters.cells_per_block),
        visualize=visualize,
        channel_axis=-1 if pixels.ndim == 3 else None,
    )
    if visualize:
        return result
    return result, None


def hog_image(image: Image.Image, parameters: HogParameters) -> tuple[Image.Image, dict[str, str]]:
    """Return the HOG drawing of ``image`` as an 8-bit L image, plus what was measured.

    Each cell shows a star of lines, one per orientation, brighter where that
    gradient direction is stronger. The drawing is stretched so the strongest line is white.
    """
    pixels, _ = image_to_unit_array(image)
    descriptor, drawing = compute_hog(pixels, parameters, visualize=True)
    # The lines are faint (most values are near 0), so they are rescaled to the full range.
    # A flat image has no gradients and stays black
    peak = drawing.max()
    if peak > 0:
        drawing = drawing / peak
    (cell_rows, cell_cols), (block_rows, block_cols) = grid_size(pixels.shape, parameters)
    measured = {
        "Descriptor length": str(descriptor.size),
        "Cells": f"{cell_rows} x {cell_cols} (rows x columns)",
        "Blocks": f"{block_rows} x {block_cols} (rows x columns)",
    }
    return unit_array_to_image(drawing), measured


def ask_hog_parameters(
    parent: tk.Misc, title: str, defaults: HogParameters, shortest_side: int
) -> HogParameters | None:
    """Ask for the orientations, cell size and block size, one dialog each; None if cancelled.

    The limits keep at least one whole block inside a ``shortest_side`` image.
    """
    orientations = simpledialog.askinteger(
        title,
        f"Orientation bins per cell (1-{MAX_ORIENTATIONS}):",
        initialvalue=defaults.orientations,
        minvalue=1,
        maxvalue=MAX_ORIENTATIONS,
        parent=parent,
    )
    if orientations is None:
        return None
    pixels_per_cell = simpledialog.askinteger(
        title,
        f"Pixels per cell side (1-{shortest_side}):",
        initialvalue=min(defaults.pixels_per_cell, shortest_side),
        minvalue=1,
        maxvalue=shortest_side,
        parent=parent,
    )
    if pixels_per_cell is None:
        return None
    # The block, cells_per_block cells per side, must still fit in the image
    largest_block = shortest_side // pixels_per_cell
    cells_per_block = simpledialog.askinteger(
        title,
        f"Cells per block side (1-{largest_block}):",
        initialvalue=min(defaults.cells_per_block, largest_block),
        minvalue=1,
        maxvalue=largest_block,
        parent=parent,
    )
    if cells_per_block is None:
        return None
    return HogParameters(orientations, pixels_per_cell, cells_per_block)


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG features"
    category = "Feature Detection"
    description = "Extract the Histogram of Oriented Gradients and draw it, with configurable bins, cells and blocks."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        source = document.current

        parameters = ask_hog_parameters(parent, self.title, HogParameters(), min(source.size))
        if parameters is None:
            return None

        output, measured = hog_image(source, parameters)
        details = {
            "Operation": self.title,
            "Orientations": parameters.orientations,
            "Pixels per cell": f"{parameters.pixels_per_cell} x {parameters.pixels_per_cell}",
            "Cells per block": f"{parameters.cells_per_block} x {parameters.cells_per_block}",
            **measured,
            "Mode": f"{source.mode} -> {output.mode}",
        }
        return ToolResult(
            image=output,
            message=f"Drew the HOG with {parameters.orientations} orientations.",
            details=details,
        )
