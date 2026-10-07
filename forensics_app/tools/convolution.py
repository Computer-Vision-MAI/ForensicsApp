"""Convolve the working image with a preset kernel of any size."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from scipy import signal

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import ask_choice
from .image_utils import image_to_unit_array, per_channel, unit_array_to_image

MEAN = "mean"
GAUSSIAN = "gaussian"
SHARPEN = "sharpen"
HIGH_PASS = "high_pass"
GRADIENT_HORIZONTAL = "gradient_horizontal"
GRADIENT_VERTICAL = "gradient_vertical"
RANDOM = "random"

KERNEL_LABELS = {
    MEAN: "Mean: equal weights (blur)",
    GAUSSIAN: "Gaussian: weights fall off from the center (soft blur)",
    SHARPEN: "Sharpen: negative weights, strong positive center",
    HIGH_PASS: "High-pass: negative weights, center cancels them (edges)",
    GRADIENT_HORIZONTAL: "Gradient: negative left half, positive right half",
    GRADIENT_VERTICAL: "Gradient: negative top half, positive bottom half",
    RANDOM: "Random weights",
}

DEFAULT_KERNEL_SIZE = 15
# An exact 8-bit level, so a signed 0 is always drawn as 128
MID_GRAY = 128 / 255


def mean_kernel(rows: int, cols: int) -> np.ndarray:
    """Return equal weights that add up to 1."""
    return np.ones((rows, cols)) / (rows * cols)


def gaussian_kernel(rows: int, cols: int) -> np.ndarray:
    """Return bell-shaped weights that add up to 1; each side spans about 3 sigmas."""

    def bell(size: int) -> np.ndarray:
        # Distance of each cell to the middle one. For size 3: -1, 0, 1
        offsets = np.arange(size) - (size - 1) / 2
        # High in the middle, low at the ends. For size 3: 0.14, 1, 0.14
        return np.exp(-0.5 * (offsets / (size / 6)) ** 2)

    # Multiply the vertical bell by the horizontal one to fill the whole grid:
    # the center gets the largest weight and the corners the smallest
    kernel = np.outer(bell(rows), bell(cols))
    # Divide by the total so the weights add up to 1 and the brightness is kept.
    # A 3 x 3 kernel ends up as:
    #   0.01  0.08  0.01
    #   0.08  0.62  0.08
    #   0.01  0.08  0.01
    return kernel / kernel.sum()


def sharpen_kernel(rows: int, cols: int) -> np.ndarray:
    """Return -1/N everywhere plus 2 at the center, adding up to 1.

    It doubles the pixel and subtracts the local mean, so differences grow.
    """
    # Start with every weight at -1/N: this subtracts the mean of the neighbors
    kernel = -mean_kernel(rows, cols)
    # Add 2 to the center cell: this adds the pixel itself twice.
    # The result is 2 x pixel - mean, so a pixel brighter than its neighbors
    # gets even brighter, and a darker one gets even darker
    kernel[rows // 2, cols // 2] += 2.0
    return kernel


def high_pass_kernel(rows: int, cols: int) -> np.ndarray:
    """Return -1/N everywhere plus 1 at the center, adding up to 0.

    It subtracts the local mean from the pixel, so flat areas give 0.
    Raises ValueError for a 1 x 1 kernel, which would be a single zero.
    """
    if rows * cols == 1:
        raise ValueError("A high-pass kernel needs more than one cell.")
    # Start with every weight at -1/N: this subtracts the mean of the neighbors
    kernel = -mean_kernel(rows, cols)
    # Add 1 to the center cell: this adds the pixel itself once.
    # The result is pixel - mean: 0 in a flat area, and not 0 where there is a change
    kernel[rows // 2, cols // 2] += 1.0
    return kernel


def gradient_kernel(rows: int, cols: int, horizontal: bool) -> np.ndarray:
    """Return a kernel that is negative on one half and positive on the other, adding up to 0.

    The halves are left and right if ``horizontal``, otherwise top and bottom; an
    odd side leaves the middle line at 0. Raises ValueError if that side has one cell.
    """
    size = cols if horizontal else rows
    if size < 2:
        raise ValueError(f"This gradient kernel needs at least 2 {'columns' if horizontal else 'rows'}.")
    # -1 for the first half of the cells and +1 for the second half.
    # For size 4: 0 1 2 3  ->  -1.5 -0.5 0.5 1.5  ->  -1 -1 1 1
    signs = np.sign(np.arange(size) - (size - 1) / 2)
    # Repeat that line to fill the grid: as rows (left/right halves) or as columns (top/bottom)
    kernel = np.tile(signs, (rows, 1)) if horizontal else np.tile(signs[:, np.newaxis], (1, cols))
    # Scale so the positive half adds up to 1 and the negative half to -1.
    # In a flat area both halves cancel and give 0; across an edge they do not
    return kernel / kernel[kernel > 0].sum()


def random_kernel(rows: int, cols: int, seed: int | None = None) -> np.ndarray:
    """Return random positive weights that add up to 1; the same ``seed`` gives the same kernel."""
    kernel = np.random.default_rng(seed).random((rows, cols))
    # Divide by the total so the weights add up to 1 and the brightness is kept
    return kernel / kernel.sum()


def build_kernel(name: str, rows: int, cols: int, seed: int | None = None) -> np.ndarray:
    """Return the ``rows`` x ``cols`` preset kernel called ``name``.

    ``seed`` only matters for the random kernel. Raises ValueError if ``name`` is
    not in KERNEL_LABELS or the size is not valid for it.
    """
    if name not in KERNEL_LABELS:
        raise ValueError(f"Unsupported kernel {name!r}. Choose one of: {', '.join(KERNEL_LABELS)}.")
    if rows < 1 or cols < 1:
        raise ValueError("The kernel needs at least one row and one column.")
    if name == MEAN:
        return mean_kernel(rows, cols)
    if name == GAUSSIAN:
        return gaussian_kernel(rows, cols)
    if name == SHARPEN:
        return sharpen_kernel(rows, cols)
    if name == HIGH_PASS:
        return high_pass_kernel(rows, cols)
    if name == RANDOM:
        return random_kernel(rows, cols, seed)
    return gradient_kernel(rows, cols, horizontal=name == GRADIENT_HORIZONTAL)


def is_zero_sum(kernel: np.ndarray) -> bool:
    """Return True if the weights cancel out, so the convolution gives signed values."""
    return bool(abs(np.sum(kernel)) < 1e-9)


def convolve_pixels(pixels: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an H x W or H x W x C array with a 2-D ``kernel``.

    Borders are mirrored and each channel of a color image is convolved on its
    own. The result equals ``scipy.ndimage.convolve(pixels, kernel)``, but it is
    computed with the FFT so that large kernels stay fast.
    """
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2:
        raise ValueError("The kernel must be a 2-D matrix.")
    rows, cols = kernel.shape
    # Pixels at the border have no neighbors on one side, so the image is first
    # enlarged by mirroring its edges. A 1 x 3 kernel needs 1 extra pixel per side:
    #   5 7 9  ->  5 | 5 7 9 | 9
    # This is how many extra pixels go (before, after) each side, for rows and for columns
    border = ((rows - 1 - rows // 2, rows // 2), (cols - 1 - cols // 2, cols // 2))

    def convolve_channel(channel: np.ndarray) -> np.ndarray:
        padded = np.pad(channel, border, mode="symmetric")
        # "valid" returns only the pixels of the original image, not the mirrored ones.
        # ndimage.convolve multiplies and adds for every pixel, so a kernel twice as
        # big takes four times longer. fftconvolve gets the same numbers through the
        # Fourier transform, and takes about the same time for any kernel size
        return signal.fftconvolve(padded, kernel, mode="valid")

    return per_channel(convolve_channel, pixels)


def convolve_image(image: Image.Image, kernel: np.ndarray) -> Image.Image:
    """Return ``image`` convolved with ``kernel``.

    A kernel that adds up to 0 gives signed values, so 0 is drawn as mid-gray.
    The result is 8-bit L or RGB (LA or RGBA if the image has transparency).
    """
    pixels, alpha = image_to_unit_array(image)
    result = convolve_pixels(pixels, kernel)
    if is_zero_sum(kernel):
        # These kernels give negative and positive values around 0. Negative values
        # cannot be shown, so everything is shifted up: 0 becomes gray, negative
        # values become darker than gray and positive ones lighter
        result = result + MID_GRAY
    return unit_array_to_image(result, alpha)


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Filtering"
    description = "Convolve the image with a preset kernel (blur, sharpen, edges, random) of a chosen size."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        source = document.current
        width, height = source.size

        names = tuple(KERNEL_LABELS)
        index = ask_choice(parent, self.title, "Kernel:", tuple(KERNEL_LABELS.values()))
        if index is None:
            return None
        name = names[index]

        rows = simpledialog.askinteger(
            self.title,
            f"Kernel rows (height), 1-{height}:",
            initialvalue=min(DEFAULT_KERNEL_SIZE, height),
            minvalue=1,
            maxvalue=height,
            parent=parent,
        )
        if rows is None:
            return None
        cols = simpledialog.askinteger(
            self.title,
            f"Kernel columns (width), 1-{width}:",
            initialvalue=min(DEFAULT_KERNEL_SIZE, width),
            minvalue=1,
            maxvalue=width,
            parent=parent,
        )
        if cols is None:
            return None

        # Drawn here and shown in the results so a random kernel can be reproduced
        seed = int(np.random.default_rng().integers(1_000_000)) if name == RANDOM else None
        kernel = build_kernel(name, rows, cols, seed)
        output = convolve_image(source, kernel)

        details = {
            "Operation": self.title,
            "Kernel": KERNEL_LABELS[name],
            "Size": f"{rows} x {cols} (rows x columns)",
            "Weights": f"{kernel.min():.4g} to {kernel.max():.4g}, adding up to {kernel.sum():.4g}",
        }
        if seed is not None:
            details["Seed"] = seed
        if is_zero_sum(kernel):
            details["Values"] = "signed; 0 is drawn as mid-gray"
        details["Mode"] = f"{source.mode} -> {output.mode}"
        return ToolResult(
            image=output,
            message=f"Convolved the image with a {rows}x{cols} {name.replace('_', ' ')} kernel.",
            details=details,
        )
