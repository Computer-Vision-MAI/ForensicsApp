"""Channel names of every Pillow image mode, and helpers for the channel tools.

Mode reference: https://pillow.readthedocs.io/en/latest/handbook/concepts.html
"""

from __future__ import annotations

from PIL import Image

ALPHA = "Alpha"
PADDING = "Padding"
FIXED_CHANNELS = frozenset([ALPHA, PADDING])

_SIXTEEN_BIT = ("Intensity (16-bit)",)

MODE_CHANNELS: dict[str, tuple[str, ...]] = {
    # Single channel
    "1": ("Binary",),
    "L": ("Intensity",),
    "I": ("Intensity (32-bit integer)",),
    "F": ("Intensity (32-bit float)",),
    "I;16": _SIXTEEN_BIT,
    "I;16L": _SIXTEEN_BIT,
    "I;16B": _SIXTEEN_BIT,
    "I;16N": _SIXTEEN_BIT,
    # Palette
    "P": ("Palette index",),
    "PA": ("Palette index", ALPHA),
    # Color
    "RGB": ("Red", "Green", "Blue"),
    "CMYK": ("Cyan", "Magenta", "Yellow", "Black"),
    "YCbCr": ("Luma (Y)", "Blue-difference (Cb)", "Red-difference (Cr)"),
    "LAB": ("Lightness (L)", "Green-red (a)", "Blue-yellow (b)"),
    "HSV": ("Hue", "Saturation", "Value"),
    # Color + alpha or padding
    "RGBA": ("Red", "Green", "Blue", ALPHA),
    "RGBX": ("Red", "Green", "Blue", PADDING),
    "RGBa": ("Red (premultiplied)", "Green (premultiplied)", "Blue (premultiplied)", ALPHA),
    "LA": ("Intensity", ALPHA),
    "La": ("Intensity (premultiplied)", ALPHA),
}


def to_multichannel(image: Image.Image) -> Image.Image:
    """Return ``image`` with at least two channels, expanding palettes to their colors.

    Raises ValueError for single-channel modes.
    """
    if image.mode in ("P", "PA"):
        has_alpha = image.mode == "PA" or "transparency" in image.info
        image = image.convert("RGBA" if has_alpha else "RGB")
    if len(MODE_CHANNELS[image.mode]) < 2:
        raise ValueError(
            f"Image mode {image.mode!r} has a single channel; there is nothing to split or swap. "
            "Use Undo or Reset to go back to a color image."
        )
    return image
