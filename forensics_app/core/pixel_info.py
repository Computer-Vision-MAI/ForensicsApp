"""Readable value of a single pixel, used by the pixel inspector."""

from __future__ import annotations

from PIL import Image


def pixel_color(image: Image.Image, column: int, row: int) -> tuple[int, int, int] | None:
    """Return the RGB color of a pixel, or None for modes without a direct color (I, F...)."""
    value = image.getpixel((column, row))
    if image.mode == "P":
        start = 3 * value
        return tuple(image.getpalette()[start:start + 3])
    if isinstance(value, tuple):
        return value[:3]
    if image.mode in ("L", "1"):
        return (value, value, value)
    return None

def describe_pixel(image: Image.Image, column: int, row: int) -> str:
    """Return e.g `x 120, y 45 | RGB (200, 20, 30)` for the pixel at (column, row) in the image."""
    value = image.getpixel((column, row))
    position = f"x {column}, y {row}"
    if image.mode == "P":
        return f"{position} | P index {value} -> RGB {pixel_color(image, column, row)}"
    if isinstance(value, float):
        return f"{position} | {image.mode} {value:.3f}"
    return f"{position} | {image.mode} {value}"