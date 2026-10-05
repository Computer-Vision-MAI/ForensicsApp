"""Readable value of a single pixel, used by the pixel inspector."""

from __future__ import annotations

from PIL import Image

from forensics_app.core.channels import NUMERIC_MODES


def pixel_color(image: Image.Image, column: int, row: int) -> tuple[int, int, int] | None:
    """Return the on-screen RGB color of a pixel, or None for numeric modes."""
    if image.mode in NUMERIC_MODES:
        return None
    pixel = image.crop((column, row, column + 1, row + 1))
    if pixel.mode == "La":
        pixel = pixel.convert("LA")
    return pixel.convert("RGB").getpixel((0, 0))


def describe_pixel(image: Image.Image, column: int, row: int) -> str:
    """Return e.g `x 120, y 45 | RGB (200, 20, 30)` for the pixel at (column, row) in the image."""
    value = image.getpixel((column, row))
    position = f"x {column}, y {row}"
    if image.mode == "P":
        return f"{position} | P index {value} -> RGB {pixel_color(image, column, row)}"
    if isinstance(value, float):
        return f"{position} | {image.mode} {value:.3f}"
    return f"{position} | {image.mode} {value}"