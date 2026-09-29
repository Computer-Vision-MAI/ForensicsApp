"""Where an image preview is drawn on canvas, and how to map points back to it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePlacement:
    """A preview of a ``source_size`` image drawn at (left, top) with ``shown_size``."""

    left: int
    top: int
    shown_size: tuple[int, int]
    source_size: tuple[int, int]

    def to_image_coords(self, x: float, y: float) -> tuple[float, float] | None:
        """Return the source pixel (column, row) under canvas point (x, y), or None if outside."""
        shown_width, shown_height = self.shown_size
        source_width, source_height = self.source_size

        inside_x = x - self.left
        inside_y = y - self.top
        if not (0 <= inside_x < shown_width and 0 <= inside_y < shown_height):
            return None

        column = int(inside_x * source_width / shown_width)
        row = int(inside_y * source_height / shown_height)
        return column, row

    