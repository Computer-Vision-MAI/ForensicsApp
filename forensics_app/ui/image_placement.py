"""Where an image preview is drawn on canvas, and how to map points back to it."""

from __future__ import annotations

from dataclasses import dataclass
import math


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

    def visible_region(
        self, view: tuple[float, float, float, float]
    ) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int]] | None:
        """Return the part of the image inside the canvas ``view`` (x0, y0, x1, y1).

        The result is ``(source_box, canvas_position, shown_size)``: crop ``source_box``
        from the source, resize it to ``shown_size`` and draw it at ``canvas_position``.
        Returns None when no part of the image is visible.
        """
        view_x0, view_y0, view_x1, view_y1 = view
        shown_width, shown_height = self.shown_size
        source_width, source_height = self.source_size
        scale_x = shown_width / source_width
        scale_y = shown_height / source_height

        x0 = max(view_x0, self.left)
        x1 = min(view_x1, self.left + shown_width)
        y0 = max(view_y0, self.top)
        y1 = min(view_y1, self.top + shown_height)
        if x0 >= x1 or y0 >= y1:
            return None

        # Whole source pixels that touch the visible area.
        col0 = max(0, math.floor((x0 - self.left) / scale_x))
        col1 = min(source_width, math.ceil((x1 - self.left) / scale_x))
        row0 = max(0, math.floor((y0 - self.top) / scale_y))
        row1 = min(source_height, math.ceil((y1 - self.top) / scale_y))

        left = round(col0 * scale_x)
        top = round(row0 * scale_y)
        size = (max(1, round(col1 * scale_x) - left), max(1, round(row1 * scale_y) - top))
        return (col0, row0, col1, row1), (self.left + left, self.top + top), size

    