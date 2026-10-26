"""Scrollable canvas that displays a PIL image fitted to the window or at a fixed zoom.

Zoom only changes how the image is drawn: the source image is never resized or modified.
"""

from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .image_placement import ImagePlacement


_DISPLAY_CONVERSIONS = {"LAB": "RGB", "La": "LA"}

# Zoom steps used by the zoom buttons (1.0 = one screen pixel per image pixel).
ZOOM_LEVELS = (
    0.05, 0.1, 0.25, 1 / 3, 0.5, 2 / 3, 0.75,
    1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0,
)
# Empty space kept around the image, in screen pixels.
MARGIN = 12

ZoomCallback = Callable[[ImagePlacement | None, bool], None]


def _displayable(image: Image.Image) -> Image.Image:
    """Return ``image`` in a mode TK can draw, converting only when needed."""
    target = _DISPLAY_CONVERSIONS.get(image.mode)
    return image.convert(target) if target else image


def _scaled_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))


def _axis_layout(canvas_length: int, shown_length: int) -> tuple[int, int]:
    """Return (image offset, scroll length) for one axis.

    An image that fits is centered; a larger one starts after the margin and scrolls.
    """
    if shown_length + 2 * MARGIN <= canvas_length:
        return (canvas_length - shown_length) // 2, canvas_length
    return MARGIN, shown_length + 2 * MARGIN


class ImageView(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        on_hover: Callable[[tuple[int, int] | None], None] | None = None,
        on_zoom: ZoomCallback | None = None,
    ) -> None:
        super().__init__(parent, padding=8)
        self.canvas = tk.Canvas(
            self,
            background="#20242b",
            borderwidth=0,
            highlightthickness=0,
        )
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self._yview)
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self._xview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(sequence, self._on_wheel)

        self._init_state()
        self.show(None)
        # Connect callbacks last: the owner may not hold a reference to this view yet.
        self.on_hover = on_hover
        self.on_zoom = on_zoom

    def _init_state(
        self,
        on_hover: Callable[[tuple[int, int] | None], None] | None = None,
        on_zoom: ZoomCallback | None = None,
    ) -> None:
        self.on_hover = on_hover
        self.on_zoom = on_zoom
        self._source: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None  # keep a reference so Tk doesn't drop it
        self._photo_source: Image.Image | None = None
        self._photo_key: tuple | None = None
        self._resize_job: str | None = None
        self._placement: ImagePlacement | None = None
        self._region: tuple[int, int] = (1, 1)
        self._pointer: tuple[int, int] | None = None
        self._zoom: float | None = None  # None means "fit to window"

    # ------------------------------------------------------------------ public API

    def show(self, image: Image.Image | None) -> None:
        """Display ``image``. The zoom is kept only while the image size stays the same."""
        if image is None or self._source is None or image.size != self._source.size:
            self._zoom = None
        self._source = image
        self._render()

    @property
    def image(self) -> Image.Image | None:
        """The image currently on screen: the working image or a preview-only result."""
        return self._source

    @property
    def placement(self) -> ImagePlacement | None:
        return self._placement

    @property
    def is_fitted(self) -> bool:
        return self._zoom is None

    @property
    def scale(self) -> float | None:
        """Screen pixels per image pixel, or None without an image."""
        if self._source is None:
            return None
        return self._zoom if self._zoom is not None else self._fit_scale()

    def zoom_in(self) -> bool:
        current = self.scale
        if current is None:
            return False
        target = next((z for z in ZOOM_LEVELS if z > current * (1 + 1e-6)), None)
        if target is None:
            return False
        self._set_zoom(target)
        return True

    def zoom_out(self) -> bool:
        current = self.scale
        if current is None:
            return False
        target = next((z for z in reversed(ZOOM_LEVELS) if z < current * (1 - 1e-6)), None)
        if target is None:
            return False
        self._set_zoom(target)
        return True

    def reset_zoom(self) -> None:
        """Go back to fitting the whole image in the window."""
        self._zoom = None
        self._render()

    # ------------------------------------------------------------------ rendering

    def _canvas_size(self) -> tuple[int, int]:
        return max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)

    def _fit_scale(self) -> float:
        assert self._source is not None
        canvas_width, canvas_height = self._canvas_size()
        available_width = max(canvas_width - 2 * MARGIN, 1)
        available_height = max(canvas_height - 2 * MARGIN, 1)
        source_width, source_height = self._source.size
        # Like a thumbnail: shrink to fit, never enlarge.
        return min(available_width / source_width, available_height / source_height, 1.0)

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _render(self) -> None:
        self._resize_job = None
        self._layout()
        self._draw()
        self._notify_hover()

    def _set_zoom(self, zoom: float) -> None:
        anchor = self._view_center_in_source()
        self._zoom = zoom
        self._layout()
        if anchor is not None:
            self._center_on(*anchor)
        self._draw()
        self._notify_hover()

    def _layout(self) -> None:
        """Compute where the image goes and how far the canvas can scroll."""
        canvas_width, canvas_height = self._canvas_size()
        if self._source is None:
            self._placement = None
            self._region = (canvas_width, canvas_height)
        else:
            shown = _scaled_size(self._source.size, self.scale)
            left, region_width = _axis_layout(canvas_width, shown[0])
            top, region_height = _axis_layout(canvas_height, shown[1])
            self._placement = ImagePlacement(left, top, shown, self._source.size)
            self._region = (region_width, region_height)

        self.canvas.configure(scrollregion=(0, 0, *self._region))
        if self._region[0] <= canvas_width:
            self.canvas.xview_moveto(0)
        if self._region[1] <= canvas_height:
            self.canvas.yview_moveto(0)
        if self.on_zoom is not None:
            self.on_zoom(self._placement, self.is_fitted)

    def _visible_view(self) -> tuple[float, float, float, float]:
        canvas_width, canvas_height = self._canvas_size()
        x0 = self.canvas.canvasx(0)
        y0 = self.canvas.canvasy(0)
        return x0, y0, x0 + canvas_width, y0 + canvas_height

    def _draw(self) -> None:
        """Draw only the visible part of the image, so large zooms stay cheap."""
        self.canvas.delete("all")
        view = self._visible_view()
        if self._source is None or self._placement is None:
            self.canvas.create_text(
                (view[0] + view[2]) / 2,
                (view[1] + view[3]) / 2,
                text="Open an image to begin",
                fill="#c7ccd4",
                font=("TkDefaultFont", 16),
            )
            self._photo = None
            self._photo_source = None
            self._photo_key = None
            return

        region = self._placement.visible_region(view)
        if region is None:
            return
        box, position, size = region
        key = (self._placement.shown_size, box)
        if self._photo_source is not self._source or self._photo_key != key:
            self._photo = ImageTk.PhotoImage(_displayable(self._crop_and_scale(box, size)))
            self._photo_source = self._source
            self._photo_key = key
        self.canvas.create_image(*position, image=self._photo, anchor="nw")

    def _crop_and_scale(self, box: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
        assert self._source is not None
        part = self._source if box == (0, 0, *self._source.size) else self._source.crop(box)
        if part.size == size:
            return part
        if size[0] >= part.width:
            # Enlarging: show every image pixel as a sharp block.
            return part.resize(size, Image.Resampling.NEAREST)
        return part.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)

    # ------------------------------------------------------------------ scrolling

    def _view_center_in_source(self) -> tuple[float, float] | None:
        """The image point (in source pixels) at the center of the visible area."""
        if self._placement is None:
            return None
        x0, y0, x1, y1 = self._visible_view()
        shown_width, shown_height = self._placement.shown_size
        source_width, source_height = self._placement.source_size
        x = ((x0 + x1) / 2 - self._placement.left) * source_width / shown_width
        y = ((y0 + y1) / 2 - self._placement.top) * source_height / shown_height
        return min(max(x, 0.0), source_width), min(max(y, 0.0), source_height)

    def _center_on(self, source_x: float, source_y: float) -> None:
        """Scroll so the image point (source_x, source_y) is at the center of the view."""
        assert self._placement is not None
        canvas_width, canvas_height = self._canvas_size()
        shown_width, shown_height = self._placement.shown_size
        source_width, source_height = self._placement.source_size
        region_width, region_height = self._region
        if region_width > canvas_width:
            target_x = self._placement.left + source_x * shown_width / source_width
            self.canvas.xview_moveto(max(0.0, (target_x - canvas_width / 2) / region_width))
        if region_height > canvas_height:
            target_y = self._placement.top + source_y * shown_height / source_height
            self.canvas.yview_moveto(max(0.0, (target_y - canvas_height / 2) / region_height))

    def _xview(self, *args: str) -> None:
        self.canvas.xview(*args)
        self._after_scroll()

    def _yview(self, *args: str) -> None:
        self.canvas.yview(*args)
        self._after_scroll()

    def _after_scroll(self) -> None:
        self._draw()
        self._notify_hover()

    def _on_wheel(self, event: tk.Event) -> str | None:
        """Wheel scrolls vertically, Shift+wheel horizontally and Ctrl+wheel zooms."""
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        elif event.delta:
            step = -1 if event.delta > 0 else 1
        else:
            return None
        if event.state & 0x0004:  # Control
            self.zoom_in() if step < 0 else self.zoom_out()
        elif event.state & 0x0001:  # Shift
            self._xview("scroll", step, "units")
        else:
            self._yview("scroll", step, "units")
        return "break"

    # ------------------------------------------------------------------ hover

    def _on_motion(self, event: tk.Event) -> None:
        self._pointer = (event.x, event.y)
        self._notify_hover()

    def _on_leave(self, _event: tk.Event) -> None:
        self._pointer = None
        self._notify_hover()

    def _notify_hover(self) -> None:
        coords = None
        if self._pointer is not None and self._placement is not None:
            # The pointer is in window coordinates; the image lives in scrolled canvas coordinates.
            x = self.canvas.canvasx(self._pointer[0])
            y = self.canvas.canvasy(self._pointer[1])
            coords = self._placement.to_image_coords(x, y)
        self.canvas.configure(cursor="crosshair" if coords else "")
        if self.on_hover is not None:
            self.on_hover(coords)
