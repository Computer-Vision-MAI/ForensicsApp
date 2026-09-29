"""Canvas that displays a PIL image scaled to the available space."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from collections.abc import Callable

from .image_placement import ImagePlacement


class ImageView(ttk.Frame):
    def __init__(
        self, 
        parent: tk.Misc,
        on_hover: Callable[[tuple[int, int] | None], None] | None = None,
    ) -> None:
        super().__init__(parent, padding=8)
        self.canvas = tk.Canvas(
            self,
            background="#20242b",
            borderwidth=0,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self._source: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None
        self.on_hover = on_hover
        self._placement: ImagePlacement | None = None
        self._pointer: tuple[int, int] | None = None
        self.show(None)

    def show(self, image: Image.Image | None) -> None:
        self._source = image
        self._render()

    @property
    def image(self) -> Image.Image | None:
        """The image currently on screen: the working image or a preview-only result."""
        return self._source

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _render(self) -> None:
        self._resize_job = None
        self.canvas.delete("all")
        if self._source is None:
            self.canvas.create_text(
                max(self.canvas.winfo_width() // 2, 1),
                max(self.canvas.winfo_height() // 2, 1),
                text="Open an image to begin",
                fill="#c7ccd4",
                font=("TkDefaultFont", 16),
            )
            self._photo = None
            self._placement = None
            return

        available = (max(self.canvas.winfo_width() - 24, 1), max(self.canvas.winfo_height() - 24, 1))
        preview = self._source.copy()
        preview.thumbnail(available, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(preview)
        left = (self.canvas.winfo_width() - preview.width) // 2
        top = (self.canvas.winfo_height() - preview.height) // 2
        self.canvas.create_image(
            left,
            top,
            image=self._photo,
            anchor="nw",
        )
        self._placement = ImagePlacement(left, top, preview.size, self._source.size)
        self._notify_hover()

    def _on_motion(self, event: tk.Event) -> None:
        self._pointer = (event.x, event.y)
        self._notify_hover()

    def _on_leave(self, _event: tk.Event) -> None:
        self._pointer = None
        self._notify_hover()

    def _notify_hover(self) -> None:
        coords = None
        if self._pointer is not None and self._placement is not None:
            coords = self._placement.to_image_coords(*self._pointer)
        self.canvas.configure(cursor="crosshair" if coords else "")
        if self.on_hover is not None:
            self.on_hover(coords)