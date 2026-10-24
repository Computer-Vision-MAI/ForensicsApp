"""Reusable modal dialogs for tools that need more than a single typed value."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from typing import Callable, TypeVar
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, UnidentifiedImageError

T = TypeVar("T")
# report(done, total, label) updates the progress bar; cancelled() tells the task to stop
ProgressReport = Callable[[int, int, str], None]

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def load_image_file(
    parent: tk.Misc,
    title: str = "Open image",
    filetypes: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
) -> tuple[Path, Image.Image] | None:
    """Open a file dialog to load an image, returning its Path and a copy of the Image."""
    filename = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes if filetypes is not None else OPEN_TYPES,
        parent=parent,
    )
    if not filename:
        return None

    try:
        source = Path(filename)
        with Image.open(source) as image:
            loaded = image.copy()
        return source, loaded
    except (OSError, UnidentifiedImageError) as error:
        messagebox.showerror("Could not open image", str(error), parent=parent)
        return None


class ChoiceDialog(simpledialog.Dialog):
    """Let the user pick exactly one option; ``result`` is its index or None."""

    def __init__(self, parent: tk.Misc, title: str, prompt: str, options: tuple[str, ...]) -> None:
        self._prompt = prompt
        self._choices = options
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        ttk.Label(master, text=self._prompt).pack(anchor="w", pady=(0, 6))
        self._choice = tk.IntVar(master, value=0)
        buttons = [
            ttk.Radiobutton(master, text=name, variable=self._choice, value=index)
            for index, name in enumerate(self._choices)
        ]
        for button in buttons:
            button.pack(anchor="w")
        return buttons[0]

    def apply(self) -> None:
        self.result = self._choice.get()


def ask_choice(parent: tk.Misc, title: str, prompt: str, options: tuple[str, ...]) -> int | None:
    """Show a ChoiceDialog and return the chosen index, or None if user cancelled."""
    return ChoiceDialog(parent, title, prompt, options).result


class OrderDialog(simpledialog.Dialog):
    """Let the user choose which source channel feeds each output channel."""

    def __init__(self, parent: tk.Misc, title: str, channels: tuple[str, ...], initial: tuple[int, ...]) -> None:
        self._channels = channels
        self._initial = initial
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        self._selectors: list[ttk.Combobox] = []
        for row, (name, start) in enumerate(zip(self._channels, self._initial)):
            ttk.Label(master, text=f"New {name} <-").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            selector = ttk.Combobox(master, values=self._channels, state="readonly", width=24)
            selector.current(start)
            selector.grid(row=row, column=1, pady=2)
            self._selectors.append(selector)
        return self._selectors[0]

    def validate(self) -> bool:
        order = [selector.current() for selector in self._selectors]
        if len(set(order)) != len(order):
            messagebox.showwarning("Invalid order", "Use each channel exactly once.", parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = tuple(selector.current() for selector in self._selectors)


def ask_channel_order(
        parent: tk.Misc, title: str, channels: tuple[str, ...], initial: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Show an OrderDialog and return the chosen order, or None if cancelled."""
    return OrderDialog(parent, title, channels, initial).result


class TaskCancelled(Exception):
    """Raised when the user cancels a task started with ``run_with_progress``."""


def run_with_progress(
    parent: tk.Misc | None,
    title: str,
    task: Callable[[ProgressReport, Callable[[], bool]], T],
) -> T:
    """Run ``task`` on a worker thread while a modal progress window keeps the UI alive.

    ``task(report, cancelled)`` must not touch Tkinter: it calls ``report(done, total,
    label)`` to move the bar and should raise ``TaskCancelled`` when ``cancelled()``
    becomes True. The window is updated from the Tk thread through ``after``.
    Exceptions raised by the task are re-raised here. With ``parent=None`` (tests)
    the task simply runs on the calling thread.
    """
    if parent is None:
        return task(lambda done, total, label: None, lambda: False)

    cancel = threading.Event()
    updates: queue.Queue[tuple[int, int, str] | None] = queue.Queue()
    outcome: dict[str, object] = {}

    window = tk.Toplevel(parent)
    window.title(title)
    window.transient(parent)
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)
    label = ttk.Label(frame, text="Starting…", width=55)
    label.pack(anchor="w")
    bar = ttk.Progressbar(frame, length=360, mode="determinate")
    bar.pack(fill="x", pady=8)
    def on_cancel() -> None:
        cancel.set()
        label.configure(text="Cancelling…")
        cancel_button.configure(state="disabled")

    cancel_button = ttk.Button(frame, text="Cancel", command=on_cancel)
    cancel_button.pack(anchor="e")
    window.protocol("WM_DELETE_WINDOW", on_cancel)

    def worker() -> None:
        try:
            outcome["result"] = task(lambda done, total, text: updates.put((done, total, text)), cancel.is_set)
        except BaseException as error:  # handed back to the Tk thread below
            outcome["error"] = error
        finally:
            updates.put(None)

    def poll() -> None:
        try:
            while True:
                item = updates.get_nowait()
                if item is None:
                    window.destroy()
                    return
                done, total, text = item
                bar.configure(maximum=max(total, 1), value=done)
                if not cancel.is_set():
                    label.configure(text=text)
        except queue.Empty:
            pass
        window.after(100, poll)

    threading.Thread(target=worker, daemon=True).start()
    window.after(100, poll)
    window.grab_set()
    parent.wait_window(window)

    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    if cancel.is_set():
        raise TaskCancelled()
    return outcome["result"]  # type: ignore[return-value]
