"""Tk dialogs of the Bag of Visual Words tools: training settings and the Classify window."""

from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from .bow_model import BowSettings

DETECTOR_CHOICES = {"SIFT keypoints (DoG)": "sift", "Dense DAISY grid": "dense"}
ASSIGNMENT_CHOICES = {"Hard (nearest word)": "hard", "Soft (Gaussian weights)": "soft"}
WEIGHTING_CHOICES = {"TF-IDF": "tfidf", "TF (frequency only)": "tf"}
NORM_CHOICES = {"L1 (bins sum to 1)": "l1", "L2 (unit length)": "l2"}
PYRAMID_CHOICES = {"2 levels: 1+4+16 cells": 2, "1 level: 1+4 cells": 1, "Off: whole image": 0}


def _label_for(choices: dict[str, object], value: object) -> str:
    return next(label for label, option in choices.items() if option == value)


def parse_k(text: str) -> int | None:
    """Empty text means the course default (10 x classes); otherwise a whole number >= 2."""
    text = text.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"K must be a whole number (got '{text}').") from error
    if value < 2:
        raise ValueError(f"K must be at least 2 (got {value}).")
    return value


class BowSettingsDialog(simpledialog.Dialog):
    """One window with every choice of the BoW pipeline; ``result`` is a BowSettings or None."""

    def __init__(self, parent: tk.Misc, title: str, defaults: BowSettings) -> None:
        self._defaults = defaults
        self.result: BowSettings | None = None
        super().__init__(parent, title)

    def _combo(self, master: tk.Frame, row: int, text: str, choices: dict[str, object], value: object) -> tk.StringVar:
        ttk.Label(master, text=text).grid(row=row, column=0, sticky="w", pady=2)
        variable = tk.StringVar(master, value=_label_for(choices, value))
        ttk.Combobox(master, textvariable=variable, values=list(choices), state="readonly", width=28).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        return variable

    def body(self, master: tk.Frame) -> tk.Widget:
        d = self._defaults
        self._detector = self._combo(master, 0, "Local features:", DETECTOR_CHOICES, d.detector)

        ttk.Label(master, text="Vocabulary size K:").grid(row=1, column=0, sticky="w", pady=2)
        self._k = tk.StringVar(master, value="" if d.k is None else str(d.k))
        k_entry = ttk.Entry(master, textvariable=self._k, width=10)
        k_entry.grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(master, text="(empty = 10 x number of classes)", foreground="gray").grid(
            row=2, column=1, sticky="w"
        )
        self._auto_k = tk.BooleanVar(master, value=d.auto_k)
        multipliers = ", ".join(str(m) for m in d.k_multipliers)
        ttk.Checkbutton(
            master, text=f"Choose K by cross-validation ({multipliers} x classes, slower)", variable=self._auto_k
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)

        self._assignment = self._combo(master, 4, "Word assignment:", ASSIGNMENT_CHOICES, d.assignment)
        self._weighting = self._combo(master, 5, "Weighting:", WEIGHTING_CHOICES, d.weighting)
        self._norm = self._combo(master, 6, "Normalisation:", NORM_CHOICES, d.norm)
        self._pyramid = self._combo(master, 7, "Spatial pyramid:", PYRAMID_CHOICES, d.pyramid_levels)

        ttk.Label(master, text="Test images (%):").grid(row=8, column=0, sticky="w", pady=2)
        self._test = tk.IntVar(master, value=round(d.test_ratio * 100))
        ttk.Spinbox(master, from_=5, to=50, textvariable=self._test, width=8).grid(row=8, column=1, sticky="w", pady=2)
        ttk.Label(master, text="Random seed:").grid(row=9, column=0, sticky="w", pady=2)
        self._seed = tk.IntVar(master, value=d.seed)
        ttk.Spinbox(master, from_=0, to=9999, textvariable=self._seed, width=8).grid(row=9, column=1, sticky="w", pady=2)
        master.columnconfigure(1, weight=1)
        return k_entry

    def _read(self) -> BowSettings:
        try:
            test_percent = int(self._test.get())
            seed = int(self._seed.get())
        except (tk.TclError, ValueError) as error:
            raise ValueError("The test percentage and the seed must be whole numbers.") from error
        return BowSettings(
            detector=DETECTOR_CHOICES[self._detector.get()],
            k=parse_k(self._k.get()),
            auto_k=bool(self._auto_k.get()),
            k_multipliers=self._defaults.k_multipliers,
            test_ratio=test_percent / 100,
            seed=seed,
            assignment=ASSIGNMENT_CHOICES[self._assignment.get()],
            soft_sigma=self._defaults.soft_sigma,
            weighting=WEIGHTING_CHOICES[self._weighting.get()],
            norm=NORM_CHOICES[self._norm.get()],
            pyramid_levels=PYRAMID_CHOICES[self._pyramid.get()],
            max_side=self._defaults.max_side,
            max_kmeans_samples=self._defaults.max_kmeans_samples,
        )

    def validate(self) -> bool:
        try:
            self._settings = self._read()
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error), parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self._settings


def ask_bow_settings(parent: tk.Misc, title: str, defaults: BowSettings | None = None) -> BowSettings | None:
    return BowSettingsDialog(parent, title, defaults or BowSettings()).result


def _fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(1.0, max_width / image.width, max_height / image.height)
    if scale >= 1.0:
        return image
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)


class ClassifyWindow(tk.Toplevel):
    """Shows the histogram of visual words; the Classify button runs the trained classifier."""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        histogram: Image.Image,
        on_classify: Callable[[], list[tuple[str, float]]],
        warning: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.ranked: list[tuple[str, float]] | None = None
        self._on_classify = on_classify

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)
        if warning:
            tk.Label(frame, text=f"⚠ {warning}", fg="#c62828", font=("TkDefaultFont", 11, "bold")).pack(pady=(0, 6))
        self._headline = ttk.Label(frame, text="Press Classify to run the trained classifier.", font=("TkDefaultFont", 16, "bold"))
        self._headline.pack(anchor="center", pady=(0, 4))
        self._others = ttk.Label(frame, text="", foreground="gray")
        self._others.pack(anchor="center")

        fitted = _fit(histogram, int(self.winfo_screenwidth() * 0.85), int(self.winfo_screenheight() * 0.7))
        self._photo = ImageTk.PhotoImage(fitted)  # keep a reference so Tk doesn't drop it
        ttk.Label(frame, image=self._photo).pack(pady=8)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        self._classify_button = ttk.Button(buttons, text="Classify", command=self._classify)
        self._classify_button.pack(side="left")
        ttk.Button(buttons, text="Close", command=self.destroy).pack(side="right")
        self.bind("<Return>", lambda _event: self._classify())
        self.bind("<Escape>", lambda _event: self.destroy())
        self._classify_button.focus_set()

    def _classify(self) -> None:
        if self.ranked is not None:
            return
        self.ranked = self._on_classify()
        best, probability = self.ranked[0]
        self._headline.configure(text=f"p({best}) = {probability:.2f}")
        self._others.configure(text="   ".join(f"p({name}) = {p:.2f}" for name, p in self.ranked[1:]))
        self._classify_button.configure(state="disabled")


def show_classify_window(
    parent: tk.Misc,
    title: str,
    histogram: Image.Image,
    on_classify: Callable[[], list[tuple[str, float]]],
    warning: str | None = None,
) -> list[tuple[str, float]] | None:
    """Open the modal Classify window; return the ranking, or None if it was closed first."""
    window = ClassifyWindow(parent, title, histogram, on_classify, warning)
    window.grab_set()
    parent.wait_window(window)
    return window.ranked
