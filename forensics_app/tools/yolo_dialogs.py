"""A small form dialog shared by the YOLO and U-Net tools (numbers, text, choices, check boxes, files, folders).

Each field is described by a ``Field``; the dialog validates the numbers when OK is
pressed and returns a ``{key: value}`` dict, or None when the user cancels. Parsing is
kept in ``parse_fields`` (no Tk) so it can be unit tested.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str  # "int", "float", "str", "choice", "check", "file", "dir", "note"
    default: Any = None
    options: Sequence[str] = ()  # for "choice": the texts shown; the value is the index
    minimum: float | None = None
    maximum: float | None = None
    filetypes: Sequence[tuple[str, str]] = (("All files", "*.*"),)
    help: str = ""


def parse_fields(fields: Sequence[Field], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the texts typed in the dialog into values; raises ValueError with a clear message."""
    values: dict[str, Any] = {}
    for f in fields:
        if f.kind == "note":
            continue
        value = raw.get(f.key, f.default)
        if f.kind in ("int", "float"):
            text = str(value).strip().replace(",", ".")
            try:
                number = int(text) if f.kind == "int" else float(text)
            except ValueError:
                kind = "a whole number" if f.kind == "int" else "a number"
                raise ValueError(f"{f.label} must be {kind}, got '{value}'.") from None
            if f.minimum is not None and number < f.minimum:
                raise ValueError(f"{f.label} must be at least {f.minimum:g}.")
            if f.maximum is not None and number > f.maximum:
                raise ValueError(f"{f.label} must be at most {f.maximum:g}.")
            values[f.key] = number
        elif f.kind == "choice":
            index = int(value)
            if not 0 <= index < len(f.options):
                raise ValueError(f"Choose a value for {f.label}.")
            values[f.key] = index
        elif f.kind == "check":
            values[f.key] = bool(value)
        elif f.kind in ("file", "dir"):
            text = str(value or "").strip()
            values[f.key] = Path(text) if text else None
        elif f.kind == "str":
            values[f.key] = str(value or "").strip()
        else:
            raise ValueError(f"Unknown field kind {f.kind!r}.")
    return values


class FormDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, title: str, fields: Sequence[Field], intro: str = "",
                 check: Callable[[dict[str, Any]], str | None] | None = None) -> None:
        self._fields = list(fields)
        self._intro = intro
        self._check = check
        self._vars: dict[str, tk.Variable] = {}
        self._parsed: dict[str, Any] | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget | None:
        row = 0
        first: tk.Widget | None = None
        if self._intro:
            ttk.Label(master, text=self._intro, wraplength=460, justify="left").grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
            row += 1
        for f in self._fields:
            if f.kind == "note":
                ttk.Label(master, text=f.label, wraplength=460, justify="left", foreground="#555").grid(
                    row=row, column=0, columnspan=3, sticky="w", pady=(4, 2))
                row += 1
                continue
            if f.kind == "check":
                var = tk.BooleanVar(master, value=bool(f.default))
                widget = ttk.Checkbutton(master, text=f.label, variable=var)
                widget.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
            else:
                ttk.Label(master, text=f.label + ":").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
                if f.kind == "choice":
                    var = tk.IntVar(master, value=int(f.default or 0))
                    widget = ttk.Combobox(master, values=list(f.options), state="readonly", width=46)
                    widget.current(int(f.default or 0))
                    widget.bind("<<ComboboxSelected>>", lambda _e, w=widget, v=var: v.set(w.current()))
                    widget.grid(row=row, column=1, columnspan=2, sticky="we", pady=2)
                elif f.kind in ("file", "dir"):
                    var = tk.StringVar(master, value="" if f.default is None else str(f.default))
                    widget = ttk.Entry(master, textvariable=var, width=40)
                    widget.grid(row=row, column=1, sticky="we", pady=2)
                    ttk.Button(master, text="Browse…", command=lambda v=var, ff=f: self._browse(v, ff)).grid(
                        row=row, column=2, padx=(4, 0))
                elif f.kind == "str":
                    var = tk.StringVar(master, value="" if f.default is None else str(f.default))
                    widget = ttk.Entry(master, textvariable=var, width=40)
                    widget.grid(row=row, column=1, columnspan=2, sticky="we", pady=2)
                else:
                    var = tk.StringVar(master, value=str(f.default))
                    widget = ttk.Entry(master, textvariable=var, width=12)
                    widget.grid(row=row, column=1, sticky="w", pady=2)
            if f.help:
                row += 1
                ttk.Label(master, text=f.help, foreground="#666", wraplength=440, justify="left").grid(
                    row=row, column=0, columnspan=3, sticky="w", padx=(12, 0))
            self._vars[f.key] = var
            first = first or widget
            row += 1
        master.columnconfigure(1, weight=1)
        return first

    def _browse(self, var: tk.StringVar, f: Field) -> None:
        current = Path(var.get()) if var.get() else None
        if f.kind == "dir":
            folder = filedialog.askdirectory(
                parent=self, title=f.label, mustexist=True,
                initialdir=str(current) if current and current.is_dir() else None,
            )
            if folder:
                var.set(folder)
            return
        name = filedialog.askopenfilename(
            parent=self, title=f.label, filetypes=list(f.filetypes),
            initialdir=str(current.parent) if current and current.parent.is_dir() else None,
        )
        if name:
            var.set(name)

    def validate(self) -> bool:
        try:
            parsed = parse_fields(self._fields, {k: v.get() for k, v in self._vars.items()})
            problem = self._check(parsed) if self._check else None
        except ValueError as error:
            messagebox.showwarning("Invalid value", str(error), parent=self)
            return False
        if problem:
            messagebox.showwarning("Invalid value", problem, parent=self)
            return False
        self._parsed = parsed
        return True

    def apply(self) -> None:
        self.result = self._parsed


def ask_form(parent: tk.Misc, title: str, fields: Sequence[Field], intro: str = "",
             check: Callable[[dict[str, Any]], str | None] | None = None) -> dict[str, Any] | None:
    """Show the form; returns the parsed values, or None if the user cancelled."""
    return FormDialog(parent, title, fields, intro, check).result
