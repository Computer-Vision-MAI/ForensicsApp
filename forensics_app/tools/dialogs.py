"""Reusable modal dialogs for tools that need more than a single typed value."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk


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