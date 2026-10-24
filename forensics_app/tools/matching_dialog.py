"""One form with every matching parameter, shared by Sets 5.3 and 5.4."""

from __future__ import annotations

from collections.abc import Mapping
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from .matching import METHODS, RANSAC_MODELS, WARP_OUTPUTS, MatchSettings

METHOD_LABELS = {"orb": "ORB (binary, fast)", "sift": "SIFT (float, more robust)"}
MODEL_LABELS = {"affine": "Affine (3 points)", "projective": "Projective / homography (4 points)"}

INT_FIELDS = ("n_keypoints", "n_octaves", "n_scales", "max_side", "max_trials")
FLOAT_FIELDS = ("sigma", "max_ratio", "residual_threshold")
FIELD_NAMES = {
    "n_keypoints": "Number of keypoints",
    "n_octaves": "Octaves",
    "n_scales": "Scales per octave",
    "max_side": "Working size",
    "max_trials": "Max trials",
    "sigma": "Sigma",
    "max_ratio": "Max ratio",
    "residual_threshold": "Residual threshold",
}


def parse_match_settings(values: Mapping[str, object]) -> MatchSettings:
    """Build MatchSettings from the raw form values, raising ValueError with a clear message.

    Numbers may arrive as text (what an Entry holds); ``method`` and
    ``ransac_model`` are keys of ``METHODS`` / ``RANSAC_MODELS``.
    """
    parsed: dict[str, object] = {}
    for name in INT_FIELDS + FLOAT_FIELDS:
        if name not in values:
            continue
        text = str(values[name]).strip()
        try:
            parsed[name] = int(text) if name in INT_FIELDS else float(text)
        except ValueError:
            kind = "a whole number" if name in INT_FIELDS else "a number"
            raise ValueError(f"{FIELD_NAMES[name]} must be {kind} (got '{text}').") from None
    for name in ("method", "ransac_model"):
        if name in values:
            parsed[name] = str(values[name]).lower()
    for name in ("cross_check", "use_ransac"):
        if name in values:
            parsed[name] = bool(values[name])
    return MatchSettings(**parsed)


class MatchSettingsDialog(simpledialog.Dialog):
    """Detector, matching, RANSAC and output options in one window.

    ``result`` is ``(MatchSettings, options)`` or None if cancelled, where
    ``options`` holds ``"output"`` and/or ``"recursive"`` when those were asked.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        initial: MatchSettings,
        output_modes: Mapping[str, str] | None = None,
        ask_recursive: bool = False,
    ) -> None:
        self._initial = initial
        self._output_modes = dict(output_modes or {})
        self._ask_recursive = ask_recursive
        self._parsed: tuple[MatchSettings, dict[str, object]] | None = None
        super().__init__(parent, title)

    # -- layout ---------------------------------------------------------------
    def body(self, master: tk.Frame) -> tk.Widget:
        init = self._initial
        self._vars: dict[str, tk.Variable] = {}
        self._widgets: dict[str, ttk.Widget] = {}

        detector = ttk.LabelFrame(master, text="Detector / descriptor", padding=6)
        detector.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._method = tk.StringVar(master, value=init.method)
        for column, method in enumerate(METHODS):
            ttk.Radiobutton(
                detector, text=METHOD_LABELS[method], value=method,
                variable=self._method, command=self._update_states,
            ).grid(row=0, column=column * 2, columnspan=2, sticky="w")
        self._entry(detector, 1, "n_keypoints", "ORB keypoints", init.n_keypoints)
        self._entry(detector, 2, "n_octaves", "SIFT octaves", init.n_octaves)
        self._entry(detector, 3, "n_scales", "SIFT scales per octave", init.n_scales)
        self._entry(detector, 4, "sigma", "SIFT sigma (≥ 1.0)", init.sigma)
        self._entry(detector, 5, "max_side", "Working size, longest side px (0 = full)", init.max_side)

        matching = ttk.LabelFrame(master, text="Matching", padding=6)
        matching.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self._check(matching, 0, "cross_check", "Cross-check (keep mutual best matches only)", init.cross_check)
        self._entry(matching, 1, "max_ratio", "Max ratio (Lowe test, 1 = off)", init.max_ratio)

        ransac = ttk.LabelFrame(master, text="Outlier rejection", padding=6)
        ransac.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self._check(ransac, 0, "use_ransac", "Use RANSAC", init.use_ransac, command=self._update_states)
        ttk.Label(ransac, text="Model").grid(row=1, column=0, sticky="w")
        model = ttk.Combobox(ransac, values=[MODEL_LABELS[m] for m in RANSAC_MODELS], state="readonly", width=32)
        model.current(list(RANSAC_MODELS).index(init.ransac_model))
        model.grid(row=1, column=1, sticky="w", pady=1)
        self._widgets["ransac_model"] = model
        self._entry(ransac, 2, "residual_threshold", "Residual threshold (px)", init.residual_threshold)
        self._entry(ransac, 3, "max_trials", "Max trials", init.max_trials)

        row = 3
        if self._output_modes:
            output = ttk.LabelFrame(master, text="Output", padding=6)
            output.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            self._output = tk.StringVar(master, value=next(iter(self._output_modes)))
            for index, (key, text) in enumerate(self._output_modes.items()):
                ttk.Radiobutton(output, text=text, value=key, variable=self._output).grid(row=index, column=0, sticky="w")
            row += 1
        if self._ask_recursive:
            self._recursive = tk.BooleanVar(master, value=False)
            ttk.Checkbutton(master, text="Include subfolders", variable=self._recursive).grid(row=row, column=0, sticky="w")

        self._update_states()
        return self._widgets["n_keypoints"]

    def _entry(self, master: tk.Misc, row: int, name: str, text: str, value: object) -> None:
        ttk.Label(master, text=text).grid(row=row, column=0, sticky="w", padx=(0, 8))
        variable = tk.StringVar(master, value=str(value))
        entry = ttk.Entry(master, textvariable=variable, width=10)
        entry.grid(row=row, column=1, sticky="w", pady=1)
        self._vars[name] = variable
        self._widgets[name] = entry

    def _check(self, master: tk.Misc, row: int, name: str, text: str, value: bool, command=None) -> None:
        variable = tk.BooleanVar(master, value=value)
        ttk.Checkbutton(master, text=text, variable=variable, command=command).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        self._vars[name] = variable

    def _update_states(self) -> None:
        """Grey out the fields that do not apply to the current choices."""
        is_orb = self._method.get() == "orb"
        use_ransac = bool(self._vars["use_ransac"].get())
        enabled = {
            "n_keypoints": is_orb,
            "n_octaves": not is_orb,
            "n_scales": not is_orb,
            "sigma": not is_orb,
            "residual_threshold": use_ransac,
            "max_trials": use_ransac,
        }
        for name, on in enabled.items():
            self._widgets[name].configure(state="normal" if on else "disabled")
        self._widgets["ransac_model"].configure(state="readonly" if use_ransac else "disabled")

    # -- result ---------------------------------------------------------------
    def validate(self) -> bool:
        values: dict[str, object] = {name: variable.get() for name, variable in self._vars.items()}
        values["method"] = self._method.get()
        values["ransac_model"] = list(RANSAC_MODELS)[self._widgets["ransac_model"].current()]
        options: dict[str, object] = {}
        if self._output_modes:
            options["output"] = self._output.get()
        if self._ask_recursive:
            options["recursive"] = bool(self._recursive.get())
        try:
            settings = parse_match_settings(values)
            if options.get("output") in WARP_OUTPUTS and not settings.use_ransac:
                raise ValueError("Transforming an image needs RANSAC: enable it or choose the matches view.")
        except ValueError as error:
            messagebox.showwarning("Invalid parameters", str(error), parent=self)
            return False
        self._parsed = (settings, options)
        return True

    def apply(self) -> None:
        self.result = self._parsed


def ask_match_settings(
    parent: tk.Misc,
    title: str,
    initial: MatchSettings,
    *,
    output_modes: Mapping[str, str] | None = None,
    ask_recursive: bool = False,
) -> tuple[MatchSettings, dict[str, object]] | None:
    """Show the matching form; returns ``(settings, options)`` or None if cancelled."""
    return MatchSettingsDialog(parent, title, initial, output_modes, ask_recursive).result
