"""Set 10 tools: CLIP zero-shot classification and text → image search.

* ``ClipZeroShotTool`` (10.1): any number of classes, any text per class, one or several
  templates (prompt ensembling, or one chart per template), one or two models.
* ``ClipSearchTool``: the notebook's text → image search over a folder or the
  scikit-image samples, with the text × image similarity matrix.

The work is in ``clip_model`` (embeddings, zero-shot maths, search) and ``clip_render``
(figures); this module holds the dialogs and the text of the results panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import display_path
from . import clip_model as cm
from .base import ForensicsTool, ToolResult
from .clip_render import render_search, render_zero_shot
from .cnn_features import to_rgb
from .dialogs import OPEN_TYPES, TaskCancelled, run_with_progress
from .tray_dataset import DEFAULT_CLASS_NAMES

CATEGORY = "Vision-Language (CLIP)"


@dataclass(frozen=True)
class Preset:
    label: str
    labels: tuple[str, ...]
    templates: tuple[str, ...]
    image: str | None = None  # suggested scikit-image sample
    compare: bool = False


PRESETS = (
    Preset("Animals (notebook 1.1)", ("cat", "dog", "tiger", "rabbit", "fox", "owl"), ("a photo of a {}.",), "chelsea"),
    Preset("How many cats? (slides)", ("one cat", "two cats", "three cats", "no cats"), ("a photo of {}.",), "chelsea"),
    Preset("Where are the cats? (notebook 1.2)", ("on a sofa", "on the grass", "in a car", "on a bed"),
           ("a photo of cats {}.",), "chelsea"),
    Preset("Photo, painting, cartoon or sketch? (notebook 1.2)", ("photo", "painting", "cartoon", "pencil sketch"),
           ("a {} of cats.",), "chelsea"),
    Preset("Drinks (slides, coffee)", ("coffee", "tea", "soup", "hot chocolate"), ("a photo of a {}.",), "coffee"),
    Preset("Prompt engineering: 3 templates (notebook 1.3)", ("coffee", "tea", "soup", "hot chocolate"),
           ("{}", "a photo of a {}.", "a photo of a cup of {}."), "coffee", compare=True),
    Preset("Prompt ensembling: 5 templates averaged (notebook)", ("coffee", "tea", "soup", "hot chocolate"),
           cm.ENSEMBLE_TEMPLATES, "coffee"),
    Preset("Domain shift: healthy vs tumour tissue (notebook 1.5)", ("healthy tissue", "tumour tissue"),
           ("a microscopy image of {}.",), "immunohistochemistry"),
    Preset("Forced choice: cat or dog on the coffee (notebook 1.5)", ("cat", "dog"), ("a photo of a {}.",), "coffee"),
    Preset("Same, with an 'other' option", ("cat", "dog", "something else"), ("a photo of {}.",), "coffee"),
    Preset("Space (astronaut / rocket)", ("an astronaut", "a rocket launch", "a pilot", "a scientist in a lab"),
           ("a photo of {}.",), "astronaut"),
    Preset("Food on a hospital tray (TrayDataset classes)",
           tuple(DEFAULT_CLASS_NAMES[c] for c in (5, 6, 14, 19, 20, 21, 24, 27, 28, 31, 32, 36)),
           ("a photo of {} on a hospital meal tray.",), None),
)
CUSTOM_PRESET = "Custom (keep the text below)"


@dataclass
class ZeroShotSettings:
    models: list[str]
    source: str  # "working", "sample", "file"
    sample: str = "chelsea"
    file: Path | None = None
    labels: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    compare_templates: bool = False
    show_cosine: bool = True


def parse_zero_shot(labels_text: str, templates_text: str) -> tuple[list[str], list[str], str | None]:
    """Classes and templates from the dialog's text boxes, and a problem (or None)."""
    labels = cm.parse_lines(labels_text)
    templates = cm.parse_lines(templates_text)
    if len(labels) < 2:
        return labels, templates, "Write at least two classes, one per line."
    return labels, templates, cm.check_templates(templates)


class ClipZeroShotDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, current_name: str | None) -> None:
        self._current_name = current_name
        self.settings: ZeroShotSettings | None = None
        super().__init__(parent, "CLIP zero-shot classification (Set 10.1)")

    def body(self, master: tk.Frame) -> tk.Widget:
        model_names = list(cm.CLIP_MODELS)
        frame = ttk.LabelFrame(master, text="Model", padding=6)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._model = ttk.Combobox(frame, state="readonly", width=62,
                                   values=[self._model_label(n) for n in model_names])
        self._model.current(0)
        self._model.grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="Model:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self._compare_model = ttk.Combobox(frame, state="readonly", width=62,
                                           values=["(none)"] + [self._model_label(n) for n in model_names])
        self._compare_model.current(0)
        ttk.Label(frame, text="Compare with:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(4, 0))
        self._compare_model.grid(row=1, column=1, sticky="w", pady=(4, 0))

        frame = ttk.LabelFrame(master, text="Image", padding=6)
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self._source = tk.StringVar(master, value="working" if self._current_name else "sample")
        row = 0
        if self._current_name:
            ttk.Radiobutton(frame, text=f"Working image ({self._current_name})", value="working",
                            variable=self._source).grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1
        ttk.Radiobutton(frame, text="Sample image (scikit-image, offline):", value="sample",
                        variable=self._source).grid(row=row, column=0, sticky="w")
        self._sample = ttk.Combobox(frame, state="readonly", width=26, values=list(cm.SAMPLE_IMAGES))
        self._sample.current(cm.SAMPLE_IMAGES.index("chelsea"))
        self._sample.grid(row=row, column=1, sticky="w", padx=(6, 0))
        self._sample.bind("<<ComboboxSelected>>", lambda _e: self._source.set("sample"))
        row += 1
        ttk.Radiobutton(frame, text="Image file:", value="file", variable=self._source).grid(row=row, column=0, sticky="w")
        self._file = tk.StringVar(master, value="")
        file_row = ttk.Frame(frame)
        file_row.grid(row=row, column=1, sticky="w", padx=(6, 0))
        ttk.Entry(file_row, textvariable=self._file, width=30).pack(side="left")
        ttk.Button(file_row, text="Browse…", command=self._browse).pack(side="left", padx=(4, 0))

        frame = ttk.LabelFrame(master, text="Classes and prompts", padding=6)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(frame, text="Preset:").grid(row=0, column=0, sticky="w")
        self._preset = ttk.Combobox(frame, state="readonly", width=56,
                                    values=[CUSTOM_PRESET] + [p.label for p in PRESETS])
        self._preset.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self._preset.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())
        ttk.Label(frame, text="Classes, one per line\n(any number, any text):", justify="left").grid(
            row=1, column=0, sticky="nw", pady=(6, 0))
        self._labels = tk.Text(frame, width=48, height=7, wrap="none", undo=True)
        self._labels.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(frame, text="Templates, one per line\n({} = the class text):", justify="left").grid(
            row=2, column=0, sticky="nw", pady=(6, 0))
        self._templates = tk.Text(frame, width=48, height=4, wrap="none", undo=True)
        self._templates.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        self._compare_templates = tk.BooleanVar(master, value=False)
        ttk.Checkbutton(frame, text="Several templates: one chart per template (plus the average) instead of "
                        "only the average", variable=self._compare_templates).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._cosine = tk.BooleanVar(master, value=True)
        ttk.Checkbutton(frame, text="Also show the raw cosine similarities (before the softmax)",
                        variable=self._cosine).grid(row=4, column=0, columnspan=2, sticky="w")
        self._preset.current(1)
        self._apply_preset()
        if self._current_name:
            self._source.set("working")
        return self._labels

    @staticmethod
    def _model_label(name: str) -> str:
        return f"{cm.CLIP_MODELS[name]}" + ("" if cm.is_downloaded(name) else "  — downloaded on first use")

    def _browse(self) -> None:
        name = filedialog.askopenfilename(parent=self, title="Image to classify", filetypes=OPEN_TYPES)
        if name:
            self._file.set(name)
            self._source.set("file")

    def _apply_preset(self) -> None:
        index = self._preset.current()
        if index <= 0:
            return
        preset = PRESETS[index - 1]
        self._labels.delete("1.0", "end")
        self._labels.insert("1.0", "\n".join(preset.labels))
        self._templates.delete("1.0", "end")
        self._templates.insert("1.0", "\n".join(preset.templates))
        self._compare_templates.set(preset.compare)
        if preset.image:
            self._sample.current(cm.SAMPLE_IMAGES.index(preset.image))

    def validate(self) -> bool:
        labels, templates, problem = parse_zero_shot(self._labels.get("1.0", "end"), self._templates.get("1.0", "end"))
        if problem:
            messagebox.showwarning("Invalid classes or templates", problem, parent=self)
            return False
        source = self._source.get()
        file = Path(self._file.get().strip()) if self._file.get().strip() else None
        if source == "file" and (file is None or not file.is_file()):
            messagebox.showwarning("No image", "Choose an existing image file.", parent=self)
            return False
        model_names = list(cm.CLIP_MODELS)
        models = [model_names[self._model.current()]]
        if self._compare_model.current() > 0:
            other = model_names[self._compare_model.current() - 1]
            if other not in models:
                models.append(other)
        self.settings = ZeroShotSettings(models, source, self._sample.get(), file, labels, templates,
                                         self._compare_templates.get(), self._cosine.get())
        return True


def zero_shot_details(run: cm.ZeroShotRun, settings: ZeroShotSettings, image_name: str) -> dict[str, object]:
    details: dict[str, object] = {
        "Operation": "CLIP zero-shot classification",
        "Image": image_name,
        "Classes": f"{len(settings.labels)}: " + "; ".join(settings.labels),
        "Templates": " | ".join(settings.templates),
        "Mode": ("one chart per template + the average" if settings.compare_templates and len(settings.templates) > 1
                 else "prompt ensembling: average of the normalised text vectors" if len(settings.templates) > 1
                 else "one template"),
        "How": "p = softmax(exp(logit_scale) · cos(image, text)) over the classes; no training",
    }
    for result in run.results:
        model = result.model.split("/")[-1]
        tag = f"{model}, {result.title}"
        order = np.argsort(-result.probs)
        details[f"[{tag}] prediction"] = f"{result.labels[order[0]]} ({result.probs[order[0]]:.1%})"
        for i in order:
            details[f"[{tag}] {result.labels[i]}"] = (
                f"p = {result.probs[i]:.3f}, cos = {result.cosine[i]:.4f}, logit = {result.logits[i]:.2f}")
    first = run.results[-1]
    spread = float(first.cosine.max() - first.cosine.min())
    details["Note: softmax sums to 1"] = (
        f"the cosines only span {spread:.3f}, but ×{first.logit_scale:.0f} the softmax makes the winner look sure; "
        "a class list without the right answer still gets 100 % in total (add an 'other' class or read the cosines)")
    for model, counts in run.token_counts.items():
        details[f"Tokens per prompt ({model.split('/')[-1]})"] = (
            f"{min(counts)}-{max(counts)} incl. start/end (limit 77)")
    for model, info in run.info.items():
        for key, value in info.items():
            details[f"{model.split('/')[-1]}: {key}"] = value
    details["Time"] = f"{run.seconds:.2f} s (the first run also loads the model)"
    return details


def zero_shot_result(image: Image.Image, image_name: str, settings: ZeroShotSettings, encoders) -> ToolResult:
    run = cm.run_zero_shot(encoders, image, settings.labels, settings.templates, settings.compare_templates)
    best = run.results[-1]
    figure = render_zero_shot(image, run.results, settings.show_cosine, f"CLIP zero-shot: {image_name}")
    message = (f"CLIP ({best.model.split('/')[-1]}): {best.labels[best.best]} "
               f"({best.probs[best.best]:.1%}) among {len(best.labels)} classes.")
    return ToolResult(message=message, image=figure, details=zero_shot_details(run, settings, image_name),
                      preview_only=True)


def _loading_label(name: str) -> str:
    size = "1.7 GB" if "large" in name else "600 MB"
    return (f"Loading {name}…" if cm.is_downloaded(name)
            else f"Downloading {name} to models/clip (first time only, ~{size})…")


class ClipZeroShotTool(ForensicsTool):
    tool_id = "clip_zero_shot"
    title = "CLIP: Zero-Shot Classification"
    category = CATEGORY
    description = (
        "Set 10.1: classifies an image into any classes written as text (any number, any sentence per class) "
        "with CLIP, without training. Supports templates, prompt ensembling and comparing two CLIP models."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        cm.require_clip()
        current = None
        if document.current is not None:
            current = document.path.name if document.path else "unsaved image"
        dialog = ClipZeroShotDialog(parent, current)
        settings = dialog.settings
        if settings is None:
            return None
        if settings.source == "working":
            image, name = to_rgb(document.current), current or "working image"
        elif settings.source == "file":
            with Image.open(settings.file) as handle:
                image, name = handle.convert("RGB"), settings.file.name
        else:
            image, name = cm.sample_image(settings.sample), f"{settings.sample} (scikit-image)"

        def task(report, cancelled):
            encoders = []
            for k, model in enumerate(settings.models):
                report(k, len(settings.models) + 1, _loading_label(model))
                encoders.append(cm.load_encoder(model))
            report(len(settings.models), len(settings.models) + 1, "Encoding the image and the prompts…")
            return zero_shot_result(image, name, settings, encoders)

        try:
            return run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None


# --------------------------------------------------------------------------- search


DEFAULT_QUERIES = ("something you drink in the morning", "a person who has been to space",
                   "a microscope image of tissue", "a photo of a cat")


@dataclass
class SearchSettings:
    model: str
    folder: Path | None  # None = scikit-image samples
    queries: list[str]
    top_k: int = 3
    show_matrix: bool = True


class ClipSearchDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc) -> None:
        self.settings: SearchSettings | None = None
        super().__init__(parent, "CLIP text → image search")

    def body(self, master: tk.Frame) -> tk.Widget:
        names = list(cm.CLIP_MODELS)
        ttk.Label(master, text="Model:").grid(row=0, column=0, sticky="w")
        self._model = ttk.Combobox(master, state="readonly", width=60,
                                   values=[ClipZeroShotDialog._model_label(n) for n in names])
        self._model.current(0)
        self._model.grid(row=0, column=1, sticky="w", pady=2)
        frame = ttk.LabelFrame(master, text="Images to search", padding=6)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        self._source = tk.StringVar(master, value="samples")
        ttk.Radiobutton(frame, text=f"scikit-image samples ({len(cm.SAMPLE_IMAGES)} images, as in the notebook)",
                        value="samples", variable=self._source).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(frame, text="All images in a folder (and its sub-folders):", value="folder",
                        variable=self._source).grid(row=1, column=0, sticky="w")
        from .tray_dataset import find_dataset_root

        tray = find_dataset_root()
        self._folder = tk.StringVar(master, value=str(tray / "XTest") if tray else "")
        row = ttk.Frame(frame)
        row.grid(row=2, column=0, columnspan=2, sticky="w", padx=(20, 0))
        ttk.Entry(row, textvariable=self._folder, width=48).pack(side="left")
        ttk.Button(row, text="Browse…", command=self._browse).pack(side="left", padx=(4, 0))
        ttk.Label(master, text="Queries, one per line:").grid(row=2, column=0, sticky="nw")
        self._queries = tk.Text(master, width=50, height=5, wrap="none", undo=True)
        self._queries.insert("1.0", "\n".join(DEFAULT_QUERIES))
        self._queries.grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(master, text="Images per query:").grid(row=3, column=0, sticky="w")
        self._top_k = tk.StringVar(master, value="3")
        ttk.Entry(master, textvariable=self._top_k, width=6).grid(row=3, column=1, sticky="w", pady=2)
        self._matrix = tk.BooleanVar(master, value=True)
        ttk.Checkbutton(master, text="Show the text × image similarity matrix", variable=self._matrix).grid(
            row=4, column=0, columnspan=2, sticky="w")
        return self._queries

    def _browse(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Folder with images", mustexist=True)
        if folder:
            self._folder.set(folder)
            self._source.set("folder")

    def validate(self) -> bool:
        queries = cm.parse_lines(self._queries.get("1.0", "end"))
        if not queries:
            messagebox.showwarning("No query", "Write at least one query.", parent=self)
            return False
        try:
            top_k = int(self._top_k.get())
        except ValueError:
            top_k = 0
        if not 1 <= top_k <= 10:
            messagebox.showwarning("Invalid value", "Show between 1 and 10 images per query.", parent=self)
            return False
        folder = None
        if self._source.get() == "folder":
            folder = Path(self._folder.get().strip())
            if not folder.is_dir():
                messagebox.showwarning("No folder", "Choose an existing folder.", parent=self)
                return False
        self.settings = SearchSettings(list(cm.CLIP_MODELS)[self._model.current()], folder, queries, top_k,
                                       self._matrix.get())
        return True


def search_result(encoder, gallery: cm.Gallery, settings: SearchSettings) -> ToolResult:
    sims = cm.search(encoder, gallery, settings.queries)
    figure = render_search(gallery, settings.queries, sims, settings.top_k, settings.show_matrix,
                           encoder.name.split("/")[-1])
    details: dict[str, object] = {
        "Operation": "CLIP text → image search",
        "Model": encoder.name,
        "Gallery": display_path(settings.folder) if settings.folder else "scikit-image samples",
        "Images": len(gallery.names),
        "How": "each image is encoded once; a query is encoded and the images are ranked by cosine similarity",
    }
    for q, query in enumerate(settings.queries):
        order = np.argsort(-sims[q])[: settings.top_k]
        details[f'"{query}"'] = ", ".join(f"{gallery.names[i]} ({sims[q, i]:.3f})" for i in order)
    best = [gallery.names[int(np.argmax(s))] for s in sims]
    return ToolResult(message=f"CLIP search over {len(gallery.names)} images: " + "; ".join(
        f"'{q}' → {b}" for q, b in zip(settings.queries, best)), image=figure, details=details, preview_only=True)


class ClipSearchTool(ForensicsTool):
    tool_id = "clip_search"
    title = "CLIP: Text → Image Search"
    category = CATEGORY
    description = (
        "Set 10 (notebook 1.4): images and text share CLIP's embedding space, so a sentence can rank images. "
        "Searches the scikit-image samples or any folder and shows the similarity matrix."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        cm.require_clip()
        settings = ClipSearchDialog(parent).settings
        if settings is None:
            return None

        def task(report, cancelled):
            report(0, 1, _loading_label(settings.model))
            encoder = cm.load_encoder(settings.model)
            gallery = cm.build_gallery(encoder, settings.folder, report, cancelled)
            report(1, 1, "Ranking the images…")
            return search_result(encoder, gallery, settings)

        try:
            return run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
