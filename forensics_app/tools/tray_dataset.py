"""Set 11: the Tray Food Segmentation dataset (NumPy/PIL only, no PyTorch).

The dataset (Kaggle ``thezaza102/tray-food-segmentation``) has four folders:
``XTrain``/``yTrain`` (1,241 images and masks) and ``XTest``/``yTest`` (8). The images
are 416 × 256 JPEGs of hospital meal trays; each mask is a PNG with, for every pixel, the
index of its class (0 = background, 1 = tray, 2 = cutlery, … 42 = water). Some masks are
saved with three identical channels, so only the first one is read.

The notebook merges the 43 classes into a binary task (food vs not food). The tools
offer three tasks (``TASKS``), all built from the same masks with a lookup table:

* ``binary``: 0 = not food (classes 0-4), 1 = food (5-42), as in the notebook;
* ``three``: background / tray / food (Exercise 6). Cutlery, the menu form and the
  straw (2-4) are neither tray nor food, so they go with the background;
* ``all``: the 43 original classes (the "Segmentation Mask" figure of the slides).

Images are resized with bilinear interpolation and masks with **nearest** interpolation,
which never invents a class index that is not in the original mask.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import threading

import numpy as np
from PIL import Image

from forensics_app.paths import DATA_DIR, PROJECT_ROOT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_FOLDERS = {"train": ("XTrain", "yTrain"), "test": ("XTest", "yTest")}
FIRST_FOOD = 5  # index of 'meatball', the first food class
DEFAULT_SEED = 42  # the notebook's seed: the same validation images
DEFAULT_VAL_FRACTION = 0.15

# The class list of classes.csv, used when the file is missing
DEFAULT_CLASS_NAMES = (
    "background", "tray", "cutlery", "form", "straw", "meatball", "beef", "roast lamb",
    "beef tomato casserole", "ham", "bean", "cucumber", "leaf", "tomato", "boiled rice",
    "beef mexican meatballs", "spinach and pumpkin risotto", "baked fish", "gravy", "zucchini",
    "carrot", "broccoli", "pumpkin", "celery", "sandwich", "side salad", "tartare sauce",
    "jacket potato", "creamed potato", "bread", "margarine", "soup", "apple", "canned fruit",
    "milk", "vanilla yogurt", "jelly", "custard", "lemon sponge", "juice", "apple juice",
    "orange juice", "water",
)

# Image sizes offered for training: (height, width), divisible by 8 (three poolings)
IMAGE_SIZES = {
    "64 × 104 (fastest)": (64, 104),
    "128 × 208 (notebook)": (128, 208),
    "256 × 416 (original size, ~4× slower)": (256, 416),
}

Report = Callable[[int, int, str], None]


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    class_names: tuple[str, ...]
    # source class index -> task class index (length = number of source classes)
    lookup: tuple[int, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def binary(self) -> bool:
        return self.num_classes == 2

    def map(self, mask: np.ndarray) -> np.ndarray:
        """Source class indices -> task class indices (unknown indices become 0)."""
        table = np.zeros(256, dtype=np.uint8)
        table[: len(self.lookup)] = self.lookup
        return table[np.asarray(mask, dtype=np.uint8)]


def make_task(key: str, source_names: Sequence[str] = DEFAULT_CLASS_NAMES) -> Task:
    n = len(source_names)
    if key == "binary":
        lookup = tuple(int(c >= FIRST_FOOD) for c in range(n))
        return Task(key, "Food vs not food (binary, notebook)", ("not food", "food"), lookup)
    if key == "three":
        lookup = tuple(2 if c >= FIRST_FOOD else 1 if c == 1 else 0 for c in range(n))
        return Task(key, "Background / tray / food (3 classes, Exercise 6)", ("background", "tray", "food"), lookup)
    if key == "all":
        return Task(key, f"All {n} classes (as in the slides)", tuple(source_names), tuple(range(n)))
    raise ValueError(f"Unknown segmentation task {key!r}.")


TASK_KEYS = ("binary", "three", "all")


# --------------------------------------------------------------------------- files


def find_dataset_root() -> Path | None:
    """The TrayDataset folder: data/TrayDataset, or next to the repository (lab materials)."""
    candidates = [DATA_DIR / "TrayDataset", PROJECT_ROOT / "TrayDataset"]
    for parent in (PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent):
        candidates += sorted(parent.glob("*/TrayDataset")) + sorted(parent.glob("*/*/TrayDataset"))
    for candidate in candidates:
        if is_dataset_root(candidate):
            return candidate
    return None


def is_dataset_root(folder: Path | None) -> bool:
    return folder is not None and all((Path(folder) / name).is_dir() for name in ("XTrain", "yTrain"))


def read_class_names(root: Path) -> tuple[str, ...]:
    """Display names from classes.csv (``_id,_class,_name``; the file starts with a BOM)."""
    path = Path(root) / "classes.csv"
    if not path.is_file():
        return DEFAULT_CLASS_NAMES
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    names = {int(row["_id"]): (row.get("_name") or row.get("_class") or "").strip() for row in rows}
    if not names:
        return DEFAULT_CLASS_NAMES
    return tuple(names.get(i, f"class {i}") for i in range(max(names) + 1))


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def pair_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Each image with the mask of the same name (``1001a01.jpg`` ↔ ``1001a01.png``)."""
    masks = {p.stem: p for p in list_images(mask_dir)}
    return [(p, masks[p.stem]) for p in list_images(image_dir) if p.stem in masks]


def split_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir, mask_dir = (Path(root) / name for name in SPLIT_FOLDERS[split])
    if not image_dir.is_dir() or not mask_dir.is_dir():
        return []
    return pair_files(image_dir, mask_dir)


def read_mask(path: Path) -> np.ndarray:
    """Class index per pixel (uint8, H × W)."""
    with Image.open(path) as handle:
        mask = np.array(handle)
    if mask.ndim == 3:  # some masks are saved with 3 identical channels
        mask = mask[..., 0]
    return mask.astype(np.uint8)


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """RGB image -> (H, W, 3) uint8 at ``size`` = (height, width), bilinear."""
    height, width = size
    return np.array(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Class mask at ``size`` with NEAREST interpolation: no new class values are invented."""
    height, width = size
    return np.array(Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize((width, height), Image.Resampling.NEAREST))


def find_mask_for(image_path: Path | None, root: Path | None = None) -> Path | None:
    """The ground-truth mask of a dataset image: same name in the y folder next to it, or in ``root``."""
    if image_path is None:
        return None
    image_path = Path(image_path)
    parent = image_path.parent
    folders = []
    if parent.name.startswith("X"):
        folders.append(parent.parent / ("y" + parent.name[1:]))
    if root is not None:
        folders += [Path(root) / m for _, m in SPLIT_FOLDERS.values()]
    for folder in folders:
        for ext in (".png", ".jpg", ".bmp"):
            candidate = folder / f"{image_path.stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


# --------------------------------------------------------------------------- arrays


@dataclass
class LoadedSplit:
    """Images and source masks of a split at one size, kept in memory."""

    names: list[str]
    images: np.ndarray  # (n, H, W, 3) uint8
    masks: np.ndarray  # (n, H, W) uint8, source class indices (0-42)

    def __len__(self) -> int:
        return len(self.names)

    def subset(self, index: np.ndarray) -> "LoadedSplit":
        return LoadedSplit([self.names[i] for i in index], self.images[index], self.masks[index])


_CACHE: dict[tuple[str, str, tuple[int, int]], LoadedSplit] = {}
_CACHE_LOCK = threading.Lock()


def load_split(
    root: Path,
    split: str,
    size: tuple[int, int],
    report: Report | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> LoadedSplit:
    """All images and masks of ``split`` resized to ``size``; cached, so later tools load instantly."""
    key = (str(Path(root).resolve()), split, tuple(size))
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    pairs = split_pairs(root, split)
    if not pairs:
        raise FileNotFoundError(
            f"No {split} images with masks in {root} (expected {'/'.join(SPLIT_FOLDERS[split])})."
        )
    height, width = size
    images = np.zeros((len(pairs), height, width, 3), dtype=np.uint8)
    masks = np.zeros((len(pairs), height, width), dtype=np.uint8)
    for i, (image_path, mask_path) in enumerate(pairs):
        if cancelled is not None and cancelled():
            from .dialogs import TaskCancelled

            raise TaskCancelled()
        if report is not None and i % 50 == 0:
            report(i, len(pairs), f"Loading {split} images {i}/{len(pairs)}…")
        images[i] = resize_image(read_image(image_path), size)
        masks[i] = resize_mask(read_mask(mask_path), size)
    loaded = LoadedSplit([p.name for p, _ in pairs], images, masks)
    with _CACHE_LOCK:
        # keep only the splits of the last size used (the full-size train split is ~400 MB)
        for other in [k for k in _CACHE if k[2] != tuple(size)]:
            del _CACHE[other]
        _CACHE[key] = loaded
    return loaded


def train_val_indices(n: int, val_fraction: float = DEFAULT_VAL_FRACTION, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """The notebook's split: a permutation with ``np.random.seed(seed)``; the first 15 % is validation."""
    order = np.random.RandomState(seed).permutation(n)
    n_val = int(val_fraction * n)
    return order[n_val:], order[:n_val]


def tray_id(name: str) -> str:
    """'1001a07.jpg' -> '1001a': XTrain has 17 trays, each photographed 73 times (rotated, moved)."""
    match = re.match(r"(\d+[a-z])", Path(name).stem, re.IGNORECASE)
    return match.group(1).lower() if match else Path(name).stem


def split_indices(names: Sequence[str], val_fraction: float = DEFAULT_VAL_FRACTION, seed: int = DEFAULT_SEED,
                  by_tray: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Training and validation indices.

    ``by_tray=False`` is the notebook's random split: photos of the same tray end up in
    both sets, so the validation images are near-duplicates of training images and the
    validation Dice is optimistic. ``by_tray=True`` keeps whole trays out (group split):
    the validation trays are new to the model, like the test images.
    """
    if not by_tray:
        return train_val_indices(len(names), val_fraction, seed)
    groups = [tray_id(n) for n in names]
    trays = sorted(set(groups))
    order = np.random.RandomState(seed).permutation(len(trays))
    n_val = min(len(trays) - 1, max(1, round(val_fraction * len(trays))))
    chosen = {trays[i] for i in order[:n_val]}
    val = np.array([i for i, g in enumerate(groups) if g in chosen], dtype=np.int64)
    train = np.array([i for i, g in enumerate(groups) if g not in chosen], dtype=np.int64)
    return train, val


# --------------------------------------------------------------------------- statistics


@dataclass
class DatasetStats:
    split: str
    images: int
    size_counts: dict[tuple[int, int], int]
    pixels_per_class: np.ndarray  # source classes
    images_with_class: np.ndarray
    food_fraction: np.ndarray  # per image


def dataset_stats(root: Path, split: str, num_classes: int, report: Report | None = None) -> DatasetStats:
    """Class frequencies on the original masks (not resized)."""
    pairs = split_pairs(root, split)
    pixels = np.zeros(num_classes, dtype=np.int64)
    images_with = np.zeros(num_classes, dtype=np.int64)
    food = np.zeros(len(pairs))
    sizes: dict[tuple[int, int], int] = {}
    for i, (_, mask_path) in enumerate(pairs):
        if report is not None and i % 100 == 0:
            report(i, len(pairs), f"Counting classes in {split} ({i}/{len(pairs)})…")
        mask = read_mask(mask_path)
        counts = np.bincount(mask.ravel(), minlength=num_classes)[:num_classes]
        pixels += counts
        images_with += counts > 0
        food[i] = counts[FIRST_FOOD:].sum() / mask.size
        sizes[(mask.shape[1], mask.shape[0])] = sizes.get((mask.shape[1], mask.shape[0]), 0) + 1
    return DatasetStats(split, len(pairs), sizes, pixels, images_with, food)
