"""Set 10 extra: CIFAR data, image patches and shifted patch tokenization (NumPy only).

The extra notebook trains a Vision Transformer on **CIFAR-100** (50,000 training and
10,000 test images of 32 × 32 px, 100 classes). CIFAR-10 (already in ``data/`` for Set 8)
is offered as well, because it needs no download.

Both datasets are Python pickles. ``pickle`` can run code while loading, so they are
read with a restricted unpickler that only accepts the NumPy functions needed to rebuild
arrays: a tampered file raises an error instead of executing anything.

Patches (ViT paper, "An Image is Worth 16x16 Words"): the image is resized to 72 × 72
and cut into non-overlapping 6 × 6 patches, giving 12 × 12 = 144 patches of
6 · 6 · 3 = 108 numbers each. Every patch becomes one token.

Shifted Patch Tokenization (SPT, Lee et al. 2021, "Vision Transformer for Small-Size
Datasets"): the image is also shifted by half a patch in the four diagonal directions,
the five images are stacked on the channel axis (15 channels), and the patches are cut
from that stack. Each token then also sees the pixels around its patch, which widens
the receptive field of the tokens (a ViT's weak point on small datasets).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import io
import pickle
from pathlib import Path
import tarfile
import threading
import urllib.request

import numpy as np

from forensics_app.paths import DATA_DIR

Report = Callable[[int, int, str], None]


@dataclass(frozen=True)
class CifarSource:
    key: str
    label: str
    folder: str
    url: str
    md5: str
    size_mb: int
    train_files: tuple[str, ...]
    test_file: str
    meta_file: str
    label_key: bytes
    names_key: bytes


DATASETS = {
    "cifar100": CifarSource(
        "cifar100", "CIFAR-100 (100 classes, notebook)", "cifar-100-python",
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85", 161,
        ("train",), "test", "meta", b"fine_labels", b"fine_label_names",
    ),
    "cifar10": CifarSource(
        "cifar10", "CIFAR-10 (10 classes, already downloaded for Set 8)", "cifar-10-batches-py",
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a", 163,
        tuple(f"data_batch_{i}" for i in range(1, 6)), "test_batch", "batches.meta", b"labels", b"label_names",
    ),
}
DATASET_KEYS = ("cifar100", "cifar10")


class _SafeUnpickler(pickle.Unpickler):
    """Only the NumPy pieces of a CIFAR pickle may be loaded; anything else is refused."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from a dataset file.")


def safe_load(path: Path) -> dict:
    with open(path, "rb") as handle:
        return _SafeUnpickler(io.BytesIO(handle.read()), encoding="bytes").load()


def dataset_folder(key: str) -> Path:
    return DATA_DIR / DATASETS[key].folder


def is_downloaded(key: str) -> bool:
    source = DATASETS[key]
    folder = dataset_folder(key)
    return all((folder / f).is_file() for f in (*source.train_files, source.test_file, source.meta_file))


def download(key: str, report: Report | None = None, cancelled: Callable[[], bool] | None = None) -> Path:
    """Download and unpack the official archive into data/ (checked against its MD5)."""
    source = DATASETS[key]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / Path(source.url).name
    digest = hashlib.md5()
    with urllib.request.urlopen(source.url, timeout=60) as response, open(archive, "wb") as out:
        total = int(response.headers.get("Content-Length", 0)) or source.size_mb * 2**20
        done = 0
        while chunk := response.read(1 << 20):
            if cancelled is not None and cancelled():
                out.close()
                archive.unlink(missing_ok=True)
                from .dialogs import TaskCancelled

                raise TaskCancelled()
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if report is not None:
                report(done, total, f"Downloading {source.label}: {done / 2**20:.0f} / {total / 2**20:.0f} MB…")
    if digest.hexdigest() != source.md5:
        archive.unlink(missing_ok=True)
        raise ValueError(f"The downloaded {archive.name} is corrupt (MD5 mismatch); try again.")
    with tarfile.open(archive) as tar:
        tar.extractall(DATA_DIR, filter="data")  # no absolute paths, links or files outside data/
    archive.unlink(missing_ok=True)
    return dataset_folder(key)


@dataclass
class CifarData:
    key: str
    class_names: list[str]
    x_train: np.ndarray  # (n, 32, 32, 3) uint8
    y_train: np.ndarray  # (n,) int64
    x_test: np.ndarray
    y_test: np.ndarray


def _decode(name) -> str:
    return name.decode() if isinstance(name, bytes) else str(name)


def _images(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw, dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


_CACHE: dict[str, CifarData] = {}
_LOCK = threading.Lock()


def load_cifar(key: str) -> CifarData:
    """The whole dataset in memory (cached)."""
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    source = DATASETS[key]
    folder = dataset_folder(key)
    if not is_downloaded(key):
        raise FileNotFoundError(f"{source.label} is not in {folder}.")
    xs, ys = [], []
    for name in source.train_files:
        batch = safe_load(folder / name)
        xs.append(_images(batch[b"data"]))
        ys.append(np.asarray(batch[source.label_key], dtype=np.int64))
    test = safe_load(folder / source.test_file)
    meta = safe_load(folder / source.meta_file)
    data = CifarData(key, [_decode(n).replace("_", " ") for n in meta[source.names_key]],
                     np.concatenate(xs), np.concatenate(ys), _images(test[b"data"]),
                     np.asarray(test[source.label_key], dtype=np.int64))
    with _LOCK:
        _CACHE[key] = data
    return data


def keras_validation_split(n: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """``model.fit(validation_split=f)`` keeps the LAST f of the samples (before shuffling) for validation."""
    split = int(n * (1 - fraction))
    return np.arange(split), np.arange(split, n)


# --------------------------------------------------------------------------- patches (for the figures)


def resize(image: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize of an (H, W, 3) uint8 image to size × size."""
    from PIL import Image

    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((size, size), Image.Resampling.BILINEAR))


def extract_patches(image: np.ndarray, patch: int) -> np.ndarray:
    """(H, W, C) -> (rows, cols, patch, patch, C): non-overlapping patches, row by row."""
    h, w, c = image.shape
    rows, cols = h // patch, w // patch
    cropped = image[: rows * patch, : cols * patch]
    return cropped.reshape(rows, patch, cols, patch, c).transpose(0, 2, 1, 3, 4)


SHIFTS = (("left-up", -1, -1), ("left-down", 1, -1), ("right-up", -1, 1), ("right-down", 1, 1))


def shift_image(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Move the content by (dy, dx) pixels and fill the uncovered border with zeros."""
    out = np.zeros_like(image)
    h, w = image.shape[:2]
    src_y = slice(max(0, -dy), h - max(0, dy))
    dst_y = slice(max(0, dy), h - max(0, -dy))
    src_x = slice(max(0, -dx), w - max(0, dx))
    dst_x = slice(max(0, dx), w - max(0, -dx))
    out[dst_y, dst_x] = image[src_y, src_x]
    return out


def shifted_images(image: np.ndarray, patch: int) -> list[tuple[str, np.ndarray]]:
    """The four diagonal shifts by half a patch used by SPT (the image itself comes first in the stack).

    The names follow the Keras example: 'left-up' crops the top-left corner away, so the
    content moves towards the top-left.
    """
    half = patch // 2
    return [(name, shift_image(image, sy * half, sx * half)) for name, sy, sx in SHIFTS]


def tokens_per_image(image_size: int, patch: int, spt: bool) -> tuple[int, int]:
    """(number of tokens, numbers per token before the projection)."""
    channels = 15 if spt else 3
    return (image_size // patch) ** 2, patch * patch * channels
