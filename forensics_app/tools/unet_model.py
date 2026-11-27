"""Set 11: train, save, load and run the U-Net on the tray dataset.

PyTorch is optional (``requirements-cnn.txt``): it is imported only inside the functions
that train or run a network, so listing models, reading passports and the metrics work
without it (and so do the tests).

Each trained model is a folder ``models/unet/<name>/`` (git-ignored) with:

* ``model.pt``: the weights (``state_dict``) and the settings needed to rebuild the
  network (classes, base channels, skip connections);
* ``passport.json``: everything else, like the Set 8 passports: task and classes, input
  size and preprocessing, architecture, training settings, the history of every epoch,
  and the validation and test metrics (Dice, IoU, per class, per image, confusion matrix,
  Dice for every threshold).

Training follows the notebook: Adam (lr 1e-3), batch 16, random horizontal flip of the
image **and** its mask, 15 % of XTrain for validation (seed 42, the same images as the
notebook), and the weights of the epoch with the best validation Dice are kept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.paths import MODELS_DIR
from .dialogs import TaskCancelled
from .segmentation_metrics import SplitEvaluation, evaluate_masks, threshold_masks, threshold_sweep
from .tray_dataset import (
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    LoadedSplit,
    Task,
    load_split,
    make_task,
    read_class_names,
    resize_image,
    split_indices,
)

UNET_DIR = MODELS_DIR / "unet"
PASSPORT_FORMAT = "forensics-unet/1"
SWEEP_THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
INSTALL_HINT = (
    "PyTorch is not installed, so the U-Net tools cannot train or run a network. Install it with:  "
    ".venv/bin/pip install -r requirements-cnn.txt"
)

Report = Callable[[int, int, str], None]


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def require_torch() -> None:
    if not torch_available():
        raise ValueError(INSTALL_HINT)


def default_device() -> str:
    """'cuda' with an NVIDIA GPU, 'mps' on Apple Silicon, otherwise 'cpu' (as in the notebook)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def device_label(device: str) -> str:
    if device == "cuda":
        try:
            import torch

            return f"GPU ({torch.cuda.get_device_name(0)})"
        except Exception:  # noqa: BLE001 - only a label
            return "GPU (CUDA)"
    if device == "mps":
        return f"Apple GPU (MPS, {platform.machine()})"
    return "CPU"


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class TrainSettings:
    root: Path
    task: str = "binary"
    size: tuple[int, int] = (128, 208)  # (height, width), divisible by 8
    epochs: int = 15
    batch: int = 16
    lr: float = 1e-3
    loss: str = "bce_dice"
    skip: bool = True
    base: int = 16
    augment: bool = True
    val_fraction: float = DEFAULT_VAL_FRACTION
    seed: int = DEFAULT_SEED
    split_by_tray: bool = False  # True: whole trays in validation (no near-duplicates of training images)
    device: str = ""  # "" = default_device()
    name: str = ""

    def resolved_device(self) -> str:
        return self.device or default_device()

    def default_name(self) -> str:
        name = default_model_name(self.task, self.loss, self.skip, self.size, self.epochs, self.base)
        return name + ("_traysplit" if self.split_by_tray else "")


def default_model_name(task: str, loss: str, skip: bool, size: tuple[int, int], epochs: int, base: int = 16) -> str:
    parts = ["unet", task, loss.replace("_", "-")]
    if not skip:
        parts.append("noskip")
    if base != 16:
        parts.append(f"base{base}")
    parts += [f"{size[0]}x{size[1]}", f"{epochs}e"]
    return "_".join(parts)


def safe_name(name: str) -> str:
    """A folder name: letters, digits, '-', '_' and '.' only."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or "unet_model"


def architecture_summary(base: int, skip: bool, out_ch: int) -> str:
    c = [base, base * 2, base * 4, base * 8]
    joined = "concatenated skip connections" if skip else "NO skip connections"
    return (f"3 encoder blocks ({c[0]}, {c[1]}, {c[2]} ch) + bottleneck ({c[3]} ch) + 3 decoder blocks, "
            f"{joined}; block = 2 × (3×3 conv, BatchNorm, ReLU); 2×2 max-pool / 2×2 transposed conv; "
            f"1×1 conv → {out_ch} output{'s' if out_ch > 1 else ''}")


# --------------------------------------------------------------------------- passport


@dataclass
class UnetPassport:
    folder: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return self.data.get("name", self.folder.name)

    @property
    def weights(self) -> Path:
        return self.folder / "model.pt"

    @property
    def class_names(self) -> list[str]:
        return list(self.data["task"]["classes"])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def binary(self) -> bool:
        return self.num_classes == 2

    @property
    def task(self) -> Task:
        task = self.data["task"]
        return Task(task["key"], task["label"], tuple(task["classes"]), tuple(task["lookup"]))

    @property
    def size(self) -> tuple[int, int]:
        size = self.data["input"]
        return int(size["height"]), int(size["width"])

    @property
    def threshold(self) -> float:
        value = self.data.get("evaluation", {}).get("threshold")  # None for the multi-class tasks (argmax)
        return 0.5 if value is None else float(value)

    @property
    def history(self) -> dict[str, list[float]]:
        return self.data.get("history", {})

    @property
    def best_epoch(self) -> int | None:
        return self.data.get("training", {}).get("best_epoch")

    def split(self, name: str) -> dict[str, Any]:
        return self.data.get("evaluation", {}).get(name, {})

    @property
    def label(self) -> str:
        test = self.split("test").get("dice_mean")
        val = self.split("val").get("dice_mean")
        score = f"val Dice {val:.3f}" if val is not None else ""
        if test is not None:
            score += f", test {test:.3f}"
        return f"{self.name} — {self.data['task']['label']}" + (f" ({score})" if score else "")


def read_passport(folder: Path) -> UnetPassport:
    path = Path(folder) / "passport.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != PASSPORT_FORMAT:
        raise ValueError(f"{path} is not a U-Net passport (format {data.get('format')!r}).")
    for key in ("task", "input"):
        if key not in data:
            raise ValueError(f"{path} has no '{key}' section.")
    return UnetPassport(Path(folder), data)


def list_models(directory: Path = UNET_DIR) -> list[UnetPassport]:
    """Trained models (folders with passport.json and model.pt), newest first."""
    if not directory.is_dir():
        return []
    found = []
    for folder in directory.iterdir():
        if (folder / "passport.json").is_file() and (folder / "model.pt").is_file():
            try:
                found.append(read_passport(folder))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    return sorted(found, key=lambda p: p.data.get("created", ""), reverse=True)


NO_MODELS = "There is no trained U-Net yet. Train one with 'U-Net: Train Segmentation Model' first."


def write_passport(folder: Path, data: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "passport.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- network


def build_network(out_ch: int, base: int = 16, skip: bool = True):
    require_torch()
    from .unet_net import UNet

    return UNet(out_ch=out_ch, base=base, skip=skip)


def output_channels(num_classes: int) -> int:
    """One logit (sigmoid) for the binary task, as the notebook; one per class (softmax) otherwise."""
    return 1 if num_classes == 2 else num_classes


_NET_CACHE: dict[tuple[str, float, str], Any] = {}
_NET_LOCK = threading.Lock()


def load_network(passport: UnetPassport, device: str | None = None):
    """The trained network on ``device``, in eval mode (cached until the file changes)."""
    require_torch()
    import torch

    device = device or default_device()
    key = (str(passport.weights), passport.weights.stat().st_mtime, device)
    with _NET_LOCK:
        if key not in _NET_CACHE:
            saved = torch.load(passport.weights, map_location="cpu", weights_only=True)
            config = saved["config"]
            net = build_network(int(config["out_channels"]), int(config["base"]), bool(config["skip"]))
            net.load_state_dict(saved["state_dict"])
            _NET_CACHE.clear()
            _NET_CACHE[key] = net.to(device).eval()
        return _NET_CACHE[key]


def _to_tensor(images: np.ndarray, device: str):
    import torch

    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div(255.0).to(device)


@dataclass
class Prediction:
    """Network output for a batch of images at the network size."""

    masks: np.ndarray  # (n, H, W) uint8 task classes
    confidence: np.ndarray  # (n, H, W) float32: P(food) for the binary task, max class probability otherwise
    probabilities: np.ndarray | None = None  # binary task only: P(food), the same as confidence


def predict_arrays(net, images: np.ndarray, num_classes: int, threshold: float = 0.5,
                   device: str | None = None, batch: int = 16) -> Prediction:
    """Segment (n, H, W, 3) uint8 images that already have the network size."""
    import torch

    device = device or next(net.parameters()).device.type
    masks = np.zeros(images.shape[:3], dtype=np.uint8)
    confidence = np.zeros(images.shape[:3], dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch):
            logits = net(_to_tensor(images[start:start + batch], device))
            if num_classes == 2:
                probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
                confidence[start:start + batch] = probs
                masks[start:start + batch] = threshold_masks(probs, threshold)
            else:
                probs = torch.softmax(logits, dim=1)
                best, index = probs.max(dim=1)
                confidence[start:start + batch] = best.cpu().numpy()
                masks[start:start + batch] = index.cpu().numpy().astype(np.uint8)
    return Prediction(masks, confidence, confidence if num_classes == 2 else None)


def segment_image(passport: UnetPassport, image: Image.Image, threshold: float | None = None,
                  device: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment any image: resized to the network size, then the result is scaled back.

    Returns ``(mask, confidence, network_input)``; ``mask`` and ``confidence`` have the
    size of ``image``. The probabilities are interpolated bilinearly before the
    threshold / argmax, which gives smoother borders than enlarging the mask.
    """
    import torch
    from torch.nn import functional as F

    threshold = passport.threshold if threshold is None else threshold
    net = load_network(passport, device)
    device = next(net.parameters()).device.type
    small = resize_image(image, passport.size)
    with torch.no_grad():
        logits = net(_to_tensor(small[None], device))
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        if passport.binary:
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            return threshold_masks(probs, threshold), probs.astype(np.float32), small
        probs = torch.softmax(logits, dim=1)[0]
        best, index = probs.max(dim=0)
        return index.cpu().numpy().astype(np.uint8), best.cpu().numpy().astype(np.float32), small


def evaluate_split(net, split: LoadedSplit, task: Task, threshold: float = 0.5, sweep: bool = False,
                   device: str | None = None) -> tuple[SplitEvaluation, Prediction]:
    """Metrics of the network on a loaded split (masks at the network size)."""
    gts = task.map(split.masks)
    prediction = predict_arrays(net, split.images, task.num_classes, threshold, device)
    evaluation = evaluate_masks(prediction.masks, gts, task.class_names, split.names,
                                threshold if task.binary else None)
    if sweep and task.binary and prediction.probabilities is not None:
        evaluation.sweep_thresholds = SWEEP_THRESHOLDS
        evaluation.sweep_dice = threshold_sweep(prediction.probabilities, gts, SWEEP_THRESHOLDS)
    return evaluation, prediction


# --------------------------------------------------------------------------- training


@dataclass
class TrainOutcome:
    passport: UnetPassport
    history: dict[str, list[float]]
    best_epoch: int
    epochs_done: int
    stopped_early: bool
    seconds: float
    val: SplitEvaluation
    test: SplitEvaluation | None
    parameters: int
    extra: dict[str, Any] = field(default_factory=dict)


def _framework_versions() -> dict[str, str]:
    import torch

    return {"torch": torch.__version__, "python": platform.python_version()}


def train(settings: TrainSettings, report: Report, cancelled: Callable[[], bool],
          target_dir: Path = UNET_DIR) -> TrainOutcome:
    """Train a U-Net, keep the best epoch, evaluate it and save it with its passport.

    Cancel stops after the current batch; if at least one epoch finished, the best
    model so far is still evaluated and saved (like YOLO in Set 9).
    """
    require_torch()
    import torch

    from .unet_net import count_parameters, segmentation_loss

    device = settings.resolved_device()
    source_names = read_class_names(settings.root)
    task = make_task(settings.task, source_names)
    started = time.perf_counter()

    data = load_split(settings.root, "train", settings.size, report, cancelled)
    try:
        test = load_split(settings.root, "test", settings.size)
    except FileNotFoundError:
        test = None
    train_idx, val_idx = split_indices(data.names, settings.val_fraction, settings.seed, settings.split_by_tray)
    train_split, val_split = data.subset(train_idx), data.subset(val_idx)
    train_x = train_split.images
    train_y = task.map(train_split.masks)
    val_y = task.map(val_split.masks)
    if len(train_split) == 0 or len(val_split) == 0:
        raise ValueError("Not enough images for a training and a validation set.")

    torch.manual_seed(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    out_ch = output_channels(task.num_classes)
    net = build_network(out_ch, settings.base, settings.skip).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [], "seconds": []}
    best_dice, best_state, best_epoch = -1.0, None, 0
    batches = (len(train_x) + settings.batch - 1) // settings.batch
    total_steps = settings.epochs * batches
    stopped = False

    # The whole (uint8) dataset is copied to the device once; batches are sliced there
    train_x_dev = torch.from_numpy(np.ascontiguousarray(train_x)).to(device)
    train_y_dev = torch.from_numpy(np.ascontiguousarray(train_y)).to(device)
    val_x_dev = torch.from_numpy(np.ascontiguousarray(val_split.images)).to(device)
    val_y_dev = torch.from_numpy(np.ascontiguousarray(val_y)).to(device)

    def as_input(images):  # (B, H, W, 3) uint8 -> (B, 3, H, W) float in [0, 1]
        return images.permute(0, 3, 1, 2).float().div_(255.0)

    def as_target(masks):  # (B, 1, H, W) float for one sigmoid output, (B, H, W) long for softmax
        return masks.float().unsqueeze(1) if out_ch == 1 else masks.long()

    for epoch in range(1, settings.epochs + 1):
        net.train()
        epoch_start, running = time.perf_counter(), 0.0
        order = torch.randperm(len(train_x), generator=generator)
        for b in range(batches):
            if cancelled():
                stopped = True
                break
            idx = order[b * settings.batch:(b + 1) * settings.batch].to(device)
            x, y = as_input(train_x_dev[idx]), as_target(train_y_dev[idx])
            if settings.augment:
                # the same random horizontal flip for each image and its mask
                flip = (torch.rand(len(idx), generator=generator) < 0.5).to(device)
                x = torch.where(flip[:, None, None, None], x.flip(-1), x)
                y = torch.where(flip.view(-1, *([1] * (y.dim() - 1))), y.flip(-1), y)
            optimizer.zero_grad()
            loss = segmentation_loss(net(x), y, settings.loss)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
            step = (epoch - 1) * batches + b + 1
            if b % 5 == 0 or b == batches - 1:
                best_text = f", best val Dice {best_dice:.3f} (epoch {best_epoch})" if best_epoch else ""
                report(step, total_steps,
                       f"Epoch {epoch}/{settings.epochs}, batch {b + 1}/{batches}: loss {loss.item():.3f}{best_text}")
        if stopped:
            break
        # validation in one pass: the loss with the same criterion, and the hard masks for the Dice
        net.eval()
        val_loss, val_masks = 0.0, []
        with torch.no_grad():
            for start in range(0, len(val_split), settings.batch):
                logits = net(as_input(val_x_dev[start:start + settings.batch]))
                val_loss += segmentation_loss(logits, as_target(val_y_dev[start:start + settings.batch]),
                                              settings.loss).item() * len(logits)
                masks = (logits[:, 0] > 0) if out_ch == 1 else logits.argmax(dim=1)  # logit > 0 ⇔ P > 0.5
                val_masks.append(masks.to(torch.uint8).cpu().numpy())
        val_eval = evaluate_masks(np.concatenate(val_masks), val_y, task.class_names, val_split.names)
        history["train_loss"].append(running / len(train_x))
        history["val_loss"].append(val_loss / len(val_split))
        history["val_dice"].append(val_eval.mean_dice)
        history["val_iou"].append(val_eval.mean_iou)
        history["seconds"].append(time.perf_counter() - epoch_start)
        if val_eval.mean_dice > best_dice:
            best_dice, best_epoch = val_eval.mean_dice, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        report(epoch * batches, total_steps,
               f"Epoch {epoch}/{settings.epochs}: loss {history['train_loss'][-1]:.3f}, "
               f"val Dice {val_eval.mean_dice:.3f} (best {best_dice:.3f}, epoch {best_epoch})")

    del train_x_dev, train_y_dev, val_x_dev, val_y_dev
    if best_state is None:
        raise TaskCancelled()
    net.load_state_dict(best_state)
    net.eval()

    report(total_steps, total_steps, "Evaluating the best epoch on validation and test…")
    val_eval, val_prediction = evaluate_split(net, val_split, task, 0.5, sweep=True, device=device)
    test_eval = evaluate_split(net, test, task, 0.5, device=device)[0] if test is not None else None
    seconds = time.perf_counter() - started

    name = safe_name(settings.name or settings.default_name())
    folder = Path(target_dir) / name
    temporary = Path(target_dir) / f".{name}.saving"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    config = {"out_channels": out_ch, "base": settings.base, "skip": settings.skip,
              "num_classes": task.num_classes}
    torch.save({"state_dict": best_state, "config": config}, temporary / "model.pt")
    parameters = count_parameters(net)
    data_out = build_passport(
        name, settings, task, parameters, history, best_epoch, len(history["train_loss"]),
        stopped, seconds, device, val_eval, test_eval, len(train_split), len(val_split),
        len(test) if test is not None else 0,
    )
    data_out["framework"] = _framework_versions()
    write_passport(temporary, data_out)
    if folder.exists():
        shutil.rmtree(folder)
    temporary.rename(folder)
    passport = read_passport(folder)
    return TrainOutcome(passport, history, best_epoch, len(history["train_loss"]), stopped, seconds,
                        val_eval, test_eval, parameters,
                        extra={"val_split": val_split, "val_masks": val_prediction.masks})


def build_passport(name: str, settings: TrainSettings, task: Task, parameters: int,
                   history: dict[str, list[float]], best_epoch: int, epochs_done: int, stopped: bool,
                   seconds: float, device: str, val: SplitEvaluation, test: SplitEvaluation | None,
                   n_train: int, n_val: int, n_test: int) -> dict[str, Any]:
    """The passport.json content (no torch: also used by the tests)."""
    height, width = settings.size
    loss_label = {"bce_dice": "BCE + Dice", "bce": "BCE", "dice": "Dice"}[settings.loss]
    if not task.binary:
        loss_label = loss_label.replace("BCE", "cross-entropy")
    evaluation: dict[str, Any] = {"threshold": 0.5 if task.binary else None, "val": val.to_dict()}
    if test is not None:
        evaluation["test"] = test.to_dict()
    return {
        "format": PASSPORT_FORMAT,
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": {"key": task.key, "label": task.label, "classes": list(task.class_names),
                 "lookup": list(task.lookup)},
        "input": {
            "height": height, "width": width,
            "preprocessing": f"RGB, resize to {width} × {height} px (bilinear; masks nearest), pixel / 255",
        },
        "architecture": {
            "summary": architecture_summary(settings.base, settings.skip, output_channels(task.num_classes)),
            "base_channels": settings.base, "skip_connections": settings.skip, "parameters": parameters,
            "output": "1 logit per pixel → sigmoid" if task.binary else f"{task.num_classes} logits per pixel → softmax",
        },
        "dataset": {"name": "Tray Food Segmentation (Kaggle thezaza102)", "root": str(settings.root),
                    "train_images": n_train, "val_images": n_val, "test_images": n_test,
                    "val_fraction": settings.val_fraction, "seed": settings.seed,
                    "split": "by tray (whole trays held out)" if settings.split_by_tray else "random (notebook)",
                    "split_by_tray": settings.split_by_tray},
        "training": {
            "optimizer": f"Adam (lr {settings.lr:g})", "loss": loss_label, "loss_key": settings.loss,
            "epochs": settings.epochs, "epochs_done": epochs_done, "stopped_early": stopped,
            "batch_size": settings.batch, "augmentation": "random horizontal flip" if settings.augment else "none",
            "best_epoch": best_epoch, "selection": "best validation Dice", "device": device,
            "seconds": round(seconds, 1),
        },
        "history": {k: [round(float(v), 5) for v in values] for k, values in history.items()},
        "evaluation": evaluation,
        "settings": {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v)
                     for k, v in asdict(settings).items()},
    }


def evaluate_passport(passport: UnetPassport, split_name: str, root: Path | None = None,
                      threshold: float | None = None, report: Report | None = None,
                      cancelled: Callable[[], bool] | None = None, device: str | None = None,
                      ) -> tuple[SplitEvaluation, LoadedSplit, Prediction]:
    """Re-run a saved model on its validation split (same seed) or on the test split."""
    root = Path(root or passport.data["dataset"]["root"])
    task = passport.task
    threshold = passport.threshold if threshold is None else threshold
    if split_name == "test":
        split = load_split(root, "test", passport.size, report, cancelled)
    else:
        data = load_split(root, "train", passport.size, report, cancelled)
        dataset = passport.data.get("dataset", {})
        _, val_idx = split_indices(data.names, dataset.get("val_fraction", DEFAULT_VAL_FRACTION),
                                   dataset.get("seed", DEFAULT_SEED), dataset.get("split_by_tray", False))
        split = data.subset(val_idx)
    if report is not None:
        report(0, 1, f"Segmenting {len(split)} {split_name} images…")
    net = load_network(passport, device)
    evaluation, prediction = evaluate_split(net, split, task, threshold, sweep=True)
    return evaluation, split, prediction
