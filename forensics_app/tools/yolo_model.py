"""Set 9: thin wrapper around Ultralytics YOLO (loading, prediction, training).

Ultralytics is an optional dependency (``requirements-yolo.txt``). It is imported only
inside the functions that need it, so the rest of the app and the tests work without it.

Paths are always explicit: the weights live in ``models/yolo`` and the training runs in
``models/yolo/runs`` (both git-ignored), never in the folder the app was started from.

About colours ("Why are people blue?"): Ultralytics follows OpenCV, so a NumPy array
given to ``model(...)`` is read as **BGR**, and ``results[0].plot()`` returns BGR too.
Showing that array with ``plt.imshow`` (which expects RGB) swaps red and blue: the
orange bus turns blue and so do the people. Here PIL images are passed instead
(Ultralytics converts them correctly) and the boxes are drawn on the original RGB image.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import csv
import importlib.util
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.paths import MODELS_DIR, PROJECT_ROOT
from .detection_metrics import Detections, batched_nms
from .dialogs import TaskCancelled

YOLO_DIR = MODELS_DIR / "yolo"
RUNS_DIR = YOLO_DIR / "runs"
BLOOD_CELL_WEIGHTS = YOLO_DIR / "blood_cell_yolo11n.pt"  # fine-tuned from the COCO weights
DATASETS_DIR = PROJECT_ROOT / "datasets"
BLOOD_CELL_DATASET = DATASETS_DIR / "blood_cell_yolo"

# Pretrained COCO models that can be chosen in the dialogs (downloaded on first use)
PRETRAINED = {
    "yolo11n.pt": "YOLO11n (course model, 2.6 M parameters)",
    "yolo11s.pt": "YOLO11s (bigger, more accurate, slower)",
    "yolo26n.pt": "YOLO26n (2026, NMS-free end-to-end head)",
}
DEFAULT_PRETRAINED = "yolo11n.pt"

INSTALL_HINT = (
    "Ultralytics is not installed, so YOLO is not available. Install the optional YOLO "
    "dependencies with:  .venv/bin/pip install -r requirements-yolo.txt"
)

Report = Callable[[int, int, str], None]


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def require_ultralytics() -> None:
    if not ultralytics_available():
        raise ValueError(INSTALL_HINT)


def default_device() -> str:
    """'0' (first GPU) when CUDA works, otherwise 'cpu'."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def device_label(device: str) -> str:
    if device == "cpu":
        return "CPU"
    try:
        import torch

        return f"GPU {device} ({torch.cuda.get_device_name(int(device))})"
    except Exception:  # noqa: BLE001 - only a label
        return f"GPU {device}"


def weights_path(name: str) -> Path:
    """Where a pretrained model is (or will be downloaded): ``models/yolo/<name>``."""
    path = Path(name)
    return path if path.is_absolute() else YOLO_DIR / path


def installed_weights_for(start_from: str) -> Path:
    """``models/yolo/blood_cell_<model>.pt``; ``_scratch`` when trained from a .yaml."""
    stem = Path(start_from).stem
    suffix = "_scratch" if Path(start_from).suffix == ".yaml" else ""
    return YOLO_DIR / f"blood_cell_{stem}{suffix}.pt"


def blood_cell_models() -> list[Path]:
    """The blood-cell models trained so far (fine-tuned first)."""
    if not YOLO_DIR.is_dir():
        return []
    found = sorted(YOLO_DIR.glob("blood_cell_*.pt"))
    return sorted(found, key=lambda p: (p != BLOOD_CELL_WEIGHTS, "scratch" in p.stem, p.name))


def load_model(weights: str | Path) -> Any:
    """Load a YOLO model. A bare pretrained name (``yolo11n.pt``) is downloaded to models/yolo."""
    require_ultralytics()
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.is_absolute():
        if weights.suffix == ".pt" and weights.name == str(weights):
            YOLO_DIR.mkdir(parents=True, exist_ok=True)
            weights = weights_path(weights.name)
        else:
            weights = PROJECT_ROOT / weights
    if weights.suffix == ".pt" and not weights.is_file() and weights.name not in PRETRAINED:
        raise FileNotFoundError(f"There is no model at {weights}.")
    return YOLO(str(weights))


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB copy of any PIL mode (YOLO needs 3 channels); always a new, fully loaded image."""
    if image.mode == "RGB":
        return image.copy()
    if image.mode in ("I", "F") or image.mode.startswith("I;16"):
        array = np.asarray(image, dtype=np.float64)
        low, high = float(array.min()), float(array.max())
        scaled = np.zeros_like(array) if high <= low else (array - low) / (high - low) * 255
        return Image.fromarray(scaled.astype(np.uint8)).convert("RGB")
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


@dataclass
class Prediction:
    """Detections of one image plus what is needed to explain them."""

    detections: Detections
    names: dict[int, str]
    seconds: float
    speed: dict[str, float] = field(default_factory=dict)  # ms: preprocess, inference, postprocess
    raw: Detections | None = None  # before our own NMS, when it was used
    nms: str = "ultralytics"

    def class_name(self, cls: int) -> str:
        return self.names.get(int(cls), str(cls))


def _result_to_detections(result: Any) -> Detections:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return Detections.empty(with_scores=True)
    return Detections(
        boxes.xyxy.cpu().numpy().astype(np.float64),
        boxes.cls.cpu().numpy().astype(np.int64),
        boxes.conf.cpu().numpy().astype(np.float64),
    )


def has_nms_free_head(model: Any) -> bool:
    """True for models with a one-to-one head (YOLO26) that can skip NMS entirely."""
    try:
        return getattr(model.model.model[-1], "one2one_cv2", None) is not None
    except (AttributeError, IndexError, TypeError):
        return False


def predict(
    model: Any,
    image: Image.Image,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
    own_nms: bool = False,
    class_agnostic: bool = False,
    max_det: int = 300,
    nms_free: bool = False,
) -> Prediction:
    """Run YOLO on a PIL image.

    With ``own_nms`` the Ultralytics NMS is practically switched off (IoU 0.99 only
    merges identical boxes) and the candidates go through ``batched_nms`` from
    ``detection_metrics`` instead, so the effect of the NMS threshold can be seen.

    With ``nms_free`` a YOLO26 model uses its one-to-one head: it was trained to give
    a single box per object, so there is no NMS at all (and no IoU threshold).
    """
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("The confidence and IoU thresholds must be between 0 and 1.")
    if nms_free and not has_nms_free_head(model):
        raise ValueError("This model has no NMS-free (one-to-one) head. Choose a YOLO26 model, e.g. yolo26n.pt.")
    # Ultralytics builds its predictor (and picks the head) on the first call only
    mode = "nms_free" if nms_free else "nms"
    if getattr(model, "_forensics_head_mode", "nms") != mode:
        model.predictor = None
    model._forensics_head_mode = mode
    extra = {"nms": False} if nms_free else {}
    device = device or default_device()
    start = time.perf_counter()
    results = model.predict(
        to_rgb(image),
        conf=conf,
        iou=0.99 if own_nms else iou,
        imgsz=imgsz,
        device=device,
        agnostic_nms=class_agnostic,
        max_det=3000 if own_nms else max_det,
        verbose=False,
        **extra,
    )
    seconds = time.perf_counter() - start
    result = results[0]
    detections = _result_to_detections(result)
    raw = None
    if own_nms and not nms_free:
        raw = detections
        keep = batched_nms(raw.boxes, raw.scores, raw.classes, iou, class_agnostic)[:max_det]
        detections = raw.subset(keep)
    return Prediction(
        detections=detections,
        names={int(k): str(v) for k, v in result.names.items()},
        seconds=seconds,
        speed=dict(getattr(result, "speed", {}) or {}),
        raw=raw,
        nms="none (NMS-free one-to-one head)" if nms_free else "own greedy NMS" if own_nms else "ultralytics",
    )


def predict_folder(
    model: Any,
    images: list[Path],
    conf: float = 0.001,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
    report: Report = lambda d, t, s: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> list[Detections]:
    """Predictions for many images (low confidence by default, as needed for the PR curve)."""
    device = device or default_device()
    out: list[Detections] = []
    for index, path in enumerate(images):
        if cancelled():
            raise TaskCancelled()
        with Image.open(path) as handle:
            image = to_rgb(handle)
            out.append(predict(model, image, conf=conf, iou=iou, imgsz=imgsz, device=device).detections)
        report(index + 1, len(images), f"Predicting {index + 1}/{len(images)}: {path.name}")
    return out


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class TrainSettings:
    data_yaml: Path
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    start_from: str = "yolo11n.pt"  # .pt = fine-tune pretrained weights, .yaml = from scratch
    patience: int = 20
    device: str | None = None
    name: str = "blood_cell"
    workers: int = 4
    seed: int = 0

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("Train for at least one epoch.")
        if self.batch < 1 and self.batch != -1:
            raise ValueError("The batch size must be positive (or -1 for automatic).")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("The image size must be a multiple of 32 (e.g. 640).")
        if not Path(self.data_yaml).is_file():
            raise FileNotFoundError(f"There is no dataset description at {self.data_yaml}. Convert the dataset first.")
        if Path(self.start_from).suffix not in (".pt", ".yaml"):
            raise ValueError("Start from a .pt (pretrained) or a .yaml (from scratch) model.")


@dataclass
class TrainOutcome:
    run_dir: Path
    best_weights: Path
    installed_weights: Path
    epochs_done: int
    stopped_early: bool
    seconds: float
    history: dict[str, list[float]]


def read_results_csv(path: str | Path) -> dict[str, list[float]]:
    """Columns of the ``results.csv`` written by Ultralytics (epoch, losses, metrics...)."""
    columns: dict[str, list[float]] = {}
    path = Path(path)
    if not path.is_file():
        return columns
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    columns.setdefault(key.strip(), []).append(float(value))
                except (TypeError, ValueError):
                    continue
    return columns


def train(
    settings: TrainSettings,
    report: Report = lambda d, t, s: None,
    cancelled: Callable[[], bool] = lambda: False,
    install_as: Path | None | str = "auto",
) -> TrainOutcome:
    """Train (or fine-tune) a YOLO model and copy ``best.pt`` to ``install_as``.

    ``"auto"`` installs it as ``installed_weights_for(settings.start_from)``; None
    leaves it only in the run folder.

    Progress is reported at the end of every batch and epoch. Cancelling sets the
    trainer's own ``stop`` flag, so Ultralytics finishes cleanly: it validates, saves
    ``last.pt``/``best.pt`` and the model trained so far can still be used.
    """
    settings.validate()
    require_ultralytics()
    from ultralytics import YOLO

    start_from = settings.start_from
    if Path(start_from).suffix == ".pt":
        start_from = str(weights_path(start_from)) if not Path(start_from).is_absolute() else start_from
        YOLO_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(start_from)
    device = settings.device or default_device()
    state = {"epoch": 0, "user_stop": False}

    def on_epoch_start(trainer: Any) -> None:
        state["batch"] = 0

    def on_batch_end(trainer: Any) -> None:
        if cancelled() and not trainer.stop:
            trainer.stop = True  # Ultralytics breaks the epoch, validates and saves
            state["user_stop"] = True
        batches = max(len(trainer.train_loader), 1)
        state["batch"] = min(state.get("batch", 0) + 1, batches)
        suffix = "  (stopping: validating and saving…)" if state["user_stop"] else ""
        report(
            trainer.epoch * batches + state["batch"],
            trainer.epochs * batches,
            f"Epoch {trainer.epoch + 1}/{trainer.epochs}, batch {state['batch']}/{batches}{suffix}",
        )

    def on_fit_epoch_end(trainer: Any) -> None:
        # Ultralytics calls this once more after the final validation of best.pt
        final = state["epoch"] >= trainer.epoch + 1 or trainer.epoch >= trainer.epochs
        state["epoch"] = max(state["epoch"], trainer.epoch + 1)
        metrics = trainer.metrics or {}
        map50 = metrics.get("metrics/mAP50(B)")
        text = "Final validation of best.pt" if final else f"Epoch {trainer.epoch + 1}/{trainer.epochs} done"
        if map50 is not None:
            text += f", validation mAP50 = {map50:.3f}"
        total = max(len(trainer.train_loader), 1) * trainer.epochs
        report(total if final else min(state["epoch"] * total // trainer.epochs, total), total, text)

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report(0, 1, f"Preparing training on {device_label(device)}…")
    model.train(
        data=str(Path(settings.data_yaml).resolve()),
        epochs=settings.epochs,
        batch=settings.batch,
        imgsz=settings.imgsz,
        patience=settings.patience,
        device=device,
        workers=settings.workers,
        seed=settings.seed,
        project=str(RUNS_DIR),
        name=settings.name,
        exist_ok=False,
        plots=True,
        verbose=False,
    )
    seconds = time.perf_counter() - started
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        best = run_dir / "weights" / "last.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but no weights were saved in {run_dir}.")
    installed = best
    if install_as == "auto":
        install_as = installed_weights_for(settings.start_from)
    if install_as is not None:
        install_as = Path(install_as)
        install_as.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, install_as)
        installed = install_as
    history = read_results_csv(run_dir / "results.csv")
    epochs_done = len(history.get("epoch", [])) or state["epoch"]
    return TrainOutcome(
        run_dir=run_dir,
        best_weights=best,
        installed_weights=installed,
        epochs_done=epochs_done,
        stopped_early=state["user_stop"] or epochs_done < settings.epochs,
        seconds=seconds,
        history=history,
    )


def ultralytics_validation(model: Any, data_yaml: Path, split: str = "test", device: str | None = None) -> dict[str, Any]:
    """The library's own metrics (``model.val``), to compare with ours."""
    metrics = model.val(
        data=str(data_yaml),
        split="val" if split == "valid" else split,
        device=device or default_device(),
        plots=False,
        verbose=False,
        workers=0,  # no worker processes: safe from a GUI thread, and fast enough for one split
        project=str(RUNS_DIR),
        name="val",
        exist_ok=True,
    )
    box = metrics.box
    names = metrics.names if hasattr(metrics, "names") else model.names
    per_class = {}
    for i, c in enumerate(getattr(box, "ap_class_index", [])):
        per_class[str(names[int(c)])] = (float(box.ap50[i]), float(box.ap[i]))
    return {"map50": float(box.map50), "map": float(box.map), "precision": float(box.mp), "recall": float(box.mr), "per_class": per_class}
