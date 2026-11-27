"""Set 10 extra: train, save, load and run the Vision Transformer of the extra notebook.

Training follows ``run_experiment`` of the notebook: AdamW (learning rate 1e-3, weight
decay 1e-4), batch 256, 10 epochs, ``validation_split=0.1`` (Keras keeps the **last** 10 %
of the training images for validation), and the weights of the epoch with the best
validation accuracy are kept (``ModelCheckpoint(save_best_only=True)``). The test set is
evaluated once: loss, accuracy and top-5 accuracy, plus the confusion matrix and the
per-class precision / recall / F1 asked by the notebook's exercises.

With little GPU memory (an 8 GB Mac) a batch of 256 images does not fit through 8
attention layers at once, so it can be split into micro-batches whose gradients are
added up before the optimiser step (gradient accumulation): mathematically the same
batch of 256, only the dropout masks differ.

Each model is a folder ``models/vit/<name>/`` (git-ignored) with ``model.pt`` (weights and
the settings to rebuild the network) and ``passport.json`` (history and test metrics).
PyTorch is imported only inside the functions that need it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
from .unet_model import default_device, require_torch
from .vit_data import DATASETS, CifarData, keras_validation_split, load_cifar

VIT_DIR = MODELS_DIR / "vit"
PASSPORT_FORMAT = "forensics-vit/1"
NO_MODELS = "There is no trained ViT yet. Train one with 'ViT: Train Classifier' first."

Report = Callable[[int, int, str], None]


@dataclass(frozen=True)
class VitSettings:
    dataset: str = "cifar100"
    spt: bool = False
    lsa: bool = False
    epochs: int = 10
    batch: int = 256
    micro_batch: int = 64  # 0 = the whole batch at once
    lr: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 72
    patch: int = 6
    dim: int = 64
    heads: int = 4
    layers: int = 8
    val_fraction: float = 0.1
    seed: int = 42
    device: str = ""
    name: str = ""

    def default_name(self) -> str:
        kind = "spt-lsa" if self.spt and self.lsa else "spt" if self.spt else "lsa" if self.lsa else "patches"
        return f"vit_{self.dataset}_{kind}_{self.epochs}e"


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or "vit_model"


def architecture_summary(s: VitSettings) -> str:
    n = (s.image_size // s.patch) ** 2
    tokens = (f"shifted patch tokenization (image + 4 half-patch shifts = 15 channels, {s.patch}×{s.patch}×15 = "
              f"{s.patch * s.patch * 15} values, LayerNorm)" if s.spt else
              f"patches {s.patch}×{s.patch}×3 = {s.patch * s.patch * 3} values")
    attention = "locality self-attention (learnt temperature, no self-token)" if s.lsa else "multi-head self-attention"
    return (f"{s.image_size}×{s.image_size} input → {n} tokens from {tokens} → Dense {s.dim} + position embedding → "
            f"{s.layers} × [LayerNorm, {attention} ({s.heads} heads, key_dim {s.dim}), skip, LayerNorm, "
            f"MLP {s.dim * 2}-{s.dim} GELU, skip] → LayerNorm → flatten ({n * s.dim}) → MLP 2048-1024 → classes")


# --------------------------------------------------------------------------- passport


@dataclass
class VitPassport:
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
        return list(self.data["classes"])

    @property
    def settings(self) -> VitSettings:
        known = VitSettings.__dataclass_fields__
        return VitSettings(**{k: v for k, v in self.data.get("settings", {}).items() if k in known})

    @property
    def history(self) -> dict[str, list[float]]:
        return self.data.get("history", {})

    @property
    def test(self) -> dict[str, Any]:
        return self.data.get("test", {})

    @property
    def label(self) -> str:
        test = self.test
        score = f" (test accuracy {test['accuracy']:.1%}, top-5 {test['top5']:.1%})" if test else ""
        return f"{self.name}{score}"


def read_passport(folder: Path) -> VitPassport:
    path = Path(folder) / "passport.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != PASSPORT_FORMAT:
        raise ValueError(f"{path} is not a ViT passport.")
    return VitPassport(Path(folder), data)


def list_models(directory: Path = VIT_DIR) -> list[VitPassport]:
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


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    """Confusion matrix and precision / recall / F1 per class (as sklearn's classification_report)."""
    confusion = np.bincount(y_true * num_classes + y_pred, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = np.diag(confusion).astype(float)
    predicted = confusion.sum(axis=0)
    support = confusion.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        "confusion": confusion.astype(int).tolist(),
        "precision": np.round(precision, 4).tolist(),
        "recall": np.round(recall, 4).tolist(),
        "f1": np.round(f1, 4).tolist(),
        "support": support.astype(int).tolist(),
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float((f1 * support).sum() / max(support.sum(), 1)), 4),
    }


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    top = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean((top == labels[:, None]).any(axis=1)))


# --------------------------------------------------------------------------- network


def build_network(s: VitSettings, num_classes: int, mean: np.ndarray, std: np.ndarray):
    require_torch()
    import torch

    from .vit_net import ViT

    return ViT(num_classes, torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32),
               s.image_size, s.patch, s.dim, s.heads, s.layers, spt=s.spt, lsa=s.lsa)


_NET_CACHE: dict[tuple[str, float, str], Any] = {}
_NET_LOCK = threading.Lock()


def load_network(passport: VitPassport, device: str | None = None):
    require_torch()
    import torch

    device = device or default_device()
    key = (str(passport.weights), passport.weights.stat().st_mtime, device)
    with _NET_LOCK:
        if key not in _NET_CACHE:
            saved = torch.load(passport.weights, map_location="cpu", weights_only=True)
            config = saved["config"]
            net = build_network(passport.settings, int(config["num_classes"]), np.array(config["mean"]),
                                np.array(config["std"]))
            net.load_state_dict(saved["state_dict"])
            _NET_CACHE.clear()
            _NET_CACHE[key] = net.to(device).eval()
        return _NET_CACHE[key]


def _batches(net, images: np.ndarray, device: str, batch: int = 256):
    import torch

    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = torch.from_numpy(np.ascontiguousarray(images[start:start + batch])).to(device)
            yield net(x.permute(0, 3, 1, 2).float())


def predict_logits(net, images: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    net.eval()
    return np.concatenate([logits.cpu().numpy() for logits in _batches(net, images, device, batch)])


# --------------------------------------------------------------------------- training


@dataclass
class VitOutcome:
    passport: VitPassport
    stopped_early: bool
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


def train(settings: VitSettings, report: Report, cancelled: Callable[[], bool],
          target_dir: Path = VIT_DIR, data: CifarData | None = None) -> VitOutcome:
    require_torch()
    import torch
    from torch.nn import functional as F

    from .vit_net import count_parameters

    device = settings.device or default_device()
    started = time.perf_counter()
    data = data or load_cifar(settings.dataset)
    num_classes = len(data.class_names)
    # Keras' Normalization.adapt(x_train): mean and variance of every channel over all training pixels
    mean = data.x_train.reshape(-1, 3).mean(axis=0)
    std = data.x_train.reshape(-1, 3).std(axis=0)
    train_idx, val_idx = keras_validation_split(len(data.x_train), settings.val_fraction)

    torch.manual_seed(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    net = build_network(settings, num_classes, mean, std).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    x_train = torch.from_numpy(np.ascontiguousarray(data.x_train[train_idx])).to(device)
    y_train = torch.from_numpy(data.y_train[train_idx]).to(device)
    x_val, y_val = data.x_train[val_idx], data.y_train[val_idx]
    micro = settings.micro_batch or settings.batch

    history: dict[str, list[float]] = {k: [] for k in ("loss", "accuracy", "top5", "val_loss", "val_accuracy",
                                                       "val_top5", "seconds")}
    best_acc, best_epoch, best_state = -1.0, 0, None
    batches = (len(train_idx) + settings.batch - 1) // settings.batch
    total = settings.epochs * batches
    stopped = False
    for epoch in range(1, settings.epochs + 1):
        net.train()
        epoch_start = time.perf_counter()
        order = torch.randperm(len(train_idx), generator=generator).to(device)
        loss_sum = correct = top5 = seen = 0.0
        for b in range(batches):
            if cancelled():
                stopped = True
                break
            idx = order[b * settings.batch:(b + 1) * settings.batch]
            optimizer.zero_grad()
            for start in range(0, len(idx), micro):
                part = idx[start:start + micro]
                logits = net(x_train[part].permute(0, 3, 1, 2).float())
                target = y_train[part]
                loss = F.cross_entropy(logits, target)
                (loss * len(part) / len(idx)).backward()  # gradients add up to those of the whole batch
                with torch.no_grad():
                    loss_sum += loss.item() * len(part)
                    best5 = logits.topk(5, dim=1).indices
                    correct += (best5[:, 0] == target).sum().item()
                    top5 += (best5 == target[:, None]).any(dim=1).sum().item()
                    seen += len(part)
            optimizer.step()
            step = (epoch - 1) * batches + b + 1
            if b % 5 == 0 or b == batches - 1:
                best = f", best val {best_acc:.1%} (epoch {best_epoch})" if best_epoch else ""
                report(step, total, f"Epoch {epoch}/{settings.epochs}, batch {b + 1}/{batches}: loss "
                                    f"{loss_sum / max(seen, 1):.3f}, accuracy {correct / max(seen, 1):.1%}{best}")
        if stopped:
            break
        val_logits = predict_logits(net, x_val, device)
        val_loss = float(F.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(y_val)).item())
        val_acc = float(np.mean(val_logits.argmax(1) == y_val))
        history["loss"].append(loss_sum / seen)
        history["accuracy"].append(correct / seen)
        history["top5"].append(top5 / seen)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_top5"].append(top_k_accuracy(val_logits, y_val))
        history["seconds"].append(time.perf_counter() - epoch_start)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        report(epoch * batches, total, f"Epoch {epoch}/{settings.epochs}: loss {history['loss'][-1]:.3f}, "
                                       f"val accuracy {val_acc:.1%} (best {best_acc:.1%}, epoch {best_epoch})")
    del x_train, y_train
    if best_state is None:
        raise TaskCancelled()

    net.load_state_dict(best_state)
    report(total, total, "Evaluating the best epoch on the test set…")
    test_logits = predict_logits(net, data.x_test, device)
    test_pred = test_logits.argmax(axis=1)
    test_loss = float(F.cross_entropy(torch.from_numpy(test_logits), torch.from_numpy(data.y_test)).item())
    seconds = time.perf_counter() - started

    name = safe_name(settings.name or settings.default_name())
    folder = Path(target_dir) / name
    temporary = Path(target_dir) / f".{name}.saving"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    config = {"num_classes": num_classes, "mean": mean.tolist(), "std": std.tolist()}
    torch.save({"state_dict": best_state, "config": config}, temporary / "model.pt")
    passport = {
        "format": PASSPORT_FORMAT,
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "framework": {"torch": torch.__version__, "python": platform.python_version()},
        "dataset": {"key": settings.dataset, "label": DATASETS[settings.dataset].label,
                    "train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(data.x_test))},
        "classes": data.class_names,
        "architecture": {"summary": architecture_summary(settings), "parameters": count_parameters(net),
                         "tokens": (settings.image_size // settings.patch) ** 2,
                         "tokenizer": "SPT" if settings.spt else "patches",
                         "attention": "LSA" if settings.lsa else "MHSA"},
        "training": {"optimizer": f"AdamW (lr {settings.lr:g}, weight decay {settings.weight_decay:g})",
                     "loss": "sparse categorical cross-entropy", "batch": settings.batch,
                     "micro_batch": settings.micro_batch, "epochs": settings.epochs,
                     "epochs_done": len(history["loss"]), "stopped_early": stopped, "best_epoch": best_epoch,
                     "selection": "best validation accuracy", "device": device, "seconds": round(seconds, 1),
                     "augmentation": "normalise, resize, random flip, rotation ±0.02 turn, zoom ±20 %"},
        "history": {k: [round(float(v), 5) for v in values] for k, values in history.items()},
        "test": {"loss": round(test_loss, 5), "accuracy": round(float(np.mean(test_pred == data.y_test)), 5),
                 "top5": round(top_k_accuracy(test_logits, data.y_test), 5),
                 "predictions": test_pred.astype(int).tolist(), "labels": data.y_test.astype(int).tolist(),
                 **classification_report(data.y_test, test_pred, num_classes)},
        "settings": asdict(settings),
    }
    (temporary / "passport.json").write_text(json.dumps(passport, ensure_ascii=False), encoding="utf-8")
    if folder.exists():
        shutil.rmtree(folder)
    temporary.rename(folder)
    return VitOutcome(read_passport(folder), stopped, seconds)


# --------------------------------------------------------------------------- one image


def to_cifar_input(image: Image.Image) -> np.ndarray:
    """Centre square of any image resized to 32 × 32: the size the network was trained on."""
    rgb = image.convert("RGB")
    side = min(rgb.size)
    left, top = (rgb.width - side) // 2, (rgb.height - side) // 2
    return np.asarray(rgb.crop((left, top, left + side, top + side)).resize((32, 32), Image.Resampling.BICUBIC))


def attention_rollout(maps: list[np.ndarray]) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020) for a model without a class token.

    Each layer's attention (mean of the heads) gets the identity added for the skip
    connection, A' = (A + I) / 2, and the layers are multiplied: R = A'_L … A'_1. R[i, j]
    says how much input patch j reaches output token i. The head reads all tokens, so
    the importance of patch j is the mean over i.
    """
    n = maps[0].shape[-1]
    rollout = np.eye(n)
    for attention in maps:
        a = attention.mean(axis=0) if attention.ndim == 3 else attention
        a = 0.5 * a + 0.5 * np.eye(n)
        a = a / a.sum(axis=-1, keepdims=True)
        rollout = a @ rollout
    importance = rollout.mean(axis=0)
    side = int(round(np.sqrt(n)))
    return importance.reshape(side, side)


@dataclass
class VitPrediction:
    probs: np.ndarray
    network_input: np.ndarray  # 32 × 32 × 3
    rollout: np.ndarray | None  # (patches per side, patches per side)
    seconds: float


def classify(passport: VitPassport, image: Image.Image | np.ndarray, rollout: bool = True,
             device: str | None = None) -> VitPrediction:
    import torch

    started = time.perf_counter()
    small = to_cifar_input(image) if isinstance(image, Image.Image) else np.asarray(image, dtype=np.uint8)
    net = load_network(passport, device)
    device = next(net.parameters()).device.type
    net.keep_attention(rollout)
    try:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(small[None])).to(device).permute(0, 3, 1, 2).float()
            probs = net(x).softmax(dim=1)[0].cpu().numpy()
        maps = [m[0].cpu().numpy() for m in net.attention_maps()] if rollout else None
    finally:
        net.keep_attention(False)
    return VitPrediction(probs, small, attention_rollout(maps) if maps else None, time.perf_counter() - started)
