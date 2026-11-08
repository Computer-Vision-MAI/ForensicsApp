"""Set 7 bonus: hand-crafted BoW features versus features learnt by a CNN.

The theory ends with "Laboratory: comparing handcrafted vs deep learning features".
Here a ResNet-18 pre-trained on ImageNet is used as a fixed feature extractor: the
image goes through the network up to the global average pooling, which gives 512
numbers per image (its "deep descriptor"). The final ImageNet layer is removed, and the
network is never retrained.

To make the comparison fair, the CNN uses exactly the same train/test split as the
trained BoW model and the same protocol: an SVM whose C is chosen by cross-validation on
train, probabilities from Platt calibration, and one evaluation on the test images.

Caveat for the discussion: ImageNet already contains dalmatians, butterflies and a
wallaby (a small kangaroo), so the CNN starts with very relevant knowledge. That is the
point of transfer learning, but it explains part of the gap.

PyTorch is an optional dependency (``requirements-cnn.txt``): it is imported only when
the CNN is actually used, so the rest of the app and the tests work without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import time

import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from .bow_model import (
    CV_FOLDS,
    SVM_C_GRID,
    UNUSED_SVC_GAMMA,
    BowModel,
    Cancelled,
    Report,
    _check,
    _never,
    _no_report,
    describe_image,
    evaluate,
    normalize,
)
from .image_lookup import load_image
from .matching import working_image

RESIZE_SIDE = 256  # shorter side before the centre crop (the ImageNet recipe)
CROP_SIDE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BATCH_SIZE = 16
INSTALL_HINT = (
    "PyTorch is not installed, so the CNN comparison is not available. Install the optional "
    "CNN dependencies with:  .venv/bin/pip install -r requirements-cnn.txt"
)

# (n, 3, 224, 224) float32 batch -> (n, d) features
EmbedFn = Callable[[np.ndarray], np.ndarray]


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


# --------------------------------------------------------------------------- input


def to_rgb(img: Image.Image) -> Image.Image:
    """RGB copy of any PIL mode; high-depth images (I, F, I;16) are normalised to 8 bits first."""
    mode = img.mode
    if mode in ("I", "F") or mode.startswith("I;16"):
        gray, _ = working_image(img, 0)  # 0: keep the size
        return gray.convert("RGB")
    if mode == "La":  # premultiplied modes cannot be converted directly
        img = img.convert("LA")
    elif mode == "RGBa":
        img = img.convert("RGBA")
    return img.convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    """The input ResNet was trained with: (3, 224, 224) float32, ImageNet-normalised.

    Resize so the shorter side is 256 px (keeping the aspect ratio), cut the central
    224 x 224 square, scale to [0, 1] and standardise each channel with the ImageNet
    mean and standard deviation. Same steps as torchvision's ResNet18_Weights.transforms().
    """
    rgb = to_rgb(img)
    if rgb.width == 0 or rgb.height == 0:
        raise ValueError("The image is empty.")
    scale = RESIZE_SIDE / min(rgb.size)
    size = (max(CROP_SIDE, round(rgb.width * scale)), max(CROP_SIDE, round(rgb.height * scale)))
    rgb = rgb.resize(size, Image.Resampling.BILINEAR)
    left = (rgb.width - CROP_SIDE) // 2
    top = (rgb.height - CROP_SIDE) // 2
    rgb = rgb.crop((left, top, left + CROP_SIDE, top + CROP_SIDE))
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0  # (224, 224, 3)
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))  # channels first, as PyTorch expects


# --------------------------------------------------------------------------- network


def load_resnet18(pretrained: bool = True) -> EmbedFn:
    """ResNet-18 without its last (ImageNet classification) layer, as a function batch -> features.

    With ``pretrained`` the ImageNet weights are used; torchvision downloads them once
    (about 45 MB) to ~/.cache/torch, outside the repository. Raises ValueError with
    installation instructions when PyTorch is missing.
    """
    if not torch_available():
        raise ValueError(INSTALL_HINT)
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    network.fc = torch.nn.Identity()  # keep the 512 pooled features, drop the 1000 ImageNet scores
    network.eval()  # inference mode for batch-norm layers

    def embed(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return network(torch.from_numpy(batch)).numpy()

    return embed


def embed_images(
    images: list[Image.Image | str | Path],
    embed_fn: EmbedFn,
    report: Report = _no_report,
    cancelled: Cancelled = _never,
    label: str = "CNN features",
) -> np.ndarray:
    """Return one row of CNN features per image (PIL images or file paths), in order."""
    rows = []
    for start in range(0, len(images), BATCH_SIZE):
        _check(cancelled)
        report(start, len(images), f"{label}: {start}/{len(images)} images")
        batch = []
        for item in images[start : start + BATCH_SIZE]:
            if isinstance(item, Image.Image):
                batch.append(preprocess(item))
                continue
            try:
                image = load_image(Path(item))
            except OSError as error:
                raise ValueError(f"Could not read '{item}': {error}. Remove or replace this file.") from error
            batch.append(preprocess(image))
        rows.append(np.asarray(embed_fn(np.stack(batch)), dtype=np.float64))
    report(len(images), len(images), f"{label}: done")
    return np.vstack(rows) if rows else np.zeros((0, 0))


# --------------------------------------------------------------------------- classifier


@dataclass
class EmbeddingClassifier:
    """Linear SVM on L2-normalised CNN features, with Platt-calibrated probabilities."""

    model: CalibratedClassifierCV
    c: float

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(normalize(np.atleast_2d(features), "l2"))


def train_embedding_classifier(
    features: np.ndarray, labels: np.ndarray, seed: int = 0, c_grid: tuple[float, ...] = SVM_C_GRID
) -> tuple[EmbeddingClassifier, float]:
    """Choose C by stratified cross-validation on train, then fit and calibrate the SVM.

    CNN features are not histograms, so a linear kernel on L2-normalised vectors is used
    (the usual choice for deep features) instead of the chi-squared kernel.
    Returns the classifier and the cross-validation accuracy of the chosen C.
    """
    labels = np.asarray(labels)
    vectors = normalize(features, "l2")
    folds = max(2, min(CV_FOLDS, int(np.bincount(labels).min())))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {c: [] for c in c_grid}
    for fit_idx, val_idx in splitter.split(vectors, labels):
        for c in c_grid:
            svc = SVC(kernel="linear", C=c, gamma=UNUSED_SVC_GAMMA, class_weight="balanced")
            svc.fit(vectors[fit_idx], labels[fit_idx])
            scores[c].append(accuracy_score(labels[val_idx], svc.predict(vectors[val_idx])))
    best = max(c_grid, key=lambda c: (np.mean(scores[c]), -c))
    calibrated = CalibratedClassifierCV(
        SVC(kernel="linear", C=best, gamma=UNUSED_SVC_GAMMA, class_weight="balanced"),
        method="sigmoid",
        ensemble=False,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
    )
    calibrated.fit(vectors, labels)
    return EmbeddingClassifier(calibrated, best), float(np.mean(scores[best]))


# --------------------------------------------------------------------------- comparison


@dataclass
class Comparison:
    """BoW and CNN evaluated on the same test images."""

    classes: list[str]
    test_paths: list[str]
    test_labels: np.ndarray
    bow_predicted: np.ndarray
    cnn_predicted: np.ndarray
    bow_metrics: dict[str, object]
    cnn_metrics: dict[str, object]
    bow_cv_accuracy: float
    cnn_cv_accuracy: float
    cnn_c: float
    cnn_dimensions: int
    bow_dimensions: int
    cnn_seconds: float
    classifier: EmbeddingClassifier = field(repr=False)

    def disagreements(self) -> list[tuple[str, str, str, str]]:
        """(file, true class, BoW prediction, CNN prediction) where the two methods differ."""
        rows = []
        for path, true, bow, cnn in zip(self.test_paths, self.test_labels, self.bow_predicted, self.cnn_predicted):
            if bow != cnn:
                rows.append((Path(path).name, self.classes[true], self.classes[bow], self.classes[cnn]))
        return rows


def bow_test_predictions(model: BowModel, report: Report = _no_report, cancelled: Cancelled = _never) -> np.ndarray:
    """Predicted class of every test image with the trained BoW model.

    Uses the classifier on the full vector, exactly like ``evaluate`` did during
    training, so the accuracy matches the one reported then.
    """
    predicted = []
    for index, path in enumerate(model.test_paths):
        _check(cancelled)
        report(index, len(model.test_paths), f"BoW predictions: {Path(path).name}")
        analysis = describe_image(model, load_image(Path(path)), path)
        probabilities = model.classifier.predict_proba(analysis.vector)[0]
        predicted.append(int(model.classifier.classes_[probabilities.argmax()]))
    return np.array(predicted, dtype=int)


def compare_with_cnn(
    model: BowModel,
    embed_fn: EmbedFn,
    report: Report = _no_report,
    cancelled: Cancelled = _never,
) -> Comparison:
    """Train and test a CNN-feature classifier on the BoW model's own split, and compare."""
    start = time.perf_counter()
    train = embed_images(model.train_paths, embed_fn, report, cancelled, "CNN features (train)")
    test = embed_images(model.test_paths, embed_fn, report, cancelled, "CNN features (test)")
    _check(cancelled)
    report(1, 1, "Training the SVM on CNN features")
    classifier, cv_accuracy = train_embedding_classifier(train, model.train_labels, model.settings.seed)
    cnn_metrics = evaluate(classifier, test, model.test_labels, model.classes)
    cnn_predicted = classifier.classes_[classifier.predict_proba(test).argmax(axis=1)]
    cnn_seconds = time.perf_counter() - start

    bow_predicted = bow_test_predictions(model, report, cancelled)
    return Comparison(
        classes=list(model.classes),
        test_paths=list(model.test_paths),
        test_labels=np.asarray(model.test_labels),
        bow_predicted=bow_predicted,
        cnn_predicted=np.asarray(cnn_predicted, dtype=int),
        bow_metrics=model.metrics,
        cnn_metrics=cnn_metrics,
        bow_cv_accuracy=float(model.metrics["cv_accuracy"]),
        cnn_cv_accuracy=cv_accuracy,
        cnn_c=classifier.c,
        cnn_dimensions=train.shape[1],
        bow_dimensions=model.classifier.train_vectors.shape[1],
        cnn_seconds=cnn_seconds,
        classifier=classifier,
    )


def classify_with_cnn(classifier: EmbeddingClassifier, embed_fn: EmbedFn, img: Image.Image, classes: list[str]) -> list[tuple[str, float]]:
    """(class, probability) of one image with the CNN classifier, most likely first."""
    probabilities = classifier.predict_proba(embed_images([img], embed_fn))[0]
    ranked = sorted(zip(classifier.classes_, probabilities), key=lambda item: -item[1])
    return [(classes[int(label)], float(p)) for label, p in ranked]
