"""Set 7: object recognition with a Bag of Visual Words (BoW).

The pipeline follows the theory slides (Class 6):

1. Extract local features: SIFT keypoints (DoG detector) or a dense DAISY grid.
2. Learn the visual vocabulary: K-Means on the *training* descriptors only, so no
   information from the test images leaks into the model. Each centroid is a visual word.
3. Represent each image as a histogram of visual words (hard or soft assignment,
   optional spatial pyramid), weighted by TF or TF-IDF and normalised (L1 or L2).
4. Classify with an SVM using the chi-squared kernel, which suits histograms.

Everything here is plain NumPy / scikit-learn: no Tkinter windows, so it can be tested
without a display and run on a worker thread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from skimage import feature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import additive_chi2_kernel, euclidean_distances
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from .dialogs import TaskCancelled
from .image_lookup import list_image_files, load_image
from .keypoint_detection import extract_keypoints
from .matching import working_image

DETECTORS = ("sift", "dense")
ASSIGNMENTS = ("hard", "soft")
WEIGHTINGS = ("tf", "tfidf")
NORMS = ("l1", "l2")
MODEL_VERSION = 1
DEFAULT_MODEL_PATH = Path("models") / "bow_model.joblib"

# Dense DAISY grid: one descriptor every DENSE_STEP pixels, with
# (rings x histograms + 1) x orientations = 104 values
DENSE_STEP = 8
DENSE_RADIUS = 15
DENSE_RINGS = 2
DENSE_HISTOGRAMS = 6
DENSE_ORIENTATIONS = 8
# Regularisation values tried for the SVM (chosen by cross-validation on train)
SVM_C_GRID = (0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5
EXEMPLARS_PER_WORD = 5
PATCH_SIDE = 32
NEIGHBOURS = 5

Report = Callable[[int, int, str], None]
Cancelled = Callable[[], bool]


def _no_report(done: int, total: int, label: str) -> None:
    """Default progress callback: ignore the update."""


def _never() -> bool:
    return False


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class BowSettings:
    """Every choice of the pipeline, with the course defaults."""

    detector: str = "sift"
    k: int | None = None  # None means 10 x number of classes
    auto_k: bool = False  # choose K by cross-validation among k_multipliers x classes
    k_multipliers: tuple[int, ...] = (5, 10, 20, 40)
    test_ratio: float = 0.2
    seed: int = 0
    assignment: str = "hard"
    soft_sigma: float = 0.25  # soft width, relative to the mean descriptor-to-word distance
    weighting: str = "tfidf"
    norm: str = "l1"
    pyramid_levels: int = 2  # 0: K bins, 1: K(1+4), 2: K(1+4+16); 2 gave the best CV accuracy
    max_side: int = 300
    max_kmeans_samples: int = 100_000

    def __post_init__(self) -> None:
        if self.detector not in DETECTORS:
            raise ValueError(f"Unknown detector '{self.detector}'. Use one of {DETECTORS}.")
        if self.assignment not in ASSIGNMENTS:
            raise ValueError(f"Unknown assignment '{self.assignment}'. Use one of {ASSIGNMENTS}.")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"Unknown weighting '{self.weighting}'. Use one of {WEIGHTINGS}.")
        if self.norm not in NORMS:
            raise ValueError(f"Unknown normalisation '{self.norm}'. Use one of {NORMS}.")
        if self.k is not None and self.k < 2:
            raise ValueError(f"K must be at least 2 (got {self.k}).")
        if not 0.05 <= self.test_ratio <= 0.5:
            raise ValueError(f"The test ratio must be between 0.05 and 0.5 (got {self.test_ratio}).")
        if not 0 <= self.pyramid_levels <= 2:
            raise ValueError(f"Pyramid levels must be 0, 1 or 2 (got {self.pyramid_levels}).")
        if self.soft_sigma <= 0:
            raise ValueError(f"The soft assignment width must be positive (got {self.soft_sigma}).")
        if self.max_side < 32:
            raise ValueError(f"The working size must be at least 32 pixels (got {self.max_side}).")
        if not self.k_multipliers or min(self.k_multipliers) < 1:
            raise ValueError("The K multipliers must be positive integers.")

    def k_for(self, n_classes: int) -> int:
        return self.k if self.k is not None else 10 * n_classes

    def describe(self) -> dict[str, object]:
        detector = "SIFT (DoG keypoints)" if self.detector == "sift" else f"Dense DAISY (every {DENSE_STEP} px)"
        cells = sum(4**level for level in range(self.pyramid_levels + 1))
        return {
            "Detector": detector,
            "Assignment": "hard (nearest word)" if self.assignment == "hard" else f"soft (sigma x{self.soft_sigma:g})",
            "Weighting": "TF-IDF" if self.weighting == "tfidf" else "TF",
            "Normalisation": self.norm.upper(),
            "Spatial pyramid": f"{self.pyramid_levels} levels ({cells} cells)" if self.pyramid_levels else "off",
            "Working size": f"longest side <= {self.max_side} px",
            "Seed": self.seed,
        }


# --------------------------------------------------------------------------- dataset


def list_dataset(folder: str | Path) -> dict[str, list[Path]]:
    """Return {class name: image paths} for a folder with one subfolder per class.

    Raises ValueError when there are fewer than two classes or a class has fewer than two images.
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"'{root}' is not a folder.")
    dataset = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith((".", "__")):
            continue
        images = list_image_files(sub, recursive=True)
        if images:
            dataset[sub.name] = images
    if len(dataset) < 2:
        raise ValueError(
            f"'{root.name}' needs at least two class subfolders with images "
            f"(found {len(dataset)}). Choose the folder that contains one folder per class."
        )
    small = [name for name, paths in dataset.items() if len(paths) < 2]
    if small:
        raise ValueError(f"Each class needs at least two images; too few in: {', '.join(small)}.")
    return dataset


def split_dataset(
    dataset: dict[str, list[Path]], test_ratio: float, seed: int
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """Stratified train/test split: every class keeps the same proportion in both parts."""
    paths = [path for name in dataset for path in dataset[name]]
    labels = [index for index, name in enumerate(dataset) for _ in dataset[name]]
    n_test = max(len(dataset), round(test_ratio * len(paths)))
    if n_test >= len(paths) - len(dataset) + 1:
        raise ValueError("The dataset is too small for this test ratio.")
    train_p, test_p, train_y, test_y = train_test_split(
        paths, labels, test_size=n_test, stratify=labels, random_state=seed
    )
    return list(train_p), list(train_y), list(test_p), list(test_y)


# --------------------------------------------------------------------------- features


@dataclass
class LocalFeatures:
    """Local descriptors of one image, in the coordinates of its working copy.

    ``positions`` are (row, col), ``scales`` the radius of each patch in pixels and
    ``shape`` the (height, width) of the working image.
    """

    positions: np.ndarray
    scales: np.ndarray
    descriptors: np.ndarray
    shape: tuple[int, int]

    def __len__(self) -> int:
        return len(self.descriptors)


def descriptor_size(detector: str) -> int:
    if detector == "sift":
        return 128
    return (DENSE_RINGS * DENSE_HISTOGRAMS + 1) * DENSE_ORIENTATIONS


def extract_features(img: Image.Image, settings: BowSettings) -> tuple[LocalFeatures, Image.Image]:
    """Return the local features of ``img`` and the grayscale working copy they refer to.

    An image without features (e.g. a flat one) returns empty arrays instead of failing.
    """
    gray, _ = working_image(img, settings.max_side)
    shape = (gray.height, gray.width)
    dim = descriptor_size(settings.detector)
    if settings.detector == "sift":
        keypoints, descriptors, sigmas, _ = extract_keypoints(gray, method="sift", return_meta=True)
        if descriptors is None or len(keypoints) == 0:
            return _empty_features(shape, dim), gray # it gives zeros in a future histogram, which is better than failing
        # the SIFT descriptor looks at a window of a few sigmas around the keypoint
        scales = np.clip(3.0 * np.asarray(sigmas, dtype=np.float64), 6.0, 48.0)
        return LocalFeatures(np.asarray(keypoints, float), scales, descriptors.astype(np.float32), shape), gray

    pixels = np.asarray(gray, dtype=np.float64) / 255.0
    if min(shape) <= 2 * DENSE_RADIUS:
        return _empty_features(shape, dim), gray
    grid = feature.daisy(
        pixels,
        step=DENSE_STEP,
        radius=DENSE_RADIUS,
        rings=DENSE_RINGS,
        histograms=DENSE_HISTOGRAMS,
        orientations=DENSE_ORIENTATIONS,
    )
    rows, cols = np.meshgrid(
        DENSE_RADIUS + DENSE_STEP * np.arange(grid.shape[0]),
        DENSE_RADIUS + DENSE_STEP * np.arange(grid.shape[1]),
        indexing="ij",
    )
    positions = np.column_stack([rows.ravel(), cols.ravel()]).astype(np.float64)
    descriptors = grid.reshape(-1, grid.shape[-1]).astype(np.float32)
    scales = np.full(len(positions), float(DENSE_RADIUS))
    return LocalFeatures(positions, scales, descriptors, shape), gray


def _empty_features(shape: tuple[int, int], dim: int) -> LocalFeatures:
    return LocalFeatures(np.empty((0, 2)), np.empty(0), np.empty((0, dim), np.float32), shape)


# --------------------------------------------------------------------------- vocabulary


def build_codebook(descriptors: np.ndarray, k: int, seed: int = 0, max_samples: int = 100_000) -> np.ndarray:
    """Cluster descriptors with K-Means and return the K centroids (the visual words).

    At most ``max_samples`` random descriptors are used, which keeps K-Means fast
    without changing the result much.
    """
    if len(descriptors) < k:
        raise ValueError(f"Only {len(descriptors)} descriptors were found, fewer than K = {k}.")
    rng = np.random.default_rng(seed)
    if len(descriptors) > max_samples:
        descriptors = descriptors[rng.choice(len(descriptors), max_samples, replace=False)]
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_.astype(np.float32)


def nearest_words(descriptors: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vector quantisation: index of the closest centroid and the distance to it."""
    if len(descriptors) == 0:
        return np.empty(0, dtype=int), np.empty(0)
    distances = euclidean_distances(descriptors, centers)
    words = distances.argmin(axis=1)
    return words, distances[np.arange(len(words)), words]


def assign_words(descriptors: np.ndarray, centers: np.ndarray, assignment: str = "hard", sigma: float = 1.0) -> np.ndarray:
    """Return an (N, K) weight matrix: how much each descriptor votes for each word.

    Hard: one vote for the closest word. Soft (kernel codebook): Gaussian weights
    ``exp(-d^2 / 2 sigma^2)`` over all words, so a descriptor near a border between two
    words votes for both. Every row sums to 1.
    """
    n, k = len(descriptors), len(centers)
    if n == 0:
        return np.zeros((0, k))
    if assignment == "hard":
        words, _ = nearest_words(descriptors, centers)
        weights = np.zeros((n, k))
        weights[np.arange(n), words] = 1.0
        return weights
    squared = euclidean_distances(descriptors, centers, squared=True)
    # subtracting each row's minimum avoids exp() underflowing to 0 everywhere
    logits = -(squared - squared.min(axis=1, keepdims=True)) / (2.0 * sigma**2)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def pyramid_length(k: int, levels: int) -> int:
    return k * sum(4**level for level in range(levels + 1))


def pyramid_level_weight(level: int, levels: int) -> float:
    """Lazebnik weights: 1/2^L for level 0 and 1/2^(L-l+1) for level l >= 1 (1 without pyramid)."""
    if levels == 0:
        return 1.0
    return 1.0 / 2**levels if level == 0 else 1.0 / 2 ** (levels - level + 1)


def encode(features: LocalFeatures, weights: np.ndarray, levels: int = 0) -> np.ndarray:
    """Return the raw (unnormalised) bag of words, with the spatial pyramid if ``levels > 0``.

    Level l splits the image into 2^l x 2^l cells and adds one K-bin histogram per cell;
    everything is concatenated (length K(1+4+16) for 2 levels). Following Lazebnik et al.
    (2006), finer levels get larger weights because matches there are more precise.
    """
    k = weights.shape[1]
    parts = []
    height, width = features.shape
    for level in range(levels + 1):
        cells = 2**level
        level_weight = pyramid_level_weight(level, levels)
        hist = np.zeros((cells, cells, k))
        if len(features):
            rows = np.minimum((features.positions[:, 0] * cells / height).astype(int), cells - 1)
            cols = np.minimum((features.positions[:, 1] * cells / width).astype(int), cells - 1)
            np.add.at(hist, (rows, cols), weights)
        parts.append(level_weight * hist.reshape(-1))
    return np.concatenate(parts)


def idf_weights(train_counts: np.ndarray) -> np.ndarray:
    """Inverse document frequency of each bin, learnt from the training histograms.

    The slide formula ``log(N / (1 + df))`` becomes 0 or negative for a word that appears
    in (almost) every image, which is common with a small vocabulary and hundreds of
    keypoints per image, and negative values would break the chi-squared kernel. We use
    the smoothed form of scikit-learn's TfidfTransformer, ``log((1 + N) / (1 + df)) + 1``:
    it keeps the same ordering (rare words weigh more) and is always positive.
    """
    n = len(train_counts)
    df = np.count_nonzero(train_counts > 0, axis=0)
    return np.log((1.0 + n) / (1.0 + df)) + 1.0


def normalize(vector: np.ndarray, norm: str = "l1") -> np.ndarray:
    """L1 (bins sum to 1) or L2 (unit length) normalisation; an all-zero vector stays zero.

    Images have different numbers of features, so raw counts depend on the image size,
    not only on its content. Normalising removes that effect.
    """
    vector = np.asarray(vector, dtype=np.float64)
    total = np.abs(vector).sum(axis=-1, keepdims=True) if norm == "l1" else np.linalg.norm(vector, axis=-1, keepdims=True)
    return np.divide(vector, total, out=np.zeros_like(vector), where=total > 0)


def to_vectors(counts: np.ndarray, settings: BowSettings, idf: np.ndarray | None) -> np.ndarray:
    """Turn raw counts into the classifier input: TF (counts / total), optional IDF, norm."""
    tf = normalize(counts, "l1")
    if settings.weighting == "tfidf" and idf is not None:
        tf = tf * idf
    return normalize(tf, settings.norm)


# --------------------------------------------------------------------------- classifier


def chi2_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Chi-squared distance ``sum (x - y)^2 / (x + y)`` between every row of ``a`` and ``b``."""
    return -additive_chi2_kernel(a, b)


def chi2_gamma(distances: np.ndarray) -> float:
    """Kernel width = 1 / mean chi-squared distance between training histograms (Zhang et al. 2007)."""
    mean = float(distances.mean())
    return 1.0 / mean if mean > 0 else 1.0


class Chi2Kernel:
    """Chi-squared kernel ``exp(-gamma * chi2(x, y))`` as a picklable callable for SVC."""

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-self.gamma * chi2_distances(a, b))


@dataclass
class Chi2Svm:
    """Chi-squared SVM whose scores become probabilities through Platt (sigmoid) calibration."""

    model: CalibratedClassifierCV
    train_vectors: np.ndarray
    gamma: float
    c: float

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(vectors))


def cross_validate_svm(
    vectors: np.ndarray, labels: np.ndarray, seed: int = 0, c_grid: tuple[float, ...] = SVM_C_GRID
) -> tuple[float, float]:
    """Return (best C, its mean accuracy) with stratified K-fold CV on the given (train) data.

    Gamma is recomputed inside every fold from that fold's training part only.
    """
    labels = np.asarray(labels)
    folds = min(CV_FOLDS, int(np.bincount(labels).min()))
    if folds < 2:
        return c_grid[len(c_grid) // 2], float("nan")
    distances = chi2_distances(vectors, vectors)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {c: [] for c in c_grid}
    for fit_idx, val_idx in splitter.split(vectors, labels):
        fit_d = distances[np.ix_(fit_idx, fit_idx)]
        gamma = chi2_gamma(fit_d)
        val_kernel = np.exp(-gamma * distances[np.ix_(val_idx, fit_idx)])
        for c in c_grid:
            svc = SVC(kernel="precomputed", C=c, class_weight="balanced")
            svc.fit(np.exp(-gamma * fit_d), labels[fit_idx])
            scores[c].append(accuracy_score(labels[val_idx], svc.predict(val_kernel)))
    best = max(c_grid, key=lambda c: (np.mean(scores[c]), -c))
    return best, float(np.mean(scores[best]))


def train_classifier(vectors: np.ndarray, labels: np.ndarray, seed: int = 0, c: float | None = None) -> tuple[Chi2Svm, float]:
    """Fit the chi-squared SVM; C is chosen by cross-validation when not given.

    Returns the classifier and the cross-validation accuracy (NaN when C was given).
    """
    labels = np.asarray(labels)
    cv_accuracy = float("nan")
    if c is None:
        c, cv_accuracy = cross_validate_svm(vectors, labels, seed)
    gamma = chi2_gamma(chi2_distances(vectors, vectors))
    svc = SVC(kernel=Chi2Kernel(gamma), C=c, class_weight="balanced")
    folds = max(2, min(CV_FOLDS, int(np.bincount(labels).min())))
    # ensemble=False: one SVM fitted on all the training data; the sigmoid that turns
    # its scores into probabilities is fitted on cross-validated scores
    calibrated = CalibratedClassifierCV(
        svc, method="sigmoid", ensemble=False, cv=StratifiedKFold(folds, shuffle=True, random_state=seed)
    )
    calibrated.fit(vectors, labels)
    return Chi2Svm(calibrated, np.asarray(vectors), gamma, c), cv_accuracy


# --------------------------------------------------------------------------- model


@dataclass
class WordExemplars:
    """Training patches closest to one visual word, and where they come from."""

    patches: np.ndarray  # (n, PATCH_SIDE, PATCH_SIDE) uint8
    sources: list[str]  # file name of each patch
    classes: list[int]  # class index of each patch
    class_share: np.ndarray  # how typical the word is of each class (frequency per class, normalised)


@dataclass
class BowModel:
    classes: list[str]
    settings: BowSettings
    centers: np.ndarray
    sigma: float  # absolute soft-assignment width
    idf: np.ndarray | None
    classifier: Chi2Svm
    train_paths: list[str]
    train_labels: np.ndarray
    test_paths: list[str]
    test_labels: np.ndarray
    metrics: dict[str, object]
    exemplars: list[WordExemplars]
    k_search: list[tuple[int, float]] = field(default_factory=list)
    version: int = MODEL_VERSION

    @property
    def k(self) -> int:
        return len(self.centers)

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> BowModel:
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict) or payload.get("version") != MODEL_VERSION:
            raise ValueError(f"'{path}' is not a BoW model saved by this version of the app.")
        return cls(**payload)


# --------------------------------------------------------------------------- training


def _check(cancelled: Cancelled) -> None:
    if cancelled():
        raise TaskCancelled()


def _extract_all(
    paths: list[Path], settings: BowSettings, report: Report, cancelled: Cancelled, offset: int, total: int
) -> tuple[list[LocalFeatures], list[np.ndarray]]:
    features, grays = [], []
    for index, path in enumerate(paths):
        _check(cancelled)
        report(offset + index, total, f"Extracting features: {path.parent.name}/{path.name}")
        found, gray = extract_features(load_image(path), settings)
        features.append(found)
        grays.append(np.asarray(gray, dtype=np.uint8))
    return features, grays


def _encode_all(features: list[LocalFeatures], centers: np.ndarray, settings: BowSettings, sigma: float) -> np.ndarray:
    return np.array(
        [encode(f, assign_words(f.descriptors, centers, settings.assignment, sigma), settings.pyramid_levels) for f in features]
    )


def _soft_sigma(descriptors: np.ndarray, centers: np.ndarray, settings: BowSettings) -> float:
    """Absolute soft-assignment width: ``soft_sigma`` x mean distance to the nearest word."""
    sample = descriptors[: min(len(descriptors), 20_000)]
    _, distances = nearest_words(sample, centers)
    mean = float(distances.mean()) if len(distances) else 1.0
    return settings.soft_sigma * (mean if mean > 0 else 1.0)


def _vectors_for(
    features: list[LocalFeatures], centers: np.ndarray, settings: BowSettings, train_descriptors: np.ndarray
) -> tuple[float, np.ndarray | None, np.ndarray]:
    """Soft width, IDF and classifier vectors of the training images for one codebook."""
    sigma = _soft_sigma(train_descriptors, centers, settings)
    counts = _encode_all(features, centers, settings, sigma)
    idf = idf_weights(counts) if settings.weighting == "tfidf" else None
    return sigma, idf, to_vectors(counts, settings, idf)


def choose_k_by_cv(
    features: list[LocalFeatures],
    labels: np.ndarray,
    candidates: list[int],
    settings: BowSettings,
    report: Report = _no_report,
    cancelled: Cancelled = _never,
) -> tuple[int, list[tuple[int, float]]]:
    """Try every K on the training images only and keep the best cross-validated accuracy.

    Ties go to the smaller K (fewer words generalise better). Returns (best K, [(K, accuracy)]).
    """
    descriptors = np.vstack([f.descriptors for f in features])
    results = []
    for index, k in enumerate(candidates):
        _check(cancelled)
        report(index, len(candidates), f"Choosing K by cross-validation: K = {k}")
        if len(descriptors) < k:
            continue
        centers = build_codebook(descriptors, k, settings.seed, settings.max_kmeans_samples)
        _, _, vectors = _vectors_for(features, centers, settings, descriptors)
        _, accuracy = cross_validate_svm(vectors, labels, settings.seed)
        results.append((k, accuracy))
    if not results:
        raise ValueError("No candidate K is smaller than the number of training descriptors.")
    best = max(results, key=lambda item: (item[1], -item[0]))[0]
    return best, results


def _crop_patch(gray: np.ndarray, row: float, col: float, radius: float) -> np.ndarray:
    r = int(round(radius))
    padded = np.pad(gray, r, mode="edge")
    top, left = int(round(row)), int(round(col))
    patch = padded[top : top + 2 * r + 1, left : left + 2 * r + 1]
    return np.asarray(Image.fromarray(patch).resize((PATCH_SIDE, PATCH_SIDE), Image.Resampling.BILINEAR))


def word_exemplars(
    features: list[LocalFeatures],
    grays: list[np.ndarray],
    paths: list[Path],
    labels: np.ndarray,
    centers: np.ndarray,
    n_classes: int,
    per_word: int = EXEMPLARS_PER_WORD,
) -> list[WordExemplars]:
    """For every word, the training patches closest to its centroid and the class share.

    At most one patch per image, so the exemplars show the variety of the word
    instead of the same object several times.
    """
    owners = np.concatenate([np.full(len(f), i) for i, f in enumerate(features)]).astype(int)
    locals_ = np.concatenate([np.arange(len(f)) for f in features]).astype(int)
    descriptors = np.vstack([f.descriptors for f in features])
    words, distances = nearest_words(descriptors, centers)
    descriptor_classes = labels[owners]
    # Classes with more images or keypoints would dominate raw counts, so every class
    # is first divided by its own number of descriptors.
    per_class = np.maximum(np.bincount(descriptor_classes, minlength=n_classes), 1).astype(float)
    result = []
    for word in range(len(centers)):
        members = np.flatnonzero(words == word)
        share = np.bincount(descriptor_classes[members], minlength=n_classes) / per_class
        share = share / share.sum() if share.sum() else share
        patches, sources, classes, used = [], [], [], set()
        for member in members[np.argsort(distances[members])]:
            owner = owners[member]
            if owner in used:
                continue
            used.add(owner)
            f = features[owner]
            point = locals_[member]
            patches.append(_crop_patch(grays[owner], *f.positions[point], f.scales[point]))
            sources.append(paths[owner].name)
            classes.append(int(labels[owner]))
            if len(patches) == per_word:
                break
        stack = np.array(patches, dtype=np.uint8) if patches else np.zeros((0, PATCH_SIDE, PATCH_SIDE), np.uint8)
        result.append(WordExemplars(stack, sources, classes, share))
    return result


def evaluate(classifier: Chi2Svm, vectors: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict[str, object]:
    """Accuracy, confusion matrix (rows = true class) and per-class precision/recall on test."""
    predicted = classifier.predict_proba(vectors).argmax(axis=1)
    report = classification_report(
        labels, predicted, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "confusion": confusion_matrix(labels, predicted, labels=list(range(len(classes)))),
        "per_class": {name: (report[name]["precision"], report[name]["recall"], int(report[name]["support"])) for name in classes},
    }


def train_bow_model(
    folder: str | Path, settings: BowSettings, report: Report = _no_report, cancelled: Cancelled = _never
) -> BowModel:
    """Run the whole BoW pipeline on a folder with one subfolder per class."""
    start = time.perf_counter()
    dataset = list_dataset(folder)
    classes = list(dataset)
    train_p, train_y, test_p, test_y = split_dataset(dataset, settings.test_ratio, settings.seed)
    train_y, test_y = np.array(train_y), np.array(test_y)
    total = len(train_p) + len(test_p)

    train_f, train_g = _extract_all(train_p, settings, report, cancelled, 0, total)
    test_f, _ = _extract_all(test_p, settings, report, cancelled, len(train_p), total)
    train_descriptors = np.vstack([f.descriptors for f in train_f])

    k_search: list[tuple[int, float]] = []
    if settings.auto_k:
        candidates = [m * len(classes) for m in settings.k_multipliers]
        k, k_search = choose_k_by_cv(train_f, train_y, candidates, settings, report, cancelled)
    else:
        k = settings.k_for(len(classes))

    _check(cancelled)
    report(total, total, f"Building the codebook (K-Means, K = {k})")
    centers = build_codebook(train_descriptors, k, settings.seed, settings.max_kmeans_samples)

    _check(cancelled)
    report(total, total, "Encoding histograms and training the chi-squared SVM")
    sigma, idf, train_v = _vectors_for(train_f, centers, settings, train_descriptors)
    classifier, cv_accuracy = train_classifier(train_v, train_y, settings.seed)
    test_v = to_vectors(_encode_all(test_f, centers, settings, sigma), settings, idf)

    _check(cancelled)
    report(total, total, "Evaluating on the test images and collecting word examples")
    metrics = evaluate(classifier, test_v, test_y, classes)
    metrics.update(
        {
            "cv_accuracy": cv_accuracy,
            "train_descriptors": len(train_descriptors),
            "seconds": time.perf_counter() - start,
        }
    )
    exemplars = word_exemplars(train_f, train_g, train_p, train_y, centers, len(classes))
    return BowModel(
        classes=classes,
        settings=settings,
        centers=centers,
        sigma=sigma,
        idf=idf,
        classifier=classifier,
        train_paths=[str(p.resolve()) for p in train_p],
        train_labels=train_y,
        test_paths=[str(p.resolve()) for p in test_p],
        test_labels=test_y,
        metrics=metrics,
        exemplars=exemplars,
        k_search=k_search,
    )


# --------------------------------------------------------------------------- one image


@dataclass
class ImageAnalysis:
    """Histogram of visual words of one image (7.2), ready to be classified (7.3)."""

    gray: Image.Image  # working copy the features refer to
    features: LocalFeatures
    words: np.ndarray  # nearest word of every feature (used for colours)
    word_frequency: np.ndarray  # level-0 histogram, L1 normalised, for the plot
    vector: np.ndarray  # final classifier input (weights, pyramid, normalisation)
    neighbours: list[tuple[str, str, float]]  # (file, class, chi2 distance) of similar train images
    split: str  # "train", "test" or "external" (not part of the dataset, or no file)

    @property
    def in_train(self) -> bool:
        return self.split == "train"


def image_split(model: BowModel, path: str | Path | None) -> str:
    """Whether ``path`` was used to train ``model``, held out for testing, or is new to it."""
    if path is None:
        return "external"
    resolved = str(Path(path).resolve())
    if resolved in set(model.train_paths):
        return "train"
    return "test" if resolved in set(model.test_paths) else "external"


def describe_image(model: BowModel, img: Image.Image, path: str | Path | None = None) -> ImageAnalysis:
    """Extract features, map each one to its visual word and build the histogram."""
    features, gray = extract_features(img, model.settings)
    weights = assign_words(features.descriptors, model.centers, model.settings.assignment, model.sigma)
    counts = encode(features, weights, model.settings.pyramid_levels)
    vector = to_vectors(counts[None, :], model.settings, model.idf)[0]
    words, _ = nearest_words(features.descriptors, model.centers)
    frequency = normalize(weights.sum(axis=0), "l1") if len(weights) else np.zeros(model.k)

    distances = chi2_distances(vector[None, :], model.classifier.train_vectors)[0]
    order = np.argsort(distances)[:NEIGHBOURS]
    neighbours = [
        (model.train_paths[i], model.classes[int(model.train_labels[i])], float(distances[i])) for i in order
    ]
    return ImageAnalysis(gray, features, words, frequency, vector, neighbours, image_split(model, path))


def classify(model: BowModel, analysis: ImageAnalysis) -> list[tuple[str, float]]:
    """Run the trained classifier: (class, probability) sorted from most to least likely."""
    probabilities = model.classifier.predict_proba(analysis.vector[None, :])[0]
    labels = model.classifier.classes_
    ranked = sorted(zip(labels, probabilities), key=lambda item: -item[1])
    return [(model.classes[int(label)], float(p)) for label, p in ranked]


# --------------------------------------------------------------------------- session


_session: dict[str, BowModel] = {}


def set_model(model: BowModel) -> None:
    _session["model"] = model


def get_model(path: str | Path | None = None) -> BowModel:
    """Return the model trained in this session, or load the last saved one."""
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if "model" not in _session:
        if not Path(path).is_file():
            raise ValueError(
                "No BoW model is available yet. Run 'BoW: Build Codebook' first "
                "and choose the dataset folder (one subfolder per class)."
            )
        _session["model"] = BowModel.load(path)
    return _session["model"]


def clear_model() -> None:
    _session.clear()
