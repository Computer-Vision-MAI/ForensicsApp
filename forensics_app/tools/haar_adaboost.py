"""Haar-like feature extraction and Viola-Jones cascade classifier for face detection.

Follows the formulation in 'docs/face_detection_and_recognition.ipynb' and the classic
Viola & Jones (2001) framework:
  1. Integral images for O(1) rectangular region feature evaluation.
  2. Haar-like feature geometries (2-rectangle, 3-rectangle, 4-rectangle).
  3. AdaBoost learning for feature selection and weak classifier weighting.
  4. Attentional cascade of sequential AdaBoost stages with hard-negative mining.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from forensics_app.paths import DATA_DIR, MODELS_DIR

ProgressCallback = Callable[[int, int, str], None]

DATA_ROOT = DATA_DIR / "faces_non_faces"
DEFAULT_TRAIN_POS = DATA_ROOT / "train_data/faces"
DEFAULT_TRAIN_NEG = DATA_ROOT / "train_data/non_faces"
DEFAULT_TEST_POS = DATA_ROOT / "test_data/faces"
DEFAULT_TEST_NEG = DATA_ROOT / "test_data/non_faces"
DEFAULT_CASCADE_PATH = MODELS_DIR / "viola_jones_cascade.json"
CANONICAL_PATCH_SIZE = (19, 19)
# Version 2: stumps use ``polarity * score < polarity * threshold`` with learned
# thresholds. Cascades saved with another version are retrained instead of misread.
CASCADE_FORMAT_VERSION = 2
# Share of held-out faces each stage must accept, and the share of faces held out.
DEFAULT_STAGE_RECALL = 0.995
DEFAULT_VALIDATION_FRACTION = 0.2
# Scores closer than this are treated as equal when placing stump thresholds.
_SCORE_TIE_EPS = 1e-9


def enum(**enums: Any) -> Any:
    """Convenience helper to emulate an enum using a generic class."""
    return type("Enum", (), enums)


FeatureType = enum(
    TWO_VERTICAL=(1, 2),
    TWO_HORIZONTAL=(2, 1),
    THREE_HORIZONTAL=(3, 1),
    THREE_VERTICAL=(1, 3),
    FOUR=(2, 2),
)

FeatureTypes = [
    FeatureType.TWO_VERTICAL,
    FeatureType.TWO_HORIZONTAL,
    FeatureType.THREE_VERTICAL,
    FeatureType.THREE_HORIZONTAL,
    FeatureType.FOUR,
]

FEATURE_TYPE_NAMES = {
    FeatureType.TWO_VERTICAL: "TWO_VERTICAL",
    FeatureType.TWO_HORIZONTAL: "TWO_HORIZONTAL",
    FeatureType.THREE_VERTICAL: "THREE_VERTICAL",
    FeatureType.THREE_HORIZONTAL: "THREE_HORIZONTAL",
    FeatureType.FOUR: "FOUR",
}

NAME_TO_FEATURE_TYPE = {
    "TWO_VERTICAL": FeatureType.TWO_VERTICAL,
    "TWO_HORIZONTAL": FeatureType.TWO_HORIZONTAL,
    "THREE_VERTICAL": FeatureType.THREE_VERTICAL,
    "THREE_HORIZONTAL": FeatureType.THREE_HORIZONTAL,
    "FOUR": FeatureType.FOUR,
}


def compute_integral_image(img_arr: np.ndarray) -> np.ndarray:
    """Calculate the integral image (summed-area table) for constant-time region sums.

    Parameters
    ----------
    img_arr : np.ndarray
        2D single-channel image array.

    Returns
    -------
    np.ndarray
        Integral image with shape (H + 1, W + 1).
    """
    integral = np.zeros((img_arr.shape[0] + 1, img_arr.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(img_arr, axis=0)
    integral = np.cumsum(integral, axis=1)
    return integral


def sum_region(
    integral_img: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
) -> float:
    """Compute the sum of pixel values within a rectangular region in O(1) time.

    Parameters
    ----------
    integral_img : np.ndarray
        2D integral image (H + 1, W + 1).
    top_left : tuple[int, int]
        (row_start, col_start) coordinates.
    bottom_right : tuple[int, int]
        (row_end, col_end) coordinates.

    Returns
    -------
    float
        Sum of pixel intensities in the rectangular region.
    """
    r1, c1 = int(top_left[0]), int(top_left[1])
    r2, c2 = int(bottom_right[0]), int(bottom_right[1])
    a = integral_img[r1, c1]
    b = integral_img[r1, c2]
    c = integral_img[r2, c1]
    d = integral_img[r2, c2]
    return float(d - b - c + a)


class HaarLikeFeature:
    """Class representing a single Haar-like rectangular feature."""

    def __init__(
        self,
        feature_type: Any,
        position: tuple[int, int],
        width: int,
        height: int,
        threshold: float = 0.0,
        polarity: int = 1,
        weight: float = 1.0,
    ) -> None:
        """Create a feature of ``feature_type`` at ``position`` (row, col) with its stump parameters."""
        self.type = feature_type
        # position = (row, col)
        self.top_left = (int(position[0]), int(position[1]))
        # width = span along row axis (height in Cartesian)
        # height = span along col axis (width in Cartesian)
        self.width = int(width)
        self.height = int(height)
        self.bottom_right = (self.top_left[0] + self.width, self.top_left[1] + self.height)
        self.threshold = float(threshold)
        self.polarity = int(polarity)
        self.weight = float(weight)

    def lobes(self) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
        """Return the feature's rectangles as ``(sign, top_left, bottom_right)`` in (row, col).

        The score is the signed sum of the rectangles' pixel sums. Scoring and drawing
        both use this single definition.
        """
        tl = self.top_left
        br = self.bottom_right
        w = self.width
        h = self.height

        if self.type == FeatureType.TWO_VERTICAL:
            return [
                (1, tl, (tl[0] + w, tl[1] + h // 2)),
                (-1, (tl[0], tl[1] + h // 2), br),
            ]
        if self.type == FeatureType.TWO_HORIZONTAL:
            return [
                (1, tl, (tl[0] + w // 2, tl[1] + h)),
                (-1, (tl[0] + w // 2, tl[1]), br),
            ]
        if self.type == FeatureType.THREE_HORIZONTAL:
            return [
                (1, tl, (tl[0] + w // 3, tl[1] + h)),
                (-1, (tl[0] + w // 3, tl[1]), (tl[0] + 2 * w // 3, tl[1] + h)),
                (1, (tl[0] + 2 * w // 3, tl[1]), br),
            ]
        if self.type == FeatureType.THREE_VERTICAL:
            return [
                (1, tl, (br[0], tl[1] + h // 3)),
                (-1, (tl[0], tl[1] + h // 3), (br[0], tl[1] + 2 * h // 3)),
                (1, (tl[0], tl[1] + 2 * h // 3), br),
            ]
        if self.type == FeatureType.FOUR:
            return [
                (1, tl, (tl[0] + w // 2, tl[1] + h // 2)),
                (-1, (tl[0] + w // 2, tl[1]), (br[0], tl[1] + h // 2)),
                (-1, (tl[0], tl[1] + h // 2), (tl[0] + w // 2, br[1])),
                (1, (tl[0] + w // 2, tl[1] + h // 2), br),
            ]
        return []

    def get_score(self, int_img: np.ndarray) -> float:
        """Compute the raw feature response (sum of + lobes minus sum of - lobes)."""
        score = 0.0
        for sign, top_left, bottom_right in self.lobes():
            score += sign * sum_region(int_img, top_left, bottom_right)
        return score

    def predict(self, int_img: np.ndarray) -> int:
        """Return the unweighted decision-stump prediction (+1 face, -1 non-face).

        h(x) = +1 if polarity * score < polarity * threshold else -1, so polarity +1
        votes "face" for scores below the threshold and polarity -1 for scores above it.
        """
        score = self.get_score(int_img)
        return 1 if self.polarity * score < self.polarity * self.threshold else -1

    def get_vote(self, int_img: np.ndarray) -> float:
        """Return the weighted vote of this weak classifier (+weight or -weight).

        The stump prediction multiplied by the AdaBoost classifier weight alpha.
        """
        return self.weight * float(self.predict(int_img))

    def with_stump(self, threshold: float, polarity: int, weight: float) -> HaarLikeFeature:
        """Return a copy of this feature with the given stump threshold, polarity and weight."""
        clone = copy.copy(self)
        clone.threshold = float(threshold)
        clone.polarity = int(polarity)
        clone.weight = float(weight)
        return clone

    def to_dict(self) -> dict[str, Any]:
        """Serialize feature to dictionary."""
        return {
            "type": FEATURE_TYPE_NAMES[self.type],
            "position": list(self.top_left),
            "width": self.width,
            "height": self.height,
            "threshold": self.threshold,
            "polarity": self.polarity,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HaarLikeFeature:
        """Deserialize feature from dictionary."""
        ft = NAME_TO_FEATURE_TYPE[data["type"]]
        return cls(
            feature_type=ft,
            position=tuple(data["position"]),
            width=int(data["width"]),
            height=int(data["height"]),
            threshold=float(data["threshold"]),
            polarity=int(data["polarity"]),
            weight=float(data["weight"]),
        )


class CascadeStage:
    """A single strong classifier stage in the Viola-Jones attentional cascade."""

    def __init__(self, classifiers: list[HaarLikeFeature], threshold: float = 0.0) -> None:
        """Create a stage from its weak classifiers and its acceptance threshold."""
        self.classifiers = classifiers
        self.threshold = threshold

    def evaluate(self, int_img: np.ndarray) -> tuple[bool, float]:
        """Evaluate integral image on this stage.

        Returns (passed, total_score).
        """
        votes = [c.get_vote(int_img) for c in self.classifiers]
        score = float(sum(votes))
        return (score > self.threshold), score

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stage to a dictionary."""
        return {
            "threshold": self.threshold,
            "classifiers": [c.to_dict() for c in self.classifiers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CascadeStage:
        """Deserialize a stage from a dictionary."""
        return cls(
            threshold=float(data.get("threshold", 0.0)),
            classifiers=[HaarLikeFeature.from_dict(item) for item in data["classifiers"]],
        )


class ViolaJonesCascade:
    """Full Viola-Jones attentional cascade consisting of multiple sequential AdaBoost stages."""

    def __init__(self, stages: list[CascadeStage]) -> None:
        """Create a cascade from its stages, evaluated in order."""
        self.stages = stages

    def predict(self, int_img: np.ndarray) -> tuple[bool, int, list[float]]:
        """Run candidate window through the cascade.

        Returns
        -------
        tuple[bool, int, list[float]]
            (is_face, stages_passed, stage_scores)
            If rejected at stage k, is_face is False and stages_passed is k.
        """
        stage_scores: list[float] = []
        for stage_idx, stage in enumerate(self.stages):
            passed, score = stage.evaluate(int_img)
            stage_scores.append(score)
            if not passed:
                return False, stage_idx, stage_scores
        return True, len(self.stages), stage_scores

    def get_all_classifiers(self) -> list[HaarLikeFeature]:
        """Return a flattened list of all classifiers across all stages."""
        result: list[HaarLikeFeature] = []
        for stage in self.stages:
            result.extend(stage.classifiers)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cascade to a dictionary."""
        return {
            "format_version": CASCADE_FORMAT_VERSION,
            "patch_size": list(CANONICAL_PATCH_SIZE),
            "stages": [s.to_dict() for s in self.stages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViolaJonesCascade:
        """Deserialize a cascade from a dictionary."""
        stages = [CascadeStage.from_dict(s) for s in data["stages"]]
        return cls(stages)


def ensemble_vote(
    int_img: np.ndarray, classifiers: list[HaarLikeFeature]
) -> tuple[bool, float, list[float]]:
    """Evaluate a flat list of weak classifiers on an integral image."""
    votes = [float(c.get_vote(int_img)) for c in classifiers]
    total_score = float(sum(votes))
    return (total_score > 0.0), total_score, votes


def create_features(
    img_height: int = 19,
    img_width: int = 19,
    min_w: int = 6,
    max_w: int = 12,
    min_h: int = 6,
    max_h: int = 12,
    step: int = 2,
) -> list[HaarLikeFeature]:
    """Generate candidate Haar-like features spanning all valid positions across the face.

    One candidate per geometry: AdaBoost learns each stump's threshold and polarity.
    """
    features: list[HaarLikeFeature] = []
    for feature in FeatureTypes:
        fsw = max(min_w, feature[0])
        for fw in range(fsw, max_w + 1, feature[0] * step):
            fsh = max(min_h, feature[1])
            for fh in range(fsh, max_h + 1, feature[1] * step):
                # Ensure the feature fits entirely within image boundaries
                for r in range(0, img_height - fw + 1, step):
                    for c in range(0, img_width - fh + 1, step):
                        features.append(HaarLikeFeature(feature, (r, c), fw, fh))
    return features


def patch_to_normalized_integral(
    image: Image.Image, patch_size: tuple[int, int] = CANONICAL_PATCH_SIZE
) -> np.ndarray:
    """Turn an image into the normalized integral image the cascade works on.

    Shared by training and inference so both see identical inputs: convert to
    grayscale, resize to the canonical window (a no-op when it already matches),
    normalize intensities by their maximum, then normalize the integral image.
    """
    gray = image.convert("L")
    if gray.size != patch_size:
        gray = gray.resize(patch_size, Image.Resampling.BILINEAR)
    arr = np.array(gray, dtype=np.float64)
    arr_norm = arr / (arr.max() + 1e-6)
    ii = compute_integral_image(arr_norm)
    return ii / (ii.max() + 1e-6)


def load_dataset_integral_images(folder: Path, max_samples: int | None = None) -> list[np.ndarray]:
    """Load 19x19 grayscale images from data/faces_non_faces and compute normalized integral images."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted([f for f in folder.glob("*.png")])
    if max_samples is not None:
        files = files[:max_samples]

    integral_images: list[np.ndarray] = []
    for file_path in files:
        with Image.open(file_path) as img:
            integral_images.append(patch_to_normalized_integral(img))

    return integral_images


def feature_scores(
    integral_images: list[np.ndarray], features: list[HaarLikeFeature]
) -> np.ndarray:
    """Return the raw score of every feature on every image, shape (n_images, n_features).

    Computed once and reused by every AdaBoost round and every cascade stage.
    """
    scores = np.empty((len(integral_images), len(features)), dtype=np.float64)
    for i, int_img in enumerate(integral_images):
        scores[i, :] = [f.get_score(int_img) for f in features]
    return scores


@dataclass(frozen=True)
class WeakClassifierChoice:
    """One AdaBoost round's result: which feature, its learned stump, and its weight."""

    feature_index: int
    threshold: float
    polarity: int
    alpha: float


def _stump_predictions(scores: np.ndarray, threshold: float, polarity: int) -> np.ndarray:
    """Apply one learned stump to a column of scores (+1 face / -1 non-face)."""
    return np.where(polarity * scores < polarity * threshold, 1.0, -1.0)


def _select_weak_classifiers(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classifiers: int,
    candidate_indexes: Iterable[int] | None = None,
) -> list[WeakClassifierChoice]:
    """Run discrete AdaBoost, learning the optimal stump of every feature in every round.

    For each feature the samples are sorted by score once; each round the weighted
    error of every possible threshold (midway between consecutive distinct scores, or
    beyond the largest one) and both polarities comes from cumulative sums of the
    positive and negative sample weights (Viola & Jones 2001, section 3).
    """
    columns = np.arange(scores.shape[1]) if candidate_indexes is None else np.asarray(list(candidate_indexes))
    cand_scores = scores[:, columns]
    order = np.argsort(cand_scores, axis=0, kind="stable")
    sorted_scores = np.take_along_axis(cand_scores, order, axis=0)

    # Candidate thresholds: split after sorted sample i (last row: everything below)
    thresholds = np.empty_like(sorted_scores)
    thresholds[:-1] = (sorted_scores[:-1] + sorted_scores[1:]) / 2.0
    thresholds[-1] = sorted_scores[-1] + 1.0
    valid_split = np.ones(sorted_scores.shape, dtype=bool)
    valid_split[:-1] = (sorted_scores[1:] - sorted_scores[:-1]) > _SCORE_TIE_EPS

    is_pos = labels > 0
    is_pos_sorted = is_pos[order]
    num_pos = int(np.sum(is_pos))
    num_neg = len(labels) - num_pos
    # Sample weights initialization: 1/(2*N_pos) for positives, 1/(2*N_neg) for negatives
    weights = np.where(is_pos, 1.0 / (2 * num_pos), 1.0 / (2 * num_neg))
    available = np.ones(len(columns), dtype=bool)
    selected: list[WeakClassifierChoice] = []

    for _ in range(num_classifiers):
        if not available.any():
            break

        weights *= 1.0 / np.sum(weights)
        total_pos = float(np.sum(weights[is_pos]))
        total_neg = float(np.sum(weights[~is_pos]))

        sorted_weights = weights[order]
        below_pos = np.cumsum(np.where(is_pos_sorted, sorted_weights, 0.0), axis=0)
        below_neg = np.cumsum(np.where(is_pos_sorted, 0.0, sorted_weights), axis=0)
        # polarity +1: scores below the threshold are faces; polarity -1: above
        err_plus = below_neg + (total_pos - below_pos)
        err_minus = below_pos + (total_neg - below_neg)
        errors = np.minimum(err_plus, err_minus)
        errors[~valid_split] = np.inf
        errors[:, ~available] = np.inf

        row, col = np.unravel_index(int(np.argmin(errors)), errors.shape)
        best_error = float(max(errors[row, col], 1e-6))
        if best_error >= 0.5:
            break

        polarity = 1 if err_plus[row, col] <= err_minus[row, col] else -1
        threshold = float(thresholds[row, col])
        # Calculate classifier weight: alpha_t = 0.5 * ln((1 - error) / error)
        alpha = float(0.5 * np.log((1.0 - best_error) / best_error))
        selected.append(WeakClassifierChoice(int(columns[col]), threshold, polarity, alpha))

        # Update sample weights: w_{t+1} = w_t * exp(-alpha * label * prediction)
        predictions = _stump_predictions(cand_scores[:, col], threshold, polarity)
        weights = weights * np.exp(-alpha * labels * predictions)
        available[col] = False

    return selected


def _build_classifiers(
    features: list[HaarLikeFeature], choices: list[WeakClassifierChoice]
) -> list[HaarLikeFeature]:
    """Turn AdaBoost choices into weighted copies of the candidate features."""
    return [
        features[c.feature_index].with_stump(c.threshold, c.polarity, c.alpha) for c in choices
    ]


def _stage_votes(scores: np.ndarray, choices: list[WeakClassifierChoice]) -> np.ndarray:
    """Return a stage's summed weighted votes for every row of a score matrix."""
    votes = np.zeros(scores.shape[0], dtype=np.float64)
    for c in choices:
        votes += c.alpha * _stump_predictions(scores[:, c.feature_index], c.threshold, c.polarity)
    return votes


def learn_adaboost(
    positive_iis: list[np.ndarray],
    negative_iis: list[np.ndarray],
    num_classifiers: int = 10,
    features: list[HaarLikeFeature] | None = None,
    img_height: int = 19,
    img_width: int = 19,
) -> list[HaarLikeFeature]:
    """Train AdaBoost ensemble selecting the best weak classifiers from candidate features.

    Each selected classifier is a copy of a candidate with a learned threshold,
    polarity and weight; the candidate ``features`` are never modified.
    """
    if features is None:
        features = create_features(img_height, img_width)

    labels = np.hstack((np.ones(len(positive_iis)), np.ones(len(negative_iis)) * -1.0))
    scores = feature_scores(positive_iis + negative_iis, features)
    return _build_classifiers(features, _select_weak_classifiers(scores, labels, num_classifiers))


def train_viola_jones_cascade(
    stage_configs: list[int] = [2, 5, 10],
    max_pos: int | None = 300,
    max_neg: int | None = 500,
    train_pos_dir: Path = DEFAULT_TRAIN_POS,
    train_neg_dir: Path = DEFAULT_TRAIN_NEG,
    progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    stage_recall: float = DEFAULT_STAGE_RECALL,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> ViolaJonesCascade:
    """Train a multi-stage Viola-Jones cascade on data/faces_non_faces using hard negative mining.

    Feature scores are computed once and reused by all stages; every stage learns its
    own stump thresholds and polarities on the faces plus the negatives that survived
    the previous stages. A random ``validation_fraction`` of the faces is held out of
    AdaBoost and used to set each stage threshold so that ``stage_recall`` of unseen
    faces pass it (thresholds set on the training faces themselves overfit).
    ``max_pos`` / ``max_neg`` of None use every image in the folder.
    ``progress(done, total, label)`` is called between steps, and when
    ``should_cancel()`` returns True training stops with ``TaskCancelled``.
    """
    from .dialogs import TaskCancelled  # local import: keeps Tk out of module import time

    def report(done: int, label: str) -> None:
        """Stop if cancellation was requested, otherwise forward progress to the caller."""
        if should_cancel is not None and should_cancel():
            raise TaskCancelled()
        if progress is not None:
            progress(done, len(stage_configs) + 1, label)

    report(0, "Loading training patches and computing Haar feature responses…")
    pos_iis = load_dataset_integral_images(train_pos_dir, max_samples=max_pos)
    all_neg_iis = load_dataset_integral_images(train_neg_dir, max_samples=max_neg)

    candidate_features = create_features(19, 19, step=2)
    all_pos_scores = feature_scores(pos_iis, candidate_features)
    neg_scores = feature_scores(all_neg_iis, candidate_features)

    # Hold out faces for threshold calibration (fixed seed: reproducible training)
    num_val = int(round(len(pos_iis) * validation_fraction)) if len(pos_iis) >= 10 else 0
    perm = np.random.default_rng(0).permutation(len(pos_iis))
    pos_scores = all_pos_scores[perm[num_val:]]
    calibration_scores = all_pos_scores[perm[:num_val]] if num_val else pos_scores
    num_pos = len(pos_scores)

    stages: list[CascadeStage] = []
    current_negs = np.arange(len(all_neg_iis))  # indexes into all_neg_iis

    for stage_idx, num_clfs in enumerate(stage_configs):
        report(stage_idx + 1, f"Training stage {stage_idx + 1} of {len(stage_configs)} ({num_clfs} classifiers)…")
        if len(current_negs) < 10:
            # If negatives were almost entirely eliminated, resample from pool
            current_negs = np.arange(min(100, len(all_neg_iis)))

        scores = np.vstack((pos_scores, neg_scores[current_negs]))
        labels = np.hstack((np.ones(num_pos), np.ones(len(current_negs)) * -1.0))
        choices = _select_weak_classifiers(scores, labels, num_clfs)
        stage_clfs = _build_classifiers(candidate_features, choices)

        # Calibrate stage threshold: ``stage_recall`` of the held-out faces pass this stage
        calibrated_th = (
            float(np.percentile(_stage_votes(calibration_scores, choices), 100.0 * (1.0 - stage_recall))) - 1e-4
        )
        stages.append(CascadeStage(classifiers=stage_clfs, threshold=calibrated_th))

        # Hard negative mining: keep negatives that survive this stage (false positives)
        neg_votes = _stage_votes(neg_scores[current_negs], choices)
        surviving_negs = current_negs[neg_votes > calibrated_th]
        if len(surviving_negs) >= 15:
            current_negs = surviving_negs

    return ViolaJonesCascade(stages)


def save_haar_model(classifiers: list[HaarLikeFeature], path: str | Path) -> None:
    """Save a list of HaarLikeFeatures to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in classifiers], f, indent=2)


def load_haar_model(path: str | Path) -> list[HaarLikeFeature]:
    """Load a list of HaarLikeFeatures from a JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [HaarLikeFeature.from_dict(item) for item in data]


def save_cascade_model(cascade: ViolaJonesCascade, path: Path = DEFAULT_CASCADE_PATH) -> None:
    """Save ViolaJonesCascade model to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cascade.to_dict(), f, indent=2)


def load_cascade_model(path: Path = DEFAULT_CASCADE_PATH) -> ViolaJonesCascade | None:
    """Load ViolaJonesCascade model from JSON.

    Returns None when the file is missing, unreadable, or was saved with an older
    stump format (so callers retrain it instead of misreading its thresholds).
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format_version") != CASCADE_FORMAT_VERSION:
            return None
        return ViolaJonesCascade.from_dict(data)
    except Exception:
        return None


def ensure_default_cascade_model(
    path: Path = DEFAULT_CASCADE_PATH,
    progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ViolaJonesCascade:
    """Return pre-trained cascade model; trains on all of data/faces_non_faces if not present.

    A cascade saved in an older format is treated as missing and retrained.
    """
    cascade = load_cascade_model(path)
    if cascade is not None and len(cascade.stages) >= 1:
        return cascade

    cascade = train_viola_jones_cascade(
        stage_configs=[2, 5, 10],
        max_pos=None,  # every training face
        max_neg=None,  # and every training non-face
        progress=progress,
        should_cancel=should_cancel,
    )
    save_cascade_model(cascade, path)
    return cascade


def ensure_default_haar_model(
    model_path: str | Path = MODELS_DIR / "haar_adaboost_model.json",
) -> list[HaarLikeFeature]:
    """Compatibility wrapper returning flattened list of classifiers from the cascade."""
    cascade = ensure_default_cascade_model(Path(model_path).parent / "viola_jones_cascade.json")
    return cascade.get_all_classifiers()


def draw_haar_overlay(
    image: Image.Image | np.ndarray,
    classifiers: list[HaarLikeFeature],
    alpha: float = 0.5,
    patch_size: tuple[int, int] = CANONICAL_PATCH_SIZE,
    target_box: tuple[int, int, int, int] | None = None,
    max_features: int = 10,
    draw_box: bool = True,
    box_label: str | None = None,
) -> Image.Image:
    """Superimpose selected Haar-like features onto an image with alpha transparency.

    Accurately maps the 19x19 feature geometry to either the whole image or a face bounding box.
    Each lobe is coloured by what the learned stump expects on a face: green where the
    area should be lighter, crimson red where it should be darker (sign x polarity).
    Deduplicates repeated filter geometries so only unique features are shown, up to max_features.
    Optionally draws a proportional bounding box and label around confirmed faces.
    """
    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
        img_np = np.array(pil_img)
    else:
        if image.ndim == 2:
            img_np = np.stack([image] * 3, axis=-1)
        else:
            img_np = image.copy()
        pil_img = Image.fromarray(img_np.astype(np.uint8))

    h_img, w_img = img_np.shape[:2]
    import cv2

    overlay = img_np.copy()

    # Determine region of interest (whole image or detected face box)
    if target_box is not None:
        bx, by, bw, bh = target_box
        bx1 = max(0, bx)
        by1 = max(0, by)
        bw = max(10, min(w_img - bx1, bw))
        bh = max(10, min(h_img - by1, bh))
    else:
        bx1, by1 = 0, 0
        bw, bh = w_img, h_img

    scale_y = bh / float(patch_size[0])
    scale_x = bw / float(patch_size[1])

    # Deduplicate classifiers by feature geometry: (type, top_left, width, height)
    # Sort by absolute classifier weight descending (most discriminative first)
    seen_signatures: set[tuple[Any, tuple[int, int], int, int]] = set()
    unique_features: list[HaarLikeFeature] = []
    for feat in sorted(classifiers, key=lambda c: abs(c.weight), reverse=True):
        sig = (feat.type, tuple(feat.top_left), feat.width, feat.height)
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_features.append(feat)
        if len(unique_features) >= max_features:
            break

    # Lobe colors: Light Green (lighter area expected on a face) vs Crimson Red (darker area)
    light_color = (60, 230, 60)
    dark_color = (230, 50, 50)

    for feat in unique_features:
        # A face vote needs polarity * score below polarity * threshold, i.e. lobes whose
        # sign matches the polarity are the darker ones.
        for sign, (r1, c1), (r2, c2) in feat.lobes():
            x1 = bx1 + int(round(c1 * scale_x))
            y1 = by1 + int(round(r1 * scale_y))
            x2 = bx1 + int(round(c2 * scale_x))
            y2 = by1 + int(round(r2 * scale_y))
            color = dark_color if sign * feat.polarity > 0 else light_color
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Alpha blend: 0.5 * overlay + 0.5 * original
    blended = cv2.addWeighted(overlay, float(alpha), img_np, float(1.0 - alpha), 0.0)

    # Draw solid proportional bounding box and label if requested
    if draw_box:
        diag = math.hypot(bw, bh)
        line_thickness = max(1, int(round(diag / 140.0)))
        font_scale = max(0.35, min(1.6, diag / 350.0))
        font_thickness = max(1, int(round(font_scale * 1.8)))

        x_b1 = max(0, bx1)
        y_b1 = max(0, by1)
        x_b2 = min(w_img - 1, bx1 + bw)
        y_b2 = min(h_img - 1, by1 + bh)

        # Draw green bounding box (RGB: 0, 255, 0)
        cv2.rectangle(blended, (x_b1, y_b1), (x_b2, y_b2), (0, 255, 0), line_thickness)

        if box_label and bw >= 40 and bh >= 40:
            text_y = y_b1 - 6 if y_b1 > 20 else y_b1 + int(18 * font_scale) + 4
            cv2.putText(
                blended,
                box_label,
                (x_b1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                font_thickness,
            )

    return Image.fromarray(blended.astype(np.uint8))
