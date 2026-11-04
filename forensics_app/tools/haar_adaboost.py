"""Haar-like feature extraction and Viola-Jones cascade classifier for face detection.

Follows the formulation in 'docs/face_detection_and_recognition.ipynb' and the classic
Viola & Jones (2001) framework:
  1. Integral images for O(1) rectangular region feature evaluation.
  2. Haar-like feature geometries (2-rectangle, 3-rectangle, 4-rectangle).
  3. AdaBoost learning for feature selection and weak classifier weighting.
  4. Attentional cascade of sequential AdaBoost stages with hard-negative mining.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

DATA_ROOT = Path("data/faces_non_faces")
DEFAULT_TRAIN_POS = DATA_ROOT / "train_data/faces"
DEFAULT_TRAIN_NEG = DATA_ROOT / "train_data/non_faces"
DEFAULT_TEST_POS = DATA_ROOT / "test_data/faces"
DEFAULT_TEST_NEG = DATA_ROOT / "test_data/non_faces"
DEFAULT_CASCADE_PATH = Path("models/viola_jones_cascade.json")
CANONICAL_PATCH_SIZE = (19, 19)


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

    def get_score(self, int_img: np.ndarray) -> float:
        """Compute the raw feature response (sum of positive lobes - sum of negative lobes)."""
        tl = self.top_left
        br = self.bottom_right
        w = self.width
        h = self.height

        if self.type == FeatureType.TWO_VERTICAL:
            first = sum_region(int_img, tl, (tl[0] + w, tl[1] + h // 2))
            second = sum_region(int_img, (tl[0], tl[1] + h // 2), br)
            return first - second

        if self.type == FeatureType.TWO_HORIZONTAL:
            first = sum_region(int_img, tl, (tl[0] + w // 2, tl[1] + h))
            second = sum_region(int_img, (tl[0] + w // 2, tl[1]), br)
            return first - second

        if self.type == FeatureType.THREE_HORIZONTAL:
            first = sum_region(int_img, tl, (tl[0] + w // 3, tl[1] + h))
            second = sum_region(int_img, (tl[0] + w // 3, tl[1]), (tl[0] + 2 * w // 3, tl[1] + h))
            third = sum_region(int_img, (tl[0] + 2 * w // 3, tl[1]), br)
            return first - second + third

        if self.type == FeatureType.THREE_VERTICAL:
            first = sum_region(int_img, tl, (br[0], tl[1] + h // 3))
            second = sum_region(int_img, (tl[0], tl[1] + h // 3), (br[0], tl[1] + 2 * h // 3))
            third = sum_region(int_img, (tl[0], tl[1] + 2 * h // 3), br)
            return first - second + third

        if self.type == FeatureType.FOUR:
            first = sum_region(int_img, tl, (tl[0] + w // 2, tl[1] + h // 2))
            second = sum_region(int_img, (tl[0] + w // 2, tl[1]), (br[0], tl[1] + h // 2))
            third = sum_region(int_img, (tl[0], tl[1] + h // 2), (tl[0] + w // 2, br[1]))
            fourth = sum_region(int_img, (tl[0] + w // 2, tl[1] + h // 2), br)
            return first - second - third + fourth

        return 0.0

    def get_vote(self, int_img: np.ndarray) -> float:
        """Return the weighted vote of this weak classifier (+weight or -weight).

        In Viola-Jones decision stumps:
          h(x) = +1 if score < polarity * threshold else -1
        Multiplied by the AdaBoost classifier weight alpha.
        """
        score = self.get_score(int_img)
        is_positive = score < (self.polarity * self.threshold)
        return self.weight * (1.0 if is_positive else -1.0)

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
        return {
            "threshold": self.threshold,
            "classifiers": [c.to_dict() for c in self.classifiers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CascadeStage:
        return cls(
            threshold=float(data.get("threshold", 0.0)),
            classifiers=[HaarLikeFeature.from_dict(item) for item in data["classifiers"]],
        )


class ViolaJonesCascade:
    """Full Viola-Jones attentional cascade consisting of multiple sequential AdaBoost stages."""

    def __init__(self, stages: list[CascadeStage]) -> None:
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
        return {
            "patch_size": list(CANONICAL_PATCH_SIZE),
            "stages": [s.to_dict() for s in self.stages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViolaJonesCascade:
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
    """Generate candidate Haar-like features spanning all valid positions across the face."""
    features: list[HaarLikeFeature] = []
    for feature in FeatureTypes:
        fsw = max(min_w, feature[0])
        for fw in range(fsw, max_w + 1, feature[0] * step):
            fsh = max(min_h, feature[1])
            for fh in range(fsh, max_h + 1, feature[1] * step):
                # Ensure the feature fits entirely within image boundaries
                for r in range(0, img_height - fw + 1, step):
                    for c in range(0, img_width - fh + 1, step):
                        # Both polarities: polarity +1 and polarity -1
                        features.append(HaarLikeFeature(feature, (r, c), fw, fh, 0.0, 1))
                        features.append(HaarLikeFeature(feature, (r, c), fw, fh, 0.0, -1))
    return features


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
            arr = np.array(img.convert("L"), dtype=np.float64)
            # Normalize pixel intensities by maximum value
            arr_norm = arr / (arr.max() + 1e-6)
            ii = compute_integral_image(arr_norm)
            # Normalize integral image values as in notebook
            ii_norm = ii / (ii.max() + 1e-6)
            integral_images.append(ii_norm)

    return integral_images


def learn_adaboost(
    positive_iis: list[np.ndarray],
    negative_iis: list[np.ndarray],
    num_classifiers: int = 10,
    features: list[HaarLikeFeature] | None = None,
    img_height: int = 19,
    img_width: int = 19,
) -> list[HaarLikeFeature]:
    """Train AdaBoost ensemble selecting the best weak classifiers from candidate features."""
    if features is None:
        features = create_features(img_height, img_width)

    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg
    images = positive_iis + negative_iis
    labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1.0))

    # Pre-calculate vote matrix: (num_imgs, num_features)
    num_features = len(features)
    votes = np.zeros((num_imgs, num_features))
    for i in range(num_imgs):
        votes[i, :] = np.array([f.get_vote(images[i]) for f in features])

    # Sample weights initialization: 1/(2*N_pos) for positives, 1/(2*N_neg) for negatives
    weights = np.hstack((np.ones(num_pos) / (2 * num_pos), np.ones(num_neg) / (2 * num_neg)))
    feature_indexes = list(range(num_features))
    selected_classifiers: list[HaarLikeFeature] = []

    for _ in range(num_classifiers):
        if not feature_indexes:
            break

        weights *= 1.0 / np.sum(weights)

        # Vectorized weighted error computation
        errors = np.zeros(len(feature_indexes))
        for p, idx in enumerate(feature_indexes):
            errors[p] = np.sum(weights[labels != votes[:, idx]])

        min_pos = int(np.argmin(errors))
        best_error = float(max(errors[min_pos], 1e-6))
        if best_error >= 0.5:
            break

        best_feat_idx = feature_indexes[min_pos]
        best_feature = features[best_feat_idx]

        # Calculate classifier weight: alpha_t = 0.5 * ln((1 - error) / error)
        alpha = 0.5 * np.log((1.0 - best_error) / best_error)
        best_feature.weight = float(alpha)
        selected_classifiers.append(best_feature)

        # Update sample weights: w_{t+1} = w_t * exp(-alpha * label * vote)
        weights = weights * np.exp(-alpha * labels * votes[:, best_feat_idx])
        feature_indexes.pop(min_pos)

    return selected_classifiers


def train_viola_jones_cascade(
    stage_configs: list[int] = [2, 5, 10],
    max_pos: int = 300,
    max_neg: int = 500,
    train_pos_dir: Path = DEFAULT_TRAIN_POS,
    train_neg_dir: Path = DEFAULT_TRAIN_NEG,
) -> ViolaJonesCascade:
    """Train a multi-stage Viola-Jones cascade on data/faces_non_faces using hard negative mining."""
    pos_iis = load_dataset_integral_images(train_pos_dir, max_samples=max_pos)
    all_neg_iis = load_dataset_integral_images(train_neg_dir, max_samples=max_neg)

    candidate_features = create_features(19, 19, step=2)
    stages: list[CascadeStage] = []
    current_negs = all_neg_iis

    for stage_idx, num_clfs in enumerate(stage_configs):
        if len(current_negs) < 10:
            # If negatives were almost entirely eliminated, resample from pool
            current_negs = all_neg_iis[: min(100, len(all_neg_iis))]

        stage_clfs = learn_adaboost(
            positive_iis=pos_iis,
            negative_iis=current_negs,
            num_classifiers=num_clfs,
            features=candidate_features,
            img_height=19,
            img_width=19,
        )
        # Calibrate stage threshold: ensure ~98.5% of positive training faces pass this stage
        pos_scores = [sum(c.get_vote(pos_ii) for c in stage_clfs) for pos_ii in pos_iis]
        calibrated_th = float(np.percentile(pos_scores, 1.5)) - 1e-4
        stage = CascadeStage(classifiers=stage_clfs, threshold=calibrated_th)
        stages.append(stage)

        # Hard negative mining: filter negatives that survive this stage (false positives)
        surviving_negs: list[np.ndarray] = []
        for neg_ii in current_negs:
            passed, _ = stage.evaluate(neg_ii)
            if passed:
                surviving_negs.append(neg_ii)

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
    """Load ViolaJonesCascade model from JSON."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ViolaJonesCascade.from_dict(data)
    except Exception:
        return None


def ensure_default_cascade_model(path: Path = DEFAULT_CASCADE_PATH) -> ViolaJonesCascade:
    """Return pre-trained cascade model; trains on data/faces_non_faces if not present."""
    cascade = load_cascade_model(path)
    if cascade is not None and len(cascade.stages) >= 1:
        return cascade

    cascade = train_viola_jones_cascade(stage_configs=[2, 5, 10], max_pos=250, max_neg=400)
    save_cascade_model(cascade, path)
    return cascade


def ensure_default_haar_model(model_path: str | Path = "models/haar_adaboost_model.json") -> list[HaarLikeFeature]:
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
    Positive lobes (where light intensity is expected for face) are drawn in green,
    and negative lobes (darker intensity) in crimson red.
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

    for feat in unique_features:
        # Map feature coordinates (top_left is (row, col))
        r1 = feat.top_left[0]
        c1 = feat.top_left[1]
        w_span = feat.width  # delta row
        h_span = feat.height  # delta col

        x1 = bx1 + int(round(c1 * scale_x))
        y1 = by1 + int(round(r1 * scale_y))
        x2 = bx1 + int(round((c1 + h_span) * scale_x))
        y2 = by1 + int(round((r1 + w_span) * scale_y))

        dx = x2 - x1
        dy = y2 - y1

        # Lobe colors: Light Green (positive lobe) vs Crimson Red (negative lobe)
        pos_color = (60, 230, 60)
        neg_color = (230, 50, 50)

        if feat.type == FeatureType.TWO_VERTICAL:
            mid_x = x1 + dx // 2
            # Polarity dictates which half is positive
            c_left = pos_color if feat.polarity == -1 else neg_color
            c_right = neg_color if feat.polarity == -1 else pos_color
            cv2.rectangle(overlay, (x1, y1), (mid_x, y2), c_left, -1)
            cv2.rectangle(overlay, (mid_x, y1), (x2, y2), c_right, -1)

        elif feat.type == FeatureType.TWO_HORIZONTAL:
            mid_y = y1 + dy // 2
            c_top = pos_color if feat.polarity == -1 else neg_color
            c_bot = neg_color if feat.polarity == -1 else pos_color
            cv2.rectangle(overlay, (x1, y1), (x2, mid_y), c_top, -1)
            cv2.rectangle(overlay, (x1, mid_y), (x2, y2), c_bot, -1)

        elif feat.type == FeatureType.THREE_HORIZONTAL:
            y_a = y1 + dy // 3
            y_b = y1 + 2 * dy // 3
            cv2.rectangle(overlay, (x1, y1), (x2, y_a), pos_color, -1)
            cv2.rectangle(overlay, (x1, y_a), (x2, y_b), neg_color, -1)
            cv2.rectangle(overlay, (x1, y_b), (x2, y2), pos_color, -1)

        elif feat.type == FeatureType.THREE_VERTICAL:
            x_a = x1 + dx // 3
            x_b = x1 + 2 * dx // 3
            cv2.rectangle(overlay, (x1, y1), (x_a, y2), pos_color, -1)
            cv2.rectangle(overlay, (x_a, y1), (x_b, y2), neg_color, -1)
            cv2.rectangle(overlay, (x_b, y1), (x2, y2), pos_color, -1)

        elif feat.type == FeatureType.FOUR:
            mid_x = x1 + dx // 2
            mid_y = y1 + dy // 2
            cv2.rectangle(overlay, (x1, y1), (mid_x, mid_y), pos_color, -1)
            cv2.rectangle(overlay, (mid_x, y1), (x2, mid_y), neg_color, -1)
            cv2.rectangle(overlay, (x1, mid_y), (mid_x, y2), neg_color, -1)
            cv2.rectangle(overlay, (mid_x, mid_y), (x2, y2), pos_color, -1)

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
