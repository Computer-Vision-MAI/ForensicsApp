"""Keypoint matching between two images (Sets 5.3 and 5.4), without any Tkinter.

The pipeline follows the course slides:

1. Detect keypoints and descriptors on both images (reuses Set 5.2's ``extract_keypoints``).
2. Pair descriptors with ``skimage.feature.match_descriptors`` (cross-check + Lowe ratio test).
3. Optionally reject outliers with RANSAC (``skimage.measure.ransac``): keep only the matches
   that agree with one geometric transform (affine or projective/homography).

Both images are first reduced to a common working size, so images of very different
resolutions are compared at a similar scale and the RANSAC threshold means the same
number of pixels for every pair.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import feature, transform
from skimage.measure import ransac

from .image_utils import image_to_unit_array, to_grayscale_array, unit_array_to_image
from .keypoint_detection import extract_keypoints

METHODS = ("orb", "sift")
RANSAC_MODELS = {
    # name: (transform class, minimum number of point pairs to fit it)
    "affine": (transform.AffineTransform, 3),
    "projective": (transform.ProjectiveTransform, 4),
}
# Fixed RANSAC seed: running the same analysis twice must give the same answer.
RANSAC_SEED = 0

OUTPUT_MATCHES = "matches"
OUTPUT_TRANSFORMED = "transformed"  # image 1 warped with the RANSAC matrix: it becomes image 2
OUTPUT_ALIGNED = "aligned"  # image 2 warped back onto image 1 (the inverse direction)
WARP_OUTPUTS = (OUTPUT_TRANSFORMED, OUTPUT_ALIGNED)

INLIER_COLOR = (0, 220, 0)
OUTLIER_COLOR = (230, 30, 30)
KEYPOINT_COLOR = (255, 0, 0)
# Line colours used when there is no RANSAC (one colour per match, as in the slides)
MATCH_PALETTE = (
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (255, 215, 0),
)
GAP = 10  # pixels between the two images in the side-by-side view


@dataclass(frozen=True)
class MatchSettings:
    """Every parameter of the matching pipeline, validated on creation."""

    method: str = "sift"
    # ORB
    n_keypoints: int = 500
    # SIFT
    n_octaves: int = 8
    n_scales: int = 3
    sigma: float = 1.6
    # Longest side, in pixels, the images are reduced to before detection (0 = full size)
    max_side: int = 640
    # Matching
    cross_check: bool = True
    max_ratio: float = 0.8
    # Outlier rejection
    use_ransac: bool = True
    ransac_model: str = "affine"
    residual_threshold: float = 5.0
    max_trials: int = 1000

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Method must be one of {', '.join(METHODS)} (got '{self.method}').")
        if self.n_keypoints < 1:
            raise ValueError("The number of ORB keypoints must be at least 1.")
        if self.n_octaves < 1 or self.n_scales < 1:
            raise ValueError("SIFT octaves and scales must be at least 1.")
        if not self.sigma >= 1.0:
            raise ValueError("SIFT sigma must be at least 1.0.")
        if self.max_side != 0 and self.max_side < 64:
            raise ValueError("The working size must be 0 (full size) or at least 64 pixels.")
        if not 0.0 < self.max_ratio <= 1.0:
            raise ValueError("The ratio test value must be in (0, 1].")
        if self.ransac_model not in RANSAC_MODELS:
            raise ValueError(
                f"RANSAC model must be one of {', '.join(RANSAC_MODELS)} (got '{self.ransac_model}')."
            )
        if not self.residual_threshold > 0.0:
            raise ValueError("The RANSAC residual threshold must be positive.")
        if self.max_trials < 1:
            raise ValueError("RANSAC needs at least 1 trial.")

    @property
    def metric(self) -> str:
        # ORB descriptors are bit strings: compare them by the number of different bits
        return "hamming" if self.method == "orb" else "euclidean"

    @property
    def min_samples(self) -> int:
        return RANSAC_MODELS[self.ransac_model][1]

    def describe(self) -> dict[str, object]:
        """Parameters worth showing in the Results panel."""
        info: dict[str, object] = {"Method": self.method.upper()}
        if self.method == "orb":
            info["n_keypoints"] = self.n_keypoints
        else:
            info["n_octaves"] = self.n_octaves
            info["n_scales"] = self.n_scales
            info["sigma"] = self.sigma
        info["Working size"] = f"{self.max_side} px" if self.max_side else "full size"
        info["Cross-check"] = self.cross_check
        info["Max ratio"] = self.max_ratio
        if self.use_ransac:
            info["RANSAC model"] = self.ransac_model
            info["Residual threshold"] = f"{self.residual_threshold} px"
            info["Max trials"] = self.max_trials
        else:
            info["RANSAC"] = "off"
        return info


@dataclass(frozen=True)
class ImageFeatures:
    """Keypoints of one image, in the coordinates of its reduced working copy."""

    image: Image.Image  # the 8-bit grayscale working copy the keypoints were found on
    keypoints: np.ndarray  # (N, 2) as (row, col)
    descriptors: np.ndarray | None
    scale: float  # working size / original size

    def __len__(self) -> int:
        return len(self.keypoints)


@dataclass(frozen=True)
class MatchResult:
    """Matches between two images and, if RANSAC ran, which ones are inliers."""

    matches: np.ndarray  # (M, 2) indices into the keypoints of image 1 and image 2
    inliers: np.ndarray | None  # (M,) bool, None when RANSAC is off
    model: transform.ProjectiveTransform | None  # maps image 1 (x, y) to image 2, working coords

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def n_inliers(self) -> int | None:
        return None if self.inliers is None else int(np.count_nonzero(self.inliers))

    @property
    def score(self) -> int:
        """How similar the two images are: inliers with RANSAC, raw matches without."""
        return self.n_matches if self.inliers is None else int(self.n_inliers)


def working_image(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Return an 8-bit grayscale copy whose longest side is at most ``max_side``.

    Any PIL mode is accepted (high-depth ones are normalised first). Returns the
    copy and the scale factor applied (1.0 when the image was already small enough).
    """
    if img.width == 0 or img.height == 0:
        raise ValueError("The image is empty.")
    gray = Image.fromarray(np.round(to_grayscale_array(img) * 255.0).astype(np.uint8))
    longest = max(gray.size)
    if max_side <= 0 or longest <= max_side:
        return gray, 1.0
    scale = max_side / longest
    size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
    return gray.resize(size, Image.Resampling.LANCZOS), scale


def detect_features(img: Image.Image, settings: MatchSettings) -> ImageFeatures:
    """Detect keypoints and descriptors on the working copy of ``img``."""
    work, scale = working_image(img, settings.max_side)
    keypoints, descriptors = extract_keypoints(
        work,
        method=settings.method,
        n_keypoints=settings.n_keypoints,
        n_octaves=settings.n_octaves,
        n_scales=settings.n_scales,
        sigma=settings.sigma,
    )
    return ImageFeatures(work, np.asarray(keypoints, dtype=np.float64), descriptors, scale)


def match_features(
    features1: ImageFeatures, features2: ImageFeatures, settings: MatchSettings
) -> np.ndarray:
    """Pair the descriptors of both images; returns an (M, 2) array of keypoint indices."""
    d1, d2 = features1.descriptors, features2.descriptors
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return np.empty((0, 2), dtype=np.intp)
    return feature.match_descriptors(
        d1,
        d2,
        metric=settings.metric,
        cross_check=settings.cross_check,
        max_ratio=settings.max_ratio,
    )


def ransac_filter(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    settings: MatchSettings,
) -> tuple[transform.ProjectiveTransform | None, np.ndarray]:
    """Fit a transform to the matches with RANSAC and flag the matches that agree with it.

    Keypoints are (row, col); skimage transforms work in (x, y), hence the ``[:, ::-1]``.
    Returns ``(model, inliers)``. The model is None, and no match is an inlier, when
    RANSAC finds no consistent model. A model always fits the ``min_samples`` points it
    was built from, so one that no other match supports is rejected too: with
    unrelated images it would otherwise report a few "inliers" by pure chance.
    """
    inliers = np.zeros(len(matches), dtype=bool)
    model_class, min_samples = RANSAC_MODELS[settings.ransac_model]
    if len(matches) < min_samples:
        return None, inliers

    src = keypoints1[matches[:, 0]][:, ::-1]
    dst = keypoints2[matches[:, 1]][:, ::-1]
    with warnings.catch_warnings():
        # skimage warns when no model fits; that case is reported through the result
        warnings.filterwarnings("ignore", message="No inliers found")
        model, found = ransac(
            (src, dst),
            model_class,
            min_samples=min_samples,
            residual_threshold=settings.residual_threshold,
            max_trials=settings.max_trials,
            rng=RANSAC_SEED,
        )
    if model is None or found is None or np.count_nonzero(found) <= min_samples:
        return None, inliers
    return model, np.asarray(found, dtype=bool)


def compare_features(
    features1: ImageFeatures, features2: ImageFeatures, settings: MatchSettings
) -> MatchResult:
    """Match two images and, if enabled, keep only the geometrically consistent matches."""
    matches = match_features(features1, features2, settings)
    if not settings.use_ransac:
        return MatchResult(matches, None, None)
    model, inliers = ransac_filter(features1.keypoints, features2.keypoints, matches, settings)
    return MatchResult(matches, inliers, model)


def full_resolution_matrix(
    model: transform.ProjectiveTransform, scale1: float, scale2: float
) -> np.ndarray:
    """Turn a working-size model into one between the original images.

    original 1 --(scale1)--> working 1 --(model)--> working 2 --(1/scale2)--> original 2
    """
    to_working = np.diag([scale1, scale1, 1.0])
    from_working = np.diag([1.0 / scale2, 1.0 / scale2, 1.0])
    return from_working @ model.params @ to_working


def align_to_reference(
    reference: Image.Image, other: Image.Image, matrix: np.ndarray
) -> Image.Image:
    """Warp ``other`` onto the pixel grid of ``reference``.

    ``matrix`` maps reference (x, y) to ``other`` (x, y), which is exactly the inverse
    map ``skimage.transform.warp`` expects. Areas with no data are left black.
    """
    pixels, _ = image_to_unit_array(other)
    output_shape = (reference.height, reference.width) + pixels.shape[2:]
    aligned = transform.warp(
        pixels,
        transform.ProjectiveTransform(matrix=matrix),
        output_shape=output_shape,
        order=1,
        cval=0.0,
    )
    return unit_array_to_image(aligned)


def _fit_height(img: Image.Image, height: int) -> tuple[Image.Image, float]:
    factor = height / img.height
    if np.isclose(factor, 1.0):
        return img, 1.0
    size = (max(1, round(img.width * factor)), height)
    return img.resize(size, Image.Resampling.LANCZOS), factor


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType support
        return ImageFont.load_default()


def render_matches(
    features1: ImageFeatures,
    features2: ImageFeatures,
    result: MatchResult,
    title: str | None = None,
    show_keypoints: bool = True,
) -> Image.Image:
    """Draw both images side by side with a line per match, like the course slides.

    Both are scaled to the same height. With RANSAC, inliers are green and
    outliers red; without it every match gets its own colour. Detected keypoints
    are drawn as small red circles. An optional ``title`` is written above.
    """
    height = max(features1.image.height, features2.image.height)
    left, f_left = _fit_height(features1.image, height)
    right, f_right = _fit_height(features2.image, height)
    offset_x = left.width + GAP

    width = offset_x + right.width
    # Keep the title readable once the view shrinks the whole picture to fit
    font = _load_font(max(16, width // 70))
    banner = 0
    if title:
        top, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), title, font=font)[1::2]
        banner = bottom - top + 12

    canvas = Image.new("RGB", (width, height + banner), (255, 255, 255))
    canvas.paste(left.convert("RGB"), (0, banner))
    canvas.paste(right.convert("RGB"), (offset_x, banner))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((canvas.width // 2, 6), title, fill=(0, 0, 0), font=font, anchor="mt")

    def to_canvas(keypoints: np.ndarray, factor: float, dx: int) -> np.ndarray:
        # (row, col) in working coords -> (x, y) on the canvas
        return np.column_stack((keypoints[:, 1] * factor + dx, keypoints[:, 0] * factor + banner))

    points1 = to_canvas(features1.keypoints, f_left, 0) if len(features1) else np.empty((0, 2))
    points2 = to_canvas(features2.keypoints, f_right, offset_x) if len(features2) else np.empty((0, 2))

    if show_keypoints:
        for x, y in np.vstack((points1, points2)):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=KEYPOINT_COLOR, width=1)

    order = np.arange(result.n_matches)
    if result.inliers is not None:
        # Outliers first so the inliers stay visible on top of them
        order = np.concatenate((order[~result.inliers], order[result.inliers]))
    for index in order:
        i, j = result.matches[index]
        (x1, y1), (x2, y2) = points1[i], points2[j]
        if result.inliers is None:
            color, width = MATCH_PALETTE[index % len(MATCH_PALETTE)], 2
        elif result.inliers[index]:
            color, width = INLIER_COLOR, 2
        else:
            color, width = OUTLIER_COLOR, 1
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        for x, y in ((x1, y1), (x2, y2)):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return canvas


def summarize_match(result: MatchResult) -> str:
    if result.inliers is None:
        return f"{result.n_matches} matches"
    return f"{result.n_inliers} inliers / {result.n_matches} matches"


def format_matrix(matrix: np.ndarray) -> str:
    return " ; ".join("[" + ", ".join(f"{value:.3f}" for value in row) + "]" for row in matrix)


def apply_feature_matching(
    image1: Image.Image,
    image2: Image.Image,
    settings: MatchSettings,
    output: str = OUTPUT_MATCHES,
) -> tuple[Image.Image, MatchResult, ImageFeatures, ImageFeatures]:
    """Run the whole pipeline on two images and return the picture to show.

    ``output`` is one of:

    - ``"matches"``: both images side by side with a line per match.
    - ``"transformed"``: ``image1`` warped with the RANSAC matrix, on the grid of
      ``image2``. If ``image2`` is ``image1`` deformed by some matrix X, the result
      is ``image2`` again, because RANSAC estimated X from the matches.
    - ``"aligned"``: the inverse, ``image2`` warped back onto ``image1``
      (slides: "transform img2 back to img1").

    Both warps need RANSAC, since it is what provides the matrix.
    """
    if output not in (OUTPUT_MATCHES, *WARP_OUTPUTS):
        raise ValueError(f"Unknown output '{output}'.")
    if output in WARP_OUTPUTS and not settings.use_ransac:
        raise ValueError("Transforming an image needs RANSAC to estimate the transform.")

    features1 = detect_features(image1, settings)
    features2 = detect_features(image2, settings)
    result = compare_features(features1, features2, settings)

    if output in WARP_OUTPUTS:
        if result.model is None:
            raise ValueError(
                f"RANSAC could not find a reliable transform ({summarize_match(result)}). "
                "Try SIFT, a higher ratio, or a larger residual threshold."
            )
        # Maps image 1 (x, y) to image 2 (x, y), in original pixels
        matrix = full_resolution_matrix(result.model, features1.scale, features2.scale)
        if output == OUTPUT_TRANSFORMED:
            # warp() wants the map from output pixels (image 2) to input ones: the inverse
            warped = align_to_reference(image2, image1, np.linalg.inv(matrix))
        else:
            warped = align_to_reference(image1, image2, matrix)
        return warped, result, features1, features2
    return render_matches(features1, features2, result), result, features1, features2
