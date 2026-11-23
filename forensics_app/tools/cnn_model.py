"""Set 8: use a CNN trained in Colab to classify images, and look at its features.

The models are trained with the notebook ``docs/cnn_classification_set8.ipynb`` (Keras,
CIFAR-10). Every trained model is one self-contained folder, copied into the app as::

    models/cnn/<model name>/
        model.keras          the network: architecture + trained weights
        passport.json        the model passport (see below)
        test_features.npz    feature vectors of the test set, for Set 8.3 (optional)
        test_samples/        a few test images named <index>_<class>.png (optional)

The passport is what makes a saved network usable outside the notebook. It records the
dataset and its sample sizes, the class names (the meaning of each network output), the
preprocessing instructions, the training history and the test results. The app reads the
preprocessing from it and applies exactly the same steps to any image before classifying.

Besides the prediction, Grad-CAM shows where in the image the network found the evidence
for the class it chose.

Nothing here imports Tkinter, and Keras / UMAP are imported only when they are actually
used, so the rest of the app and the tests run without TensorFlow (as in CI).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from forensics_app.paths import MODELS_DIR, display_path
from .cnn_features import to_rgb

# --------------------------------------------------------------------------- constants

CNN_MODELS_DIR = MODELS_DIR / "cnn"  # one sub-folder per trained model
PASSPORT_FILE = "passport.json"
PASSPORT_VERSION = 1  # bumped by the notebook if the passport layout ever changes
DEFAULT_FILES = {"model": "model.keras", "test_features": "test_features.npz", "test_samples": "test_samples"}
RESIZE_MODES = ("center_crop", "stretch")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# Test samples are saved by the notebook as "<index>_<class name>.png", e.g. "00042_cat.png"
SAMPLE_NAME = re.compile(r"^\d+_(?P<label>.+)$")
# Features are reduced to this many dimensions with PCA before t-SNE / UMAP
PCA_DIMENSIONS = 50
PROJECTION_METHODS = {"tsne": "t-SNE", "umap": "UMAP"}

KERAS_INSTALL_HINT = (
    "TensorFlow / Keras is not installed, so the trained CNN cannot be loaded. Install the "
    "optional Set 8 dependencies with:  .venv/bin/pip install -r requirements-keras.txt"
)
UMAP_INSTALL_HINT = (
    "UMAP is not installed (package umap-learn). Choose t-SNE, or install the optional "
    "Set 8 dependencies with:  .venv/bin/pip install -r requirements-keras.txt"
)

# A predictor turns a batch of preprocessed images (n, height, width, 3) into class
# probabilities (n, number of classes). The real one (KerasNetwork) wraps the Keras model;
# tests pass a fake. A predictor may also offer ``gradcam_maps`` (see ``KerasNetwork``).
Predictor = Callable[[np.ndarray], np.ndarray]
# Layer whose output gives the class scores before the softmax (named so by the notebook)
LOGITS_LAYER = "logits"


def keras_available() -> bool:
    """True when Keras can be imported (it is an optional dependency)."""
    return importlib.util.find_spec("keras") is not None


def umap_available() -> bool:
    """True when umap-learn can be imported (it is an optional dependency)."""
    return importlib.util.find_spec("umap") is not None


# --------------------------------------------------------------------------- preprocessing


@dataclass(frozen=True)
class Preprocessing:
    """The preprocessing instructions of a passport: how to turn any image into a network input.

    ``input_size`` is (height, width) of the network input. ``resize_mode`` says how the
    image reaches that size: ``center_crop`` cuts the largest centred region with the
    input's aspect ratio (a square for CIFAR-10) and resizes it, so objects keep their
    proportions; ``stretch`` resizes the whole image. Finally every pixel value v in
    0..255 becomes ``v * scale + offset`` (e.g. 1/255 and 0 gives the range [0, 1]).
    """

    input_size: tuple[int, int]
    resize_mode: str = "center_crop"
    scale: float = 1.0 / 255.0
    offset: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Preprocessing:
        """Read the ``preprocessing`` section of a passport, with clear errors for bad values."""
        try:
            height, width = (int(value) for value in data["input_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("The passport preprocessing needs 'input_size': [height, width].") from error
        if height < 1 or width < 1:
            raise ValueError(f"The passport input size must be positive (got {height} x {width}).")
        mode = data.get("resize_mode", "center_crop")
        if mode not in RESIZE_MODES:
            raise ValueError(f"Unknown resize_mode {mode!r} in the passport (expected one of {RESIZE_MODES}).")
        return cls((height, width), mode, float(data.get("scale", 1.0 / 255.0)), float(data.get("offset", 0.0)))

    def describe(self) -> str:
        """One readable line, shown in the results panel."""
        height, width = self.input_size
        resize = "centre crop to the input shape, then resize" if self.resize_mode == "center_crop" else "resize (stretch)"
        return f"RGB, {resize} to {width} x {height} px (bilinear), pixel value x {self.scale:.6g} + {self.offset:g}"


def crop_box(size: tuple[int, int], preprocessing: Preprocessing) -> tuple[int, int, int, int]:
    """Region (left, top, right, bottom) of an image of ``size`` = (width, height) that the network sees.

    With ``center_crop`` it is the largest centred box with the same aspect ratio as the
    network input. A 640 x 480 webcam photo and a 32 x 32 network input give the centred
    480 x 480 square, so a cat is not squashed. With ``stretch`` it is the whole image.
    """
    width, height = size
    if preprocessing.resize_mode == "stretch":
        return 0, 0, width, height
    target_height, target_width = preprocessing.input_size
    aspect = target_width / target_height
    crop_width = min(float(width), height * aspect)
    crop_height = crop_width / aspect
    left = int(round((width - crop_width) / 2))
    top = int(round((height - crop_height) / 2))
    right = min(width, left + max(1, int(round(crop_width))))
    bottom = min(height, top + max(1, int(round(crop_height))))
    return left, top, right, bottom


def prepare_input(image: Image.Image, preprocessing: Preprocessing) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Apply the passport preprocessing: returns the (height, width, 3) float32 input and the crop box.

    Any PIL mode is accepted (grayscale, RGBA, 16-bit...): it is converted to RGB first,
    because the networks were trained on colour images.
    """
    rgb = to_rgb(image)
    if rgb.width == 0 or rgb.height == 0:
        raise ValueError("The image is empty.")
    box = crop_box(rgb.size, preprocessing)
    height, width = preprocessing.input_size
    patch = rgb.crop(box).resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(patch, dtype=np.float32) * preprocessing.scale + preprocessing.offset
    return pixels.astype(np.float32), box


def input_to_image(pixels: np.ndarray, preprocessing: Preprocessing) -> Image.Image:
    """Undo the pixel scaling of a network input, to show what the network actually sees."""
    values = (np.asarray(pixels, dtype=np.float64) - preprocessing.offset) / preprocessing.scale
    return Image.fromarray(np.clip(np.round(values), 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- passport


@dataclass(frozen=True)
class ModelPassport:
    """A trained model folder and its passport (the parsed ``passport.json``).

    ``data`` keeps the passport exactly as the notebook saved it; the properties below
    read the parts the app needs, with defaults for the optional ones.
    """

    folder: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        """Short identifier of the model (e.g. 'vgg_cnn'); the folder name if the passport has none."""
        return str(self.data.get("name", self.folder.name))

    @property
    def title(self) -> str:
        """Readable name shown in the UI (e.g. 'VGG-style CNN (2 conv layers)')."""
        return str(self.data.get("title", self.name))

    @property
    def class_names(self) -> list[str]:
        """Meaning of each network output, in order: output i is the probability of class_names[i]."""
        return list(self.data["class_names"])

    @property
    def preprocessing(self) -> Preprocessing:
        """The preprocessing instructions, parsed (and checked) from the passport."""
        return Preprocessing.from_dict(self.data["preprocessing"])

    def _file(self, key: str) -> Path:
        """Path of one of the model's files, as named in the passport's ``files`` section."""
        return self.folder / str(self.data.get("files", {}).get(key, DEFAULT_FILES[key]))

    @property
    def model_path(self) -> Path:
        """The Keras network file (architecture + weights), normally model.keras."""
        return self._file("model")

    @property
    def features_path(self) -> Path:
        """The saved test-set feature vectors for Set 8.3, normally test_features.npz."""
        return self._file("test_features")

    @property
    def samples_dir(self) -> Path:
        """Folder with test images named <index>_<class>.png, normally test_samples/."""
        return self._file("test_samples")

    @property
    def history(self) -> dict[str, list[float]]:
        """Training curves per epoch, e.g. {'loss': [...], 'val_accuracy': [...]}."""
        return {key: [float(v) for v in values] for key, values in self.data.get("history", {}).items()}

    @property
    def test_accuracy(self) -> float | None:
        """Accuracy on the CIFAR-10 test set measured by the notebook, or None if not saved."""
        value = self.data.get("evaluation", {}).get("test_accuracy")
        return None if value is None else float(value)

    @property
    def confusion(self) -> np.ndarray | None:
        """Test-set confusion matrix (rows = true class), if the notebook saved it."""
        matrix = self.data.get("evaluation", {}).get("confusion_matrix")
        return None if matrix is None else np.asarray(matrix, dtype=int)

    @property
    def gradcam_layer(self) -> str | None:
        """Convolutional layer used for Grad-CAM, if the passport names it (else it is found automatically)."""
        return self.data.get("architecture", {}).get("gradcam_layer")

    @property
    def label(self) -> str:
        """Short text for the model selector, e.g. 'VGG-style CNN (test accuracy 71.2%)'."""
        accuracy = self.test_accuracy
        return self.title if accuracy is None else f"{self.title} (test accuracy {accuracy:.1%})"


def load_passport(folder: Path) -> ModelPassport:
    """Read and check ``passport.json`` in ``folder``; raises ValueError explaining what is wrong."""
    folder = Path(folder)
    path = folder / PASSPORT_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"There is no {PASSPORT_FILE} in {display_path(folder)}.") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {display_path(path)}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{display_path(path)} is not a model passport (expected a JSON object).")
    if data.get("format_version") != PASSPORT_VERSION:
        raise ValueError(
            f"{display_path(path)} has format_version {data.get('format_version')!r}, but this app reads "
            f"version {PASSPORT_VERSION}. Re-export the model with the Set 8 notebook."
        )
    names = data.get("class_names")
    if not isinstance(names, list) or len(names) < 2 or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{display_path(path)} needs 'class_names': a list of at least two names.")
    if not isinstance(data.get("preprocessing"), dict):
        raise ValueError(f"{display_path(path)} needs a 'preprocessing' section.")
    passport = ModelPassport(folder, data)
    passport.preprocessing  # parsing it now reports a bad input size or resize mode early
    return passport


def list_models(root: Path | None = None) -> tuple[list[ModelPassport], list[str]]:
    """Every model folder under ``root`` (default ``models/cnn``) with a valid passport.

    Returns (models sorted by folder name, problems): a folder whose passport cannot be
    read is skipped and its error is reported, instead of hiding the other models.
    """
    root = Path(root) if root is not None else CNN_MODELS_DIR
    if not root.is_dir():
        return [], []
    models, problems = [], []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (folder / PASSPORT_FILE).exists():
            continue  # e.g. an unrelated folder
        try:
            models.append(load_passport(folder))
        except ValueError as error:
            problems.append(str(error))
    return models, problems


def no_models_message(problems: list[str]) -> str:
    """Error text when no usable model is found: where to put one, and what went wrong."""
    text = (
        f"No trained CNN was found in {display_path(CNN_MODELS_DIR)}. Run the notebook "
        "docs/cnn_classification_set8.ipynb in Colab, download cnn_models.zip and unzip it so that "
        "every model has its own folder: models/cnn/<model>/passport.json."
    )
    if problems:
        text += "\n\nProblems found:\n" + "\n".join(problems)
    return text


# --------------------------------------------------------------------------- classification


def last_spatial_layer(model: Any) -> str:
    """Name of the last layer that still outputs feature maps (height x width x channels).

    Used for Grad-CAM when the passport does not name a layer: the last convolutional
    layer if there is one at the top level of the model, otherwise the last layer with a
    4-D output (for InceptionV3, the whole pre-trained base, which ends with feature maps).
    """
    spatial = []
    for layer in model.layers:
        try:
            if len(layer.output.shape) == 4:  # (batch, height, width, channels)
                spatial.append(layer)
        except (AttributeError, ValueError):
            continue  # a layer without a single symbolic output
    if not spatial:
        raise ValueError("The model has no layer with feature maps, so Grad-CAM cannot be computed.")
    convolutions = [layer for layer in spatial if "conv" in type(layer).__name__.lower()]
    return (convolutions or spatial)[-1].name


class KerasNetwork:
    """A loaded Keras model. Calling it gives class probabilities; ``gradcam_maps`` serves Grad-CAM."""

    def __init__(self, model: Any, gradcam_layer: str | None = None) -> None:
        """``model`` is the loaded keras.Model; ``gradcam_layer`` the layer named by the passport, if any."""
        self.model = model
        self._gradcam_layer = gradcam_layer

    @property
    def gradcam_layer(self) -> str:
        """Layer whose feature maps Grad-CAM explains; found automatically when the passport has none."""
        if self._gradcam_layer is None:  # looked up only when Grad-CAM is actually asked for
            self._gradcam_layer = last_spatial_layer(self.model)
        return self._gradcam_layer

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        """(n, height, width, 3) preprocessed images -> (n, classes) probabilities."""
        return np.asarray(self.model.predict(batch, verbose=0), dtype=np.float64)

    def gradcam_maps(self, pixels: np.ndarray, class_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Feature maps A of the Grad-CAM layer and the gradients dy/dA of the class score y.

        The layers are called one by one inside a gradient tape, so the tape records the
        path from the feature maps to the class score. (A helper model built from the
        loaded model's inputs and layer outputs can end up disconnected from the score,
        and then TensorFlow returns no gradient.) The Set 8 models are simple chains of
        layers, which this needs. The score is the class logit (before the softmax, as in
        the Grad-CAM paper) when the model has a ``logits`` layer, otherwise the
        log-probability. Returns two (h, w, k) arrays.
        """
        import tensorflow as tf  # Keras runs on TensorFlow here (requirements-keras.txt)

        target = self.gradcam_layer
        x = tf.convert_to_tensor(pixels[np.newaxis], dtype=tf.float32)
        maps = logits = None
        with tf.GradientTape() as tape:
            for layer in self.model.layers:
                if type(layer).__name__ == "InputLayer":
                    continue
                x = layer(x)
                if layer.name == target:
                    maps = x
                    tape.watch(maps)  # we want d(score) / d(maps)
                if layer.name == LOGITS_LAYER:
                    logits = x
            if maps is None:
                raise ValueError(f"The model has no layer named '{target}' for Grad-CAM.")
            # x is now the softmax output: without a logits layer, explain the log-probability
            score = logits[:, class_index] if logits is not None else tf.math.log(x[:, class_index] + 1e-9)
        gradients = tape.gradient(score, maps)
        if gradients is None:
            raise ValueError("No gradient reached the Grad-CAM layer: the model is not a simple chain of layers.")
        return np.asarray(maps[0], dtype=np.float64), np.asarray(gradients[0], dtype=np.float64)


_loaded: dict[tuple[str, float], KerasNetwork] = {}  # the last model loaded, so re-running a tool is fast


def load_network(passport: ModelPassport) -> KerasNetwork:
    """Load the Keras model of ``passport``.

    The model is loaded with ``compile=False``: only inference is needed, not the
    optimizer state. The last loaded model is kept in memory, keyed by file and
    modification time, so classifying several images does not reload it every time.
    """
    path = passport.model_path
    if not path.exists():
        raise ValueError(f"The model file {display_path(path)} is missing.")
    if not keras_available():
        raise ValueError(KERAS_INSTALL_HINT)
    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _loaded:
        import keras  # imported here: TensorFlow takes seconds to import and is optional

        model = keras.models.load_model(str(path), compile=False)
        _loaded.clear()  # keep a single model in memory (InceptionV3 alone is about 90 MB)
        _loaded[key] = KerasNetwork(model, passport.gradcam_layer)
    return _loaded[key]


@dataclass(frozen=True)
class Classification:
    """Result of classifying one image."""

    ranking: list[tuple[str, float]]  # (class name, probability), most likely first
    network_input: np.ndarray  # (height, width, 3) array that was given to the network
    crop_box: tuple[int, int, int, int]  # region of the original image the network saw

    @property
    def predicted(self) -> str:
        """The most likely class."""
        return self.ranking[0][0]

    @property
    def confidence(self) -> float:
        """Probability of the most likely class."""
        return self.ranking[0][1]

    def probability(self, name: str) -> float:
        """Probability the network gives to the class ``name``."""
        return dict(self.ranking)[name]


def classify(image: Image.Image, passport: ModelPassport, predictor: Predictor) -> Classification:
    """Preprocess ``image`` as the passport says, run the network and rank the classes."""
    pixels, box = prepare_input(image, passport.preprocessing)
    # The network works on batches: add a batch axis of size 1, then flatten the (1, classes) output
    probabilities = np.asarray(predictor(pixels[np.newaxis]), dtype=np.float64).reshape(-1)
    names = passport.class_names
    if probabilities.shape[0] != len(names):
        raise ValueError(
            f"The model returns {probabilities.shape[0]} scores but its passport lists {len(names)} classes."
        )
    order = np.argsort(-probabilities, kind="stable")  # most likely first; ties keep the class order
    return Classification([(names[i], float(probabilities[i])) for i in order], pixels, box)


def true_class_from_path(path: Path | str | None, class_names: list[str]) -> str | None:
    """Ground truth of a test image saved by the notebook as ``<index>_<class>.png``, else None."""
    if path is None:
        return None
    match = SAMPLE_NAME.match(Path(path).stem)  # "00042_cat" -> label "cat"
    if match and match["label"] in class_names:
        return match["label"]
    return None


def list_test_samples(passport: ModelPassport) -> list[tuple[Path, str]]:
    """(file, true class) of the test images saved next to the model, sorted by file name."""
    folder = passport.samples_dir
    if not folder.is_dir():
        return []
    samples = []
    for path in sorted(folder.iterdir()):
        label = true_class_from_path(path, passport.class_names)
        if path.suffix.lower() in IMAGE_SUFFIXES and label is not None:
            samples.append((path, label))
    return samples


# --------------------------------------------------------------------------- Grad-CAM


def gradcam_heatmap(activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017): where in the image the evidence for a class is.

    ``activations`` are the k feature maps A^k of a convolutional layer, shape (h, w, k),
    and ``gradients`` the derivatives dy/dA^k of the class score y, same shape:

    1. alpha_k = mean of the gradients of map k: how much map k matters for the class;
    2. weighted sum of the maps: sum over k of alpha_k * A^k;
    3. ReLU: keep only the regions that raise the score (evidence *for* the class);
    4. divide by the maximum, so the heatmap goes from 0 to 1.

    Returns an (h, w) map at the resolution of the layer (e.g. 30 x 30 for the small CNN).
    """
    activations = np.asarray(activations, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    if activations.ndim != 3 or activations.shape != gradients.shape:
        raise ValueError("Grad-CAM needs feature maps and gradients of the same (height, width, maps) shape.")
    weights = gradients.mean(axis=(0, 1))  # alpha_k, one weight per feature map
    heatmap = np.maximum(activations @ weights, 0.0)  # sum_k alpha_k A^k, then ReLU
    peak = heatmap.max()
    return heatmap / peak if peak > 0 else heatmap


def explain(network: Predictor, result: Classification, passport: ModelPassport) -> np.ndarray:
    """Grad-CAM heatmap of the predicted class of a classified image.

    ``network`` must offer ``gradcam_maps(pixels, class_index)``, like ``KerasNetwork``.
    """
    gradcam_maps = getattr(network, "gradcam_maps", None)
    if gradcam_maps is None:
        raise ValueError("This network does not provide gradients, so Grad-CAM is not available.")
    class_index = passport.class_names.index(result.predicted)
    return gradcam_heatmap(*gradcam_maps(result.network_input, class_index))


# --------------------------------------------------------------------------- Set 8.3: feature projection


@dataclass(frozen=True)
class FeatureSet:
    """Feature vectors of test images, as extracted by the notebook from the ``features`` layer."""

    features: np.ndarray  # (n, d): one row per test image
    labels: np.ndarray  # (n,): true class index
    predictions: np.ndarray | None  # (n,): class predicted by the network, if saved

    def __len__(self) -> int:
        """Number of test images."""
        return len(self.labels)


def load_test_features(passport: ModelPassport) -> FeatureSet:
    """Load ``test_features.npz`` (arrays ``features``, ``labels`` and optionally ``predictions``)."""
    path = passport.features_path
    if not path.exists():
        raise ValueError(
            f"{passport.title} has no saved test features ({display_path(path)}). Run the Set 8.3 "
            "section of the notebook to extract them."
        )
    try:
        with np.load(path, allow_pickle=False) as data:  # no pickles: never run code from a file
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"]).astype(int).ravel()
            predictions = np.asarray(data["predictions"]).astype(int).ravel() if "predictions" in data.files else None
    except KeyError as error:
        raise ValueError(f"{display_path(path)} must contain the arrays 'features' and 'labels' ({error}).") from error
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read {display_path(path)}: {error}") from error
    n_classes = len(passport.class_names)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError(f"{display_path(path)}: 'features' must have one row per label.")
    if predictions is not None and len(predictions) != len(labels):
        raise ValueError(f"{display_path(path)}: 'predictions' must have one value per label.")
    if len(labels) and (labels.min() < 0 or labels.max() >= n_classes):
        raise ValueError(f"{display_path(path)}: labels must be class indexes between 0 and {n_classes - 1}.")
    return FeatureSet(features, labels, predictions)


def subsample(feature_set: FeatureSet, max_points: int, seed: int = 0) -> FeatureSet:
    """A random subset of at most ``max_points`` test images (same subset for the same seed).

    t-SNE costs grow quickly with the number of points; a few thousand already show the
    class structure. The test set is balanced, so a random subset stays balanced.
    """
    if len(feature_set) <= max_points:
        return feature_set
    rng = np.random.default_rng(seed)
    # sorted indexes: the subset keeps the original order (the notebook picks the same points)
    keep = np.sort(rng.choice(len(feature_set), max_points, replace=False))
    predictions = None if feature_set.predictions is None else feature_set.predictions[keep]
    return FeatureSet(feature_set.features[keep], feature_set.labels[keep], predictions)


def reduce_dimensions(features: np.ndarray, dimensions: int = PCA_DIMENSIONS) -> np.ndarray:
    """PCA down to ``dimensions`` (when there are more): the usual first step before t-SNE / UMAP.

    It removes noise and makes t-SNE much faster while keeping the neighbourhoods that
    both methods try to preserve.
    """
    features = np.asarray(features, dtype=np.float64)
    target = min(dimensions, features.shape[0], features.shape[1])
    if features.shape[1] <= target:
        return features
    return PCA(n_components=target, random_state=0).fit_transform(features)


def project(
    features: np.ndarray, method: str, *, perplexity: float = 30.0, n_neighbors: int = 15, seed: int = 0
) -> np.ndarray:
    """Project the feature vectors onto 2-D with t-SNE (``tsne``) or UMAP (``umap``): shape (n, 2).

    Both try to keep close in 2-D the images that are close in feature space.
    t-SNE's ``perplexity`` is roughly the number of neighbours each point cares about;
    UMAP's ``n_neighbors`` plays the same role. Both are capped by the number of points.
    """
    n = len(features)
    if n < 5:
        raise ValueError("At least 5 feature vectors are needed for a projection.")
    reduced = reduce_dimensions(features)
    if method == "tsne":
        effective = min(float(perplexity), (n - 1) / 3.0)  # t-SNE needs perplexity < number of points
        return TSNE(n_components=2, perplexity=effective, init="pca", random_state=seed).fit_transform(reduced)
    if method == "umap":
        if not umap_available():
            raise ValueError(UMAP_INSTALL_HINT)
        import umap  # imported here: optional, and slow to import (numba)

        reducer = umap.UMAP(n_components=2, n_neighbors=min(int(n_neighbors), n - 1), min_dist=0.1, random_state=seed)
        return np.asarray(reducer.fit_transform(reduced))
    raise ValueError(f"Unknown projection method {method!r} (expected one of {tuple(PROJECTION_METHODS)}).")


def separation_score(embedding: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette score of the classes in the 2-D projection, between -1 and 1.

    Close to 1: every class forms a compact cluster far from the others. Around 0:
    classes overlap. None when it cannot be computed (a single class, or too few points).
    """
    n_classes = len(np.unique(labels))
    if n_classes < 2 or n_classes >= len(labels):
        return None
    return float(silhouette_score(embedding, labels))
