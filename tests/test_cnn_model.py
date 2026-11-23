"""Tests for Set 8: CNN classification with a model passport, and feature projection.

The real models are Keras networks trained in Colab. These tests build small model
folders in a temporary directory (passport, test samples, test features) and replace the
network by a NumPy "predictor", so they run without TensorFlow (as in CI). Only the last
class needs Keras and is skipped when it is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import cnn_model as cm
from forensics_app.tools import cnn_model_tools as ct
from forensics_app.tools.dialogs import TaskCancelled

T = "forensics_app.tools.cnn_model_tools"
CLASSES = ["cat", "dog", "ship"]
COLORS = {"cat": (220, 30, 30), "dog": (30, 220, 30), "ship": (30, 30, 220)}  # each class has its own colour


def passport_data(**overrides: object) -> dict:
    """A complete passport like the one the notebook writes, for a 3-class model with 8 x 8 inputs."""
    data = {
        "format_version": cm.PASSPORT_VERSION,
        "name": "tiny_cnn",
        "title": "Tiny CNN",
        "description": "Test model",
        "created": "2026-11-17T10:00:00+00:00",
        "framework": {"Keras": "3.10.0", "TensorFlow": "2.19.0"},
        "architecture": {"summary": "Conv -> Dense", "parameters": 1234, "feature_layer": "features", "feature_size": 6,
                         "gradcam_layer": "last_conv"},
        "dataset": {"name": "Toy", "source": "tests", "train_samples": 90, "validation_samples": 10, "test_samples": 30},
        "class_names": CLASSES,
        "preprocessing": {"input_size": [8, 8], "resize_mode": "center_crop", "scale": 1 / 255, "offset": 0.0},
        "training": {"optimizer": "SGD", "loss": "categorical_crossentropy", "epochs": 3, "batch_size": 32,
                     "seconds": 4.2, "phases": [{"name": "frozen", "epochs": 2}, {"name": "fine-tune", "epochs": 1}]},
        "history": {"loss": [1.0, 0.7, 0.5], "accuracy": [0.4, 0.6, 0.8],
                    "val_loss": [1.1, 0.8, 0.7], "val_accuracy": [0.4, 0.55, 0.7]},
        "evaluation": {"test_loss": 0.6, "test_accuracy": 0.75, "confusion_matrix": [[9, 1, 0], [2, 8, 0], [0, 0, 10]]},
    }
    data.update(overrides)
    return data


def make_model_folder(root: Path, name: str = "tiny_cnn", samples: bool = True, features: bool = True,
                      **overrides: object) -> Path:
    """Write a model folder: passport.json, an (empty) model.keras, test samples and test features."""
    folder = root / name
    folder.mkdir(parents=True)
    (folder / cm.PASSPORT_FILE).write_text(json.dumps(passport_data(name=name, **overrides)))
    (folder / "model.keras").write_bytes(b"not a real network")
    if samples:
        (folder / "test_samples").mkdir()
        for index, label in enumerate(CLASSES):
            Image.new("RGB", (8, 8), COLORS[label]).save(folder / "test_samples" / f"{index:05d}_{label}.png")
        Image.new("RGB", (8, 8)).save(folder / "test_samples" / "notes.png")  # not <index>_<class>: ignored
    if features:
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(3), 20)
        vectors = np.eye(3)[labels] * 10 + rng.normal(0, 0.5, (60, 3))
        predictions = labels.copy()
        predictions[:4] = (predictions[:4] + 1) % 3  # four mistakes
        np.savez_compressed(folder / "test_features.npz", features=np.hstack([vectors, vectors]).astype(np.float32),
                            labels=labels, predictions=predictions)
    return folder


def colour_predictor(batch: np.ndarray) -> np.ndarray:
    """Stand-in for the network: the class whose colour channel is strongest wins (softmax of channel means)."""
    means = batch.mean(axis=(1, 2)) * 10
    exp = np.exp(means - means.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class FakeNetwork:
    """The colour predictor, plus Grad-CAM maps whose evidence is in the top-left corner."""

    gradcam_layer = "last_conv"

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return colour_predictor(batch)

    def gradcam_maps(self, pixels: np.ndarray, class_index: int) -> tuple[np.ndarray, np.ndarray]:
        activations = np.zeros((4, 4, 2))
        activations[0, 0, 0] = 1.0  # map 0 fires top-left, and raises the score
        activations[3, 3, 1] = 1.0  # map 1 fires bottom-right, and lowers the score
        gradients = np.stack([np.ones((4, 4)), -np.ones((4, 4))], axis=2)
        return activations, gradients


class TempModelsTestCase(unittest.TestCase):
    """Creates a temporary models/cnn folder with one complete model."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.folder = make_model_folder(self.root)
        self.passport = cm.load_passport(self.folder)

    def tearDown(self) -> None:
        self._tmp.cleanup()


# --------------------------------------------------------------------------- preprocessing


class PreprocessingTests(unittest.TestCase):
    """The passport's preprocessing: crop box, resize, pixel scaling, any PIL mode."""

    def test_from_dict_uses_defaults_for_optional_fields(self) -> None:
        prep = cm.Preprocessing.from_dict({"input_size": [32, 24]})
        self.assertEqual(prep, cm.Preprocessing((32, 24), "center_crop", 1 / 255, 0.0))
        self.assertIn("24 x 32 px", prep.describe())

    def test_from_dict_reports_bad_values(self) -> None:
        for data, message in (({}, "input_size"), ({"input_size": [0, 5]}, "positive"),
                              ({"input_size": [8, 8], "resize_mode": "zoom"}, "resize_mode")):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, message):
                cm.Preprocessing.from_dict(data)

    def test_center_crop_is_the_centred_square_for_a_square_input(self) -> None:
        square = cm.Preprocessing((32, 32))
        self.assertEqual(cm.crop_box((640, 480), square), (80, 0, 560, 480))
        self.assertEqual(cm.crop_box((480, 640), square), (0, 80, 480, 560))
        self.assertEqual(cm.crop_box((32, 32), square), (0, 0, 32, 32))

    def test_center_crop_follows_a_non_square_input_shape(self) -> None:
        tall = cm.Preprocessing((50, 37))  # LFW-like: 50 high, 37 wide
        left, top, right, bottom = cm.crop_box((640, 480), tall)
        self.assertEqual((top, bottom), (0, 480))
        self.assertAlmostEqual((right - left) / (bottom - top), 37 / 50, places=2)

    def test_stretch_uses_the_whole_image(self) -> None:
        self.assertEqual(cm.crop_box((640, 480), cm.Preprocessing((32, 32), "stretch")), (0, 0, 640, 480))

    def test_prepare_input_shape_and_scaling(self) -> None:
        prep = cm.Preprocessing((8, 6), scale=1 / 127.5, offset=-1.0)  # InceptionV3 scaling to [-1, 1]
        pixels, box = cm.prepare_input(Image.new("RGB", (60, 80), (255, 0, 128)), prep)
        self.assertEqual(pixels.shape, (8, 6, 3))
        self.assertEqual(pixels.dtype, np.float32)
        np.testing.assert_allclose(pixels[0, 0], [1.0, -1.0, 128 / 127.5 - 1], atol=1e-6)
        self.assertEqual(box, (0, 0, 60, 80))

    def test_prepare_input_keeps_the_centre_of_a_wide_image(self) -> None:
        image = Image.new("RGB", (300, 100), (0, 0, 0))
        image.paste((255, 255, 255), (100, 0, 200, 100))  # the white centre is exactly the square crop
        pixels, _ = cm.prepare_input(image, cm.Preprocessing((10, 10)))
        self.assertGreater(pixels.min(), 0.9)

    def test_prepare_input_accepts_any_mode(self) -> None:
        prep = cm.Preprocessing((4, 4))
        for image in (Image.new("L", (9, 9), 200), Image.new("RGBA", (9, 9), (1, 2, 3, 4)),
                      Image.new("I;16", (9, 9), 4000), Image.new("P", (9, 9))):
            with self.subTest(mode=image.mode):
                self.assertEqual(cm.prepare_input(image, prep)[0].shape, (4, 4, 3))

    def test_empty_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            cm.prepare_input(Image.new("RGB", (0, 0)), cm.Preprocessing((4, 4)))

    def test_input_to_image_undoes_the_scaling(self) -> None:
        prep = cm.Preprocessing((4, 4), scale=1 / 127.5, offset=-1.0)
        image = Image.new("RGB", (4, 4), (10, 128, 250))
        shown = cm.input_to_image(cm.prepare_input(image, prep)[0], prep)
        self.assertEqual(shown.getpixel((0, 0)), (10, 128, 250))


# --------------------------------------------------------------------------- passport


class PassportTests(TempModelsTestCase):
    """Reading, checking and listing model passports."""

    def test_load_reads_every_part(self) -> None:
        passport = self.passport
        self.assertEqual(passport.name, "tiny_cnn")
        self.assertEqual(passport.class_names, CLASSES)
        self.assertEqual(passport.preprocessing.input_size, (8, 8))
        self.assertEqual(passport.model_path, self.folder / "model.keras")
        self.assertEqual(passport.history["val_accuracy"][-1], 0.7)
        self.assertEqual(passport.test_accuracy, 0.75)
        self.assertEqual(passport.confusion.shape, (3, 3))
        self.assertEqual(passport.label, "Tiny CNN (test accuracy 75.0%)")

    def test_optional_fields_have_defaults(self) -> None:
        folder = self.root / "minimal"
        folder.mkdir()
        (folder / cm.PASSPORT_FILE).write_text(json.dumps(
            {"format_version": 1, "class_names": ["a", "b"], "preprocessing": {"input_size": [4, 4]}}))
        passport = cm.load_passport(folder)
        self.assertEqual((passport.name, passport.title, passport.label), ("minimal", "minimal", "minimal"))
        self.assertEqual(passport.history, {})
        self.assertIsNone(passport.test_accuracy)
        self.assertIsNone(passport.confusion)

    def test_files_section_renames_the_files(self) -> None:
        folder = make_model_folder(self.root, "renamed", files={"model": "net.keras", "test_samples": "pics"})
        passport = cm.load_passport(folder)
        self.assertEqual(passport.model_path.name, "net.keras")
        self.assertEqual(passport.samples_dir.name, "pics")
        self.assertEqual(passport.features_path.name, "test_features.npz")

    def test_invalid_passports_explain_the_problem(self) -> None:
        cases = {
            "version": ({"format_version": 99}, "format_version 99"),
            "classes": ({"class_names": ["only one"]}, "class_names"),
            "preprocessing": ({"preprocessing": None}, "preprocessing"),
            "size": ({"preprocessing": {"input_size": "big"}}, "input_size"),
        }
        for name, (override, message) in cases.items():
            with self.subTest(name):
                folder = make_model_folder(self.root, name, samples=False, features=False, **override)
                with self.assertRaisesRegex(ValueError, message):
                    cm.load_passport(folder)

    def test_missing_or_unreadable_passport(self) -> None:
        with self.assertRaisesRegex(ValueError, "no passport.json"):
            cm.load_passport(self.root / "nowhere")
        broken = self.root / "broken"
        broken.mkdir()
        (broken / cm.PASSPORT_FILE).write_text("{not json")
        with self.assertRaisesRegex(ValueError, "Could not read"):
            cm.load_passport(broken)
        (broken / cm.PASSPORT_FILE).write_text("[1, 2]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            cm.load_passport(broken)

    def test_list_models_sorts_skips_and_reports(self) -> None:
        make_model_folder(self.root, "a_first", samples=False, features=False)
        make_model_folder(self.root, "bad", samples=False, features=False, format_version=0)
        (self.root / "unrelated").mkdir()
        models, problems = cm.list_models(self.root)
        self.assertEqual([model.name for model in models], ["a_first", "tiny_cnn"])
        self.assertEqual(len(problems), 1)
        self.assertIn("format_version", problems[0])

    def test_list_models_without_folder(self) -> None:
        self.assertEqual(cm.list_models(self.root / "missing"), ([], []))

    def test_no_models_message_points_to_the_notebook(self) -> None:
        text = cm.no_models_message(["bad passport"])
        self.assertIn("cnn_classification_set8.ipynb", text)
        self.assertIn("bad passport", text)


# --------------------------------------------------------------------------- classification


class ClassifyTests(TempModelsTestCase):
    """Classification with a fake predictor, test samples and model loading."""

    def test_classify_ranks_all_classes(self) -> None:
        result = cm.classify(Image.new("RGB", (40, 20), COLORS["dog"]), self.passport, colour_predictor)
        self.assertEqual(result.predicted, "dog")
        self.assertEqual([name for name, _ in result.ranking][0], "dog")
        self.assertAlmostEqual(sum(p for _, p in result.ranking), 1.0)
        self.assertEqual(result.confidence, result.probability("dog"))
        self.assertEqual(result.network_input.shape, (8, 8, 3))
        self.assertEqual(result.crop_box, (10, 0, 30, 20))

    def test_output_size_must_match_the_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "4 scores .* 3 classes"):
            cm.classify(Image.new("RGB", (8, 8)), self.passport, lambda batch: np.ones((1, 4)) / 4)

    def test_true_class_from_file_name(self) -> None:
        self.assertEqual(cm.true_class_from_path("00012_cat.png", CLASSES), "cat")
        self.assertEqual(cm.true_class_from_path(Path("x/3_ship.jpg"), CLASSES), "ship")
        for name in ("cat.png", "00012_horse.png", "abc_cat.png", None):
            self.assertIsNone(cm.true_class_from_path(name, CLASSES))

    def test_list_test_samples(self) -> None:
        samples = cm.list_test_samples(self.passport)
        self.assertEqual([label for _, label in samples], CLASSES)  # notes.png is ignored
        no_samples = cm.load_passport(make_model_folder(self.root, "bare", samples=False))
        self.assertEqual(cm.list_test_samples(no_samples), [])

    def test_load_network_errors(self) -> None:
        with patch.object(cm, "keras_available", return_value=False), self.assertRaisesRegex(ValueError, "requirements-keras"):
            cm.load_network(self.passport)
        self.passport.model_path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            cm.load_network(self.passport)

    def test_load_network_loads_once_and_wraps_predict(self) -> None:
        model = MagicMock()
        model.predict.return_value = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
        fake_keras = types.SimpleNamespace(models=types.SimpleNamespace(load_model=MagicMock(return_value=model)))
        cm._loaded.clear()
        with patch.dict(sys.modules, {"keras": fake_keras}), patch.object(cm, "keras_available", return_value=True):
            network = cm.load_network(self.passport)
            self.assertIs(cm.load_network(self.passport), network)  # cached
        fake_keras.models.load_model.assert_called_once_with(str(self.passport.model_path), compile=False)
        output = network(np.zeros((1, 8, 8, 3), np.float32))
        self.assertEqual(output.dtype, np.float64)
        np.testing.assert_allclose(output, [[0.2, 0.3, 0.5]], rtol=1e-6)
        self.assertEqual(network.gradcam_layer, "last_conv")  # named by the passport
        cm._loaded.clear()


# --------------------------------------------------------------------------- Set 8.3: projection


class FeatureTests(TempModelsTestCase):
    """Set 8.3: loading test features, subsampling, PCA, t-SNE / UMAP and the silhouette score."""

    def test_load_test_features(self) -> None:
        features = cm.load_test_features(self.passport)
        self.assertEqual(features.features.shape, (60, 6))
        self.assertEqual(len(features), 60)
        self.assertEqual(int(np.sum(features.predictions != features.labels)), 4)

    def test_features_file_problems(self) -> None:
        bare = cm.load_passport(make_model_folder(self.root, "bare", features=False))
        with self.assertRaisesRegex(ValueError, "no saved test features"):
            cm.load_test_features(bare)
        cases = {
            "no labels": {"features": np.zeros((3, 2))},
            "lengths": {"features": np.zeros((3, 2)), "labels": np.zeros(2)},
            "class index": {"features": np.zeros((2, 2)), "labels": np.array([0, 7])},
            "predictions": {"features": np.zeros((2, 2)), "labels": np.zeros(2), "predictions": np.zeros(5)},
        }
        for name, arrays in cases.items():
            with self.subTest(name):
                np.savez(bare.features_path, **arrays)
                with self.assertRaises(ValueError):
                    cm.load_test_features(bare)

    def test_features_without_predictions(self) -> None:
        np.savez(self.passport.features_path, features=np.ones((4, 2)), labels=np.array([0, 1, 2, 0]))
        self.assertIsNone(cm.load_test_features(self.passport).predictions)

    def test_subsample_is_reproducible_and_keeps_rows_together(self) -> None:
        features = cm.load_test_features(self.passport)
        first, second = cm.subsample(features, 25), cm.subsample(features, 25)
        self.assertEqual(len(first), 25)
        np.testing.assert_array_equal(first.labels, second.labels)
        # every kept feature row still has its own label: feature k*10 belongs to class k
        np.testing.assert_array_equal(first.features[:, :3].argmax(axis=1), first.labels)
        self.assertIs(cm.subsample(features, 100), features)

    def test_reduce_dimensions(self) -> None:
        wide = np.random.default_rng(0).normal(size=(80, 70))
        self.assertEqual(cm.reduce_dimensions(wide).shape, (80, cm.PCA_DIMENSIONS))
        self.assertEqual(cm.reduce_dimensions(wide[:, :5]).shape, (80, 5))

    def test_tsne_projection_separates_the_classes(self) -> None:
        features = cm.load_test_features(self.passport)
        points = cm.project(features.features, "tsne", perplexity=30)  # perplexity is capped for 60 points
        self.assertEqual(points.shape, (60, 2))
        self.assertGreater(cm.separation_score(points, features.labels), 0.5)

    def test_projection_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least 5"):
            cm.project(np.zeros((3, 4)), "tsne")
        with self.assertRaisesRegex(ValueError, "Unknown projection"):
            cm.project(np.zeros((10, 4)), "pca")
        with patch.object(cm, "umap_available", return_value=False), self.assertRaisesRegex(ValueError, "umap-learn"):
            cm.project(np.zeros((10, 4)), "umap")

    def test_umap_is_called_with_the_reduced_features(self) -> None:
        reducer = MagicMock()
        reducer.fit_transform.side_effect = lambda data: data[:, :2]
        fake_umap = types.SimpleNamespace(UMAP=MagicMock(return_value=reducer))
        with patch.dict(sys.modules, {"umap": fake_umap}), patch.object(cm, "umap_available", return_value=True):
            points = cm.project(np.random.default_rng(0).normal(size=(12, 3)), "umap", n_neighbors=50)
        self.assertEqual(points.shape, (12, 2))
        self.assertEqual(fake_umap.UMAP.call_args.kwargs["n_neighbors"], 11)  # capped by the number of points

    def test_separation_score_needs_several_classes(self) -> None:
        points = np.random.default_rng(0).normal(size=(10, 2))
        self.assertIsNone(cm.separation_score(points, np.zeros(10)))
        self.assertIsNone(cm.separation_score(points[:3], np.arange(3)))


# --------------------------------------------------------------------------- Grad-CAM


class GradcamTests(TempModelsTestCase):
    """Grad-CAM: the heatmap formula, the explained class, the overlay and the layer choice."""

    def test_heatmap_weights_maps_by_their_mean_gradient(self) -> None:
        activations, gradients = FakeNetwork().gradcam_maps(np.zeros((8, 8, 3)), 0)
        heatmap = cm.gradcam_heatmap(activations, gradients)
        self.assertEqual(heatmap.shape, (4, 4))
        self.assertEqual(heatmap[0, 0], 1.0)  # evidence for the class, scaled to 1
        self.assertEqual(heatmap[3, 3], 0.0)  # evidence against it is removed by the ReLU
        self.assertEqual(heatmap.sum(), 1.0)

    def test_heatmap_matches_the_formula(self) -> None:
        rng = np.random.default_rng(0)
        activations, gradients = rng.random((5, 6, 3)), rng.normal(size=(5, 6, 3))
        expected = np.maximum(sum(gradients[..., k].mean() * activations[..., k] for k in range(3)), 0)
        np.testing.assert_allclose(cm.gradcam_heatmap(activations, gradients), expected / expected.max())

    def test_heatmap_without_evidence_is_zero(self) -> None:
        self.assertEqual(cm.gradcam_heatmap(np.ones((3, 3, 2)), -np.ones((3, 3, 2))).max(), 0.0)
        with self.assertRaisesRegex(ValueError, "same"):
            cm.gradcam_heatmap(np.ones((3, 3, 2)), np.ones((3, 3, 4)))

    def test_explain_uses_the_predicted_class(self) -> None:
        network = FakeNetwork()
        network.gradcam_maps = MagicMock(wraps=network.gradcam_maps)
        result = cm.classify(Image.new("RGB", (8, 8), COLORS["ship"]), self.passport, network)
        heatmap = cm.explain(network, result, self.passport)
        self.assertEqual(network.gradcam_maps.call_args.args[1], CLASSES.index("ship"))
        self.assertEqual(heatmap.shape, (4, 4))
        with self.assertRaisesRegex(ValueError, "Grad-CAM is not available"):
            cm.explain(colour_predictor, result, self.passport)

    def test_overlay_covers_the_region_the_network_saw(self) -> None:
        image = Image.new("RGB", (400, 200), (128, 128, 128))
        heatmap = np.zeros((4, 4))
        heatmap[0, 0] = 1.0  # evidence in the top-left corner of the region
        overlay = ct.gradcam_overlay(image, (100, 0, 300, 200), heatmap, side=100)
        self.assertEqual(overlay.size, (100, 100))
        corner, opposite = overlay.getpixel((2, 2)), overlay.getpixel((97, 97))
        self.assertGreater(corner[0], corner[2])  # red where the evidence is
        self.assertGreater(opposite[2], opposite[0])  # blue where there is none

    def test_last_spatial_layer_prefers_convolutions(self) -> None:
        def layer(name: str, kind: str, rank: int) -> object:
            return type(kind, (), {"name": name, "output": types.SimpleNamespace(shape=(None,) * rank)})()

        model = types.SimpleNamespace(layers=[layer("conv_a", "Conv2D", 4), layer("pool", "MaxPooling2D", 4),
                                              layer("flat", "Flatten", 2), layer("dense", "Dense", 2)])
        self.assertEqual(cm.last_spatial_layer(model), "conv_a")
        nested = types.SimpleNamespace(layers=[layer("resize", "Resizing", 4), layer("inception_v3", "Functional", 4),
                                               layer("pool", "GlobalAveragePooling2D", 2)])
        self.assertEqual(cm.last_spatial_layer(nested), "inception_v3")
        with self.assertRaisesRegex(ValueError, "feature maps"):
            cm.last_spatial_layer(types.SimpleNamespace(layers=[layer("dense", "Dense", 2)]))

    def test_network_finds_the_layer_when_the_passport_does_not_name_it(self) -> None:
        conv = type("Conv2D", (), {"name": "conv2d_7", "output": types.SimpleNamespace(shape=(None, 5, 5, 8))})()
        network = cm.KerasNetwork(types.SimpleNamespace(layers=[conv]), gradcam_layer=None)
        self.assertEqual(network.gradcam_layer, "conv2d_7")
        self.assertIsNone(cm.ModelPassport(self.folder, {"architecture": {}}).gradcam_layer)
        self.assertEqual(self.passport.gradcam_layer, "last_conv")


# --------------------------------------------------------------------------- figures and text


class RenderTests(TempModelsTestCase):
    """The figures and the results-panel text."""

    def test_classification_figure_with_and_without_ground_truth(self) -> None:
        image = Image.new("RGB", (64, 48), COLORS["cat"])
        result = cm.classify(image, self.passport, colour_predictor)
        for true_class in ("cat", "ship", None):
            with self.subTest(true_class=true_class):
                figure = ct.render_classification(image, result, self.passport, true_class)
                self.assertEqual(figure.mode, "RGB")
                self.assertEqual(figure.size, (1400, 860))

    def test_classification_figure_without_history_or_confusion(self) -> None:
        folder = make_model_folder(self.root, "plain", history={}, evaluation={},
                                   preprocessing={"input_size": [8, 8], "resize_mode": "stretch"})
        passport = cm.load_passport(folder)
        big = Image.new("RGB", (2000, 1000), COLORS["ship"])  # shrunk for plotting
        result = cm.classify(big, passport, colour_predictor)
        self.assertEqual(ct.render_classification(big, result, passport, None).size, (1400, 860))

    def test_projection_figure(self) -> None:
        features = cm.load_test_features(self.passport)
        points = features.features[:, :2]
        one = ct.render_projection({"tsne": points}, features, self.passport, {"tsne": 0.7}, True)
        two = ct.render_projection({"tsne": points, "umap": points}, features, self.passport, {"tsne": 0.7, "umap": None}, False)
        self.assertGreater(two.width, one.width)
        self.assertEqual(one.height, two.height)

    def test_class_colors(self) -> None:
        self.assertEqual(ct.class_colors(3).shape, (3, 4))
        self.assertEqual(len({tuple(row) for row in ct.class_colors(15)}), 15)
        self.assertEqual(ct.class_colors(40).shape, (40, 4))

    def test_passport_details(self) -> None:
        details = ct.passport_details(self.passport)
        self.assertEqual(details["Parameters"], "1,234")
        self.assertEqual(details["Samples (train / validation / test)"], "90 / 10 / 30")
        self.assertEqual(details["Classes"], "3: cat, dog, ship")
        self.assertEqual(details["Final validation accuracy"], "70.0%")
        self.assertEqual(details["Test accuracy"], "75.0%")
        self.assertIn("8 x 8 px", details["Preprocessing"])
        minimal = cm.ModelPassport(self.folder, {"class_names": ["a", "b"], "preprocessing": {"input_size": [2, 2]}})
        self.assertEqual(set(ct.passport_details(minimal)), {"Model", "Model folder", "Classes", "Preprocessing"})

    def test_classification_details(self) -> None:
        image = Image.new("RGB", (16, 8), COLORS["dog"])
        result = cm.classify(image, self.passport, colour_predictor)
        details = ct.classification_details(result, self.passport, image, "00001_dog.png", "dog")
        self.assertTrue(details["Prediction"].startswith("dog ("))
        self.assertEqual(details["Correct?"], "Yes")
        self.assertIn("p(dog)", details)
        self.assertIn("resized to 8 × 8 px", details["Network input"])
        self.assertIn("Test accuracy", details)  # the passport follows the prediction
        unknown = ct.classification_details(result, self.passport, image, "photo.png", None)
        self.assertNotIn("Correct?", unknown)
        self.assertIn("unknown", unknown["Ground truth"])

    def test_projection_details(self) -> None:
        features = cm.load_test_features(self.passport)
        details = ct.projection_details(self.passport, features, 100, {"tsne": 0.81, "umap": None},
                                        {"perplexity": 30.0, "n_neighbors": 15})
        self.assertIn("60 of 100", details["Feature vectors"])
        self.assertIn("0.81", details["Class separation t-SNE"])
        self.assertNotIn("Class separation UMAP", details)
        self.assertIn("4 misclassified", details["Network accuracy on these images"])
        self.assertEqual(details["Points of dog"], 20)


# --------------------------------------------------------------------------- tools


def classify_dialog(passport: cm.ModelPassport, source: str, sample: Path | None = None,
                    gradcam: bool = True) -> MagicMock:
    dialog = MagicMock()
    dialog.user_confirmed = True
    dialog.model, dialog.source, dialog.sample, dialog.gradcam = passport, source, sample, gradcam
    return dialog


class ClassifyToolTests(TempModelsTestCase):
    """The 'CNN: Classify Image' tool end to end, with its dialog and the network replaced."""

    def setUp(self) -> None:
        super().setUp()
        self.patches = [
            patch.object(cm, "CNN_MODELS_DIR", self.root),
            patch(f"{T}.keras_available", return_value=True),
            patch(f"{T}.load_network", return_value=FakeNetwork()),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in self.patches:
            item.stop()
        super().tearDown()

    def run_tool(self, dialog: MagicMock, document: ImageDocument | None = None):
        with patch(f"{T}.CnnClassifyDialog", return_value=dialog) as dialog_class:
            result = ct.CnnClassifyTool().run(None, document or ImageDocument())
        return result, dialog_class

    def test_saved_test_image_is_classified_with_its_true_class(self) -> None:
        sample = self.passport.samples_dir / "00002_ship.png"
        result, _ = self.run_tool(classify_dialog(self.passport, "sample", sample))
        self.assertTrue(result.preview_only)
        self.assertEqual(result.details["Ground truth"], "ship")
        self.assertEqual(result.details["Correct?"], "Yes")
        self.assertIn("'ship'", result.message)
        self.assertIn("correct", result.message)
        self.assertEqual(result.details["Model"], "Tiny CNN")
        self.assertEqual(result.image.size, (1400, 860))

    def test_current_image_of_the_document(self) -> None:
        document = ImageDocument()
        document.load(self.passport.samples_dir / "00000_cat.png")  # a test sample opened in the app
        result, dialog_class = self.run_tool(classify_dialog(self.passport, "current"), document)
        self.assertEqual(dialog_class.call_args.args[2], "00000_cat.png")  # offered as the current image
        self.assertEqual(result.details["Prediction"].split()[0], "cat")
        self.assertEqual(result.details["Ground truth"], "cat")

    def test_unsaved_current_image_has_no_ground_truth(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (30, 20), COLORS["dog"])
        result, dialog_class = self.run_tool(classify_dialog(self.passport, "current"), document)
        self.assertEqual(dialog_class.call_args.args[2], "unsaved image")
        self.assertIn("unknown", result.details["Ground truth"])
        self.assertIn("current image", result.details["Image"])

    def test_image_opened_from_disk(self) -> None:
        photo = Image.new("RGB", (640, 480), COLORS["dog"])
        with patch(f"{T}.load_image_file", return_value=(Path("holiday.jpg"), photo)):
            result, _ = self.run_tool(classify_dialog(self.passport, "browse"))
        self.assertIn("holiday.jpg (640 × 480 px)", result.details["Image"])
        with patch(f"{T}.load_image_file", return_value=None):
            self.assertIsNone(self.run_tool(classify_dialog(self.passport, "browse"))[0])

    def test_cancelled_dialog_or_progress(self) -> None:
        dialog = classify_dialog(self.passport, "sample", self.passport.samples_dir / "00000_cat.png")
        dialog.user_confirmed = False
        self.assertIsNone(self.run_tool(dialog)[0])
        dialog.user_confirmed = True
        with patch(f"{T}.run_with_progress", side_effect=TaskCancelled()):
            self.assertIsNone(self.run_tool(dialog)[0])

    def test_grad_cam_is_drawn_and_described(self) -> None:
        sample = self.passport.samples_dir / "00001_dog.png"
        result, _ = self.run_tool(classify_dialog(self.passport, "sample", sample))
        self.assertIn("class 'dog', layer 'last_conv' (4 × 4 maps)", result.details["Grad-CAM"])
        without, _ = self.run_tool(classify_dialog(self.passport, "sample", sample, gradcam=False))
        self.assertNotIn("Grad-CAM", without.details)
        self.assertEqual(without.image.size, result.image.size)

    def test_grad_cam_failure_still_shows_the_prediction(self) -> None:
        sample = self.passport.samples_dir / "00001_dog.png"
        with patch(f"{T}.load_network", return_value=colour_predictor):  # no gradcam_maps
            result, _ = self.run_tool(classify_dialog(self.passport, "sample", sample))
        self.assertIn("not available", result.details["Grad-CAM"])
        self.assertTrue(result.details["Prediction"].startswith("dog"))

    def test_without_models_or_keras(self) -> None:
        with patch.object(cm, "CNN_MODELS_DIR", self.root / "empty"), self.assertRaisesRegex(ValueError, "notebook"):
            ct.CnnClassifyTool().run(None, ImageDocument())
        with patch(f"{T}.keras_available", return_value=False), self.assertRaisesRegex(ValueError, "requirements-keras"):
            ct.CnnClassifyTool().run(None, ImageDocument())


class ProjectionToolTests(TempModelsTestCase):
    """The 'CNN: Feature Projection' tool end to end, with its dialog replaced."""

    def dialog(self, methods: list[str]) -> MagicMock:
        dialog = MagicMock()
        dialog.user_confirmed = True
        dialog.model, dialog.methods = self.passport, methods
        dialog.max_points, dialog.perplexity, dialog.n_neighbors, dialog.mark_errors = 40, 10.0, 8, True
        return dialog

    def run_tool(self, dialog: MagicMock):
        with patch.object(cm, "CNN_MODELS_DIR", self.root), patch(f"{T}.CnnProjectionDialog", return_value=dialog):
            return ct.CnnProjectionTool().run(None, ImageDocument())

    def test_tsne_projection(self) -> None:
        result = self.run_tool(self.dialog(["tsne"]))
        self.assertTrue(result.preview_only)
        self.assertIn("40 test feature vectors projected with t-SNE", result.message)
        self.assertIn("40 of 60", result.details["Feature vectors"])
        self.assertIn("silhouette", result.details["Class separation t-SNE"])
        self.assertEqual(result.details["t-SNE"], "perplexity 10, random seed 0")

    def test_both_methods(self) -> None:
        with patch(f"{T}.project", side_effect=lambda features, method, **kw: features[:, :2]):
            result = self.run_tool(self.dialog(["tsne", "umap"]))
        self.assertIn("t-SNE and UMAP", result.message)
        self.assertIn("n_neighbors 8", result.details["UMAP"])

    def test_cancelled(self) -> None:
        dialog = self.dialog(["tsne"])
        dialog.user_confirmed = False
        self.assertIsNone(self.run_tool(dialog))
        with patch(f"{T}.run_with_progress", side_effect=TaskCancelled()):
            self.assertIsNone(self.run_tool(self.dialog(["tsne"])))

    def test_models_without_features_are_not_offered(self) -> None:
        make_model_folder(self.root, "no_features", features=False)
        with patch.object(cm, "CNN_MODELS_DIR", self.root), patch(f"{T}.CnnProjectionDialog") as dialog_class:
            dialog_class.return_value.user_confirmed = False
            ct.CnnProjectionTool().run(None, ImageDocument())
        offered = dialog_class.call_args.args[1]
        self.assertEqual([model.name for model in offered], ["tiny_cnn"])
        self.passport.features_path.unlink()
        with patch.object(cm, "CNN_MODELS_DIR", self.root), self.assertRaisesRegex(ValueError, "test_features.npz"):
            ct.CnnProjectionTool().run(None, ImageDocument())


class RegistryTests(unittest.TestCase):
    """Both tools appear in the sidebar registry."""

    def test_tools_are_registered_in_their_category(self) -> None:
        tools = {tool.tool_id: tool for tool in build_tool_registry().all()}
        for tool_id in ("cnn_classify", "cnn_projection"):
            self.assertIn(tool_id, tools)
            self.assertEqual(tools[tool_id].category, ct.CATEGORY)
            self.assertFalse(tools[tool_id].requires_image)


@unittest.skipUnless(cm.keras_available(), "Keras is not installed (optional Set 8 dependency)")
class RealKerasTests(unittest.TestCase):
    """A tiny real Keras model through load_network, classify and Grad-CAM (needs TensorFlow)."""

    def test_saved_keras_model_classifies_through_its_passport(self) -> None:
        import keras

        with tempfile.TemporaryDirectory() as tmp:
            folder = make_model_folder(Path(tmp), samples=False, features=False)
            model = keras.Sequential([
                keras.Input(shape=(8, 8, 3)),
                keras.layers.Conv2D(4, (3, 3), activation="relu", name="last_conv"),
                keras.layers.Flatten(),
                keras.layers.Dense(4, activation="relu", name="features"),
                keras.layers.Dense(3, name="logits"),
                keras.layers.Activation("softmax", name="predictions"),
            ])
            model.save(folder / "model.keras")
            passport = cm.load_passport(folder)
            network = cm.load_network(passport)
            result = cm.classify(Image.new("RGB", (20, 10), (5, 100, 200)), passport, network)
            heatmap = cm.explain(network, result, passport)
        self.assertAlmostEqual(sum(p for _, p in result.ranking), 1.0, places=5)
        self.assertEqual(heatmap.shape, (6, 6))  # 8 x 8 input, 3 x 3 convolution without padding
        self.assertTrue(0.0 <= heatmap.min() and heatmap.max() <= 1.0)
        cm._loaded.clear()


if __name__ == "__main__":
    unittest.main()
