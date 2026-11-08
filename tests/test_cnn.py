"""Tests for the Set 7 bonus: BoW versus CNN features.

Almost everything uses a fake "network" written in NumPy, so these tests run without
PyTorch (as in CI). Only the last class needs PyTorch and is skipped when it is missing.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import CnnCompareTool, build_tool_registry
from forensics_app.tools import bow_model as bm
from forensics_app.tools import cnn_features as cf
from forensics_app.tools.cnn_tools import comparison_details, render_comparison
from forensics_app.tools.dialogs import TaskCancelled

FAST = bm.BowSettings(detector="dense", k=6, test_ratio=0.25, pyramid_levels=1, max_side=96)


def texture(kind: str, seed: int, size: int = 96) -> Image.Image:
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size]
    period = rng.integers(10, 14)
    shift = rng.integers(0, period)
    if kind == "stripes":
        pattern = ((rows + shift) // (period // 2)) % 2
    elif kind == "checker":
        pattern = (((rows + shift) // period) + ((cols + shift) // period)) % 2
    else:
        pattern = (((rows + shift) % period - period / 2) ** 2 + ((cols + shift) % period - period / 2) ** 2) < 9
    pixels = 0.2 + 0.6 * pattern + rng.normal(0, 0.05, (size, size))
    return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def fake_embed(batch: np.ndarray) -> np.ndarray:
    """Stand-in for ResNet: mean absolute gradients along rows and columns, mean and spread."""
    gray = batch.mean(axis=1)
    dy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    dx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    return np.column_stack([dy, dx, gray.mean(axis=(1, 2)) + 3, gray.std(axis=(1, 2))])


class PreprocessTests(unittest.TestCase):
    def test_shape_and_imagenet_normalisation(self) -> None:
        pixels = cf.preprocess(Image.new("RGB", (300, 200), (128, 128, 128)))
        self.assertEqual(pixels.shape, (3, cf.CROP_SIDE, cf.CROP_SIDE))
        self.assertEqual(pixels.dtype, np.float32)
        expected = (128 / 255 - cf.IMAGENET_MEAN) / cf.IMAGENET_STD
        np.testing.assert_allclose(pixels.mean(axis=(1, 2)), expected, rtol=1e-5)

    def test_centre_crop_keeps_the_middle(self) -> None:
        image = Image.new("RGB", (600, 200), (0, 0, 0))
        image.paste((255, 255, 255), (250, 0, 350, 200))  # white band in the centre
        pixels = cf.preprocess(image)
        centre = pixels[0, :, cf.CROP_SIDE // 2]
        border = pixels[0, :, 0]
        self.assertGreater(centre.mean(), border.mean())  # the band survives, the sides are cut

    def test_every_mode_is_converted(self) -> None:
        color = Image.new("RGB", (64, 48), (200, 50, 20))
        gray = np.full((48, 64), 120, dtype=np.uint8)
        images = {mode: color.convert(mode) for mode in ("RGBA", "P", "L", "LA", "1", "CMYK", "YCbCr")}
        images["I;16"] = Image.fromarray(gray.astype(np.uint16) * 256)
        images["F"] = Image.fromarray(gray.astype(np.float32))
        for mode, image in images.items():
            with self.subTest(mode=mode):
                self.assertEqual(cf.preprocess(image).shape, (3, cf.CROP_SIDE, cf.CROP_SIDE))

    def test_empty_image_is_rejected(self) -> None:
        empty = Image.new("RGB", (0, 0))
        with self.assertRaisesRegex(ValueError, "empty"):
            cf.preprocess(empty)


class EmbeddingTests(unittest.TestCase):
    def test_batches_keep_the_order(self) -> None:
        calls = []

        def embed(batch: np.ndarray) -> np.ndarray:
            calls.append(len(batch))
            return batch[:, 0, 0, :1]  # first pixel of the red channel identifies each image

        images = [Image.new("RGB", (40, 40), (index * 10, 0, 0)) for index in range(20)]
        progress = []
        features = cf.embed_images(images, embed, lambda *args: progress.append(args))
        self.assertEqual(calls, [16, 4])
        self.assertTrue(np.all(np.diff(features[:, 0]) > 0))  # same order as the input
        self.assertEqual(progress[-1][0], 20)

    def test_paths_cancel_and_unreadable_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.png"
            texture("dots", 1).save(good)
            broken = Path(tmp) / "broken.png"
            broken.write_bytes(b"not an image")
            self.assertEqual(cf.embed_images([good, str(good)], fake_embed).shape, (2, 4))
            with self.assertRaises(TaskCancelled):
                cf.embed_images([good], fake_embed, cancelled=lambda: True)
            with self.assertRaisesRegex(ValueError, "Could not read .*broken.png"):
                cf.embed_images([good, broken], fake_embed)


class ClassifierTests(unittest.TestCase):
    def test_linear_svm_on_separable_features(self) -> None:
        rng = np.random.default_rng(0)
        centres = np.eye(3) * 5 + 1
        features = np.vstack([rng.normal(c, 0.3, (10, 3)) for c in centres])
        labels = np.repeat([0, 1, 2], 10)
        classifier, cv_accuracy = cf.train_embedding_classifier(features, labels)
        self.assertEqual(cv_accuracy, 1.0)
        self.assertIn(classifier.c, bm.SVM_C_GRID)
        probabilities = classifier.predict_proba(centres)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), [0, 1, 2])
        # L2 normalisation inside: scaling a feature vector does not change its prediction
        np.testing.assert_allclose(classifier.predict_proba(centres * 7), probabilities)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name) / "data"
        for kind in ("checker", "dots", "stripes"):
            (root / kind).mkdir(parents=True)
            for index in range(8):
                texture(kind, seed=index + 100 * len(kind)).save(root / kind / f"{kind}_{index:02d}.png")
        cls.model = bm.train_bow_model(root, FAST)
        cls.comparison = cf.compare_with_cnn(cls.model, fake_embed)

    @classmethod
    def tearDownClass(cls) -> None:
        bm.clear_model()
        cls._tmp.cleanup()

    def tearDown(self) -> None:
        bm.clear_model()

    def test_uses_the_bow_split_and_reproduces_its_accuracy(self) -> None:
        comparison = self.comparison
        self.assertEqual(comparison.test_paths, self.model.test_paths)
        np.testing.assert_array_equal(comparison.test_labels, self.model.test_labels)
        recomputed = float(np.mean(comparison.bow_predicted == comparison.test_labels))
        self.assertAlmostEqual(recomputed, self.model.metrics["accuracy"])
        self.assertEqual(comparison.cnn_dimensions, 4)
        self.assertEqual(comparison.bow_dimensions, bm.pyramid_length(6, 1))
        self.assertEqual(np.asarray(comparison.cnn_metrics["confusion"]).sum(), len(self.model.test_paths))

    def test_disagreements_and_details_are_consistent(self) -> None:
        comparison = self.comparison
        differ = int(np.sum(comparison.bow_predicted != comparison.cnn_predicted))
        self.assertEqual(len(comparison.disagreements()), differ)
        details = comparison_details(comparison)
        counted = sum(details[key] for key in ("Both right", "Only BoW right", "Only CNN right", "Both wrong"))
        self.assertEqual(counted, len(comparison.test_paths))
        self.assertIn("Recall stripes", details)

    def test_classify_one_image(self) -> None:
        ranked = cf.classify_with_cnn(self.comparison.classifier, fake_embed, texture("stripes", 999), self.model.classes)
        self.assertEqual(len(ranked), 3)
        self.assertAlmostEqual(sum(p for _, p in ranked), 1)
        self.assertEqual(ranked, sorted(ranked, key=lambda item: -item[1]))

    def test_render(self) -> None:
        image = render_comparison(self.comparison)
        self.assertEqual(image.mode, "RGB")
        self.assertGreater(image.width, 500)

    def test_tool_without_pytorch_explains_how_to_install(self) -> None:
        tool, document = CnnCompareTool(), ImageDocument()
        with patch("forensics_app.tools.cnn_tools.torch_available", return_value=False), \
             self.assertRaisesRegex(ValueError, "requirements-cnn.txt"):
            tool.run(None, document)

    def test_tool_without_bow_model(self) -> None:
        tool, document = CnnCompareTool(), ImageDocument()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(bm, "DEFAULT_MODEL_PATH", Path(tmp) / "missing.joblib"), \
             patch("forensics_app.tools.cnn_tools.torch_available", return_value=True), \
             self.assertRaisesRegex(ValueError, "Build Codebook"):
            tool.run(None, document)

    def test_tool_compares_and_classifies_the_open_image(self) -> None:
        bm.set_model(self.model)
        document = ImageDocument()
        document.load(self.model.test_paths[0])
        with patch("forensics_app.tools.cnn_tools.torch_available", return_value=True), \
             patch("forensics_app.tools.cnn_tools.load_resnet18", return_value=fake_embed):
            result = CnnCompareTool().run(None, document)
        self.assertTrue(result.preview_only)
        self.assertIn("BoW test accuracy", result.details)
        self.assertIn("CNN test accuracy", result.details)
        self.assertTrue(result.details["Current image CNN"].startswith("p("))
        self.assertIn("(test)", result.details["Current image"])

    def test_tool_without_an_open_image(self) -> None:
        bm.set_model(self.model)
        with patch("forensics_app.tools.cnn_tools.torch_available", return_value=True), \
             patch("forensics_app.tools.cnn_tools.load_resnet18", return_value=fake_embed):
            result = CnnCompareTool().run(None, ImageDocument())
        self.assertNotIn("Current image", result.details)

    def test_tool_cancel(self) -> None:
        bm.set_model(self.model)
        with patch("forensics_app.tools.cnn_tools.torch_available", return_value=True), \
             patch("forensics_app.tools.cnn_tools.run_with_progress", side_effect=TaskCancelled()):
            self.assertIsNone(CnnCompareTool().run(object(), ImageDocument()))

    def test_registered(self) -> None:
        self.assertIn("cnn_compare", [tool.tool_id for tool in build_tool_registry().all()])


@unittest.skipUnless(cf.torch_available(), "PyTorch is not installed (optional CNN dependency)")
class RealResnetTests(unittest.TestCase):
    def test_resnet18_gives_512_non_negative_features(self) -> None:
        embed = cf.load_resnet18(pretrained=False)  # random weights: no download needed
        features = cf.embed_images([texture("dots", 1), texture("stripes", 2)], embed)
        self.assertEqual(features.shape, (2, 512))
        self.assertTrue((features >= 0).all())  # ReLU + average pooling


if __name__ == "__main__":
    unittest.main()
