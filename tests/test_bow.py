"""Tests for Set 7: Bag of Visual Words (codebook, histograms, chi-squared SVM, tools)."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools import BowClassifyTool, BowCodebookTool, BowHistogramTool, build_tool_registry
from forensics_app.tools import bow_model as bm
from forensics_app.tools.bow_dialogs import parse_k
from forensics_app.tools.bow_render import display_copy, render_codebook, render_confusion, render_word_histogram, word_tile
from forensics_app.tools.bow_tools import TRAIN_WARNING, model_details
from forensics_app.tools.dialogs import TaskCancelled

# Dense features on small images keep the end-to-end tests fast and deterministic
FAST = bm.BowSettings(detector="dense", k=6, test_ratio=0.25, pyramid_levels=1, max_side=96)


def texture(kind: str, seed: int, size: int = 96) -> Image.Image:
    """Synthetic class images: stripes, dots or a checkerboard, with random phase and noise."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size]
    period = rng.integers(10, 14)
    shift = rng.integers(0, period)
    if kind == "stripes":
        pattern = ((rows + shift) // (period // 2)) % 2
    elif kind == "checker":
        pattern = (((rows + shift) // period) + ((cols + shift) // period)) % 2
    else:  # dots
        pattern = (((rows + shift) % period - period / 2) ** 2 + ((cols + shift) % period - period / 2) ** 2) < 9
    pixels = 0.2 + 0.6 * pattern + rng.normal(0, 0.05, (size, size))
    return Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def make_dataset(root: Path, per_class: int = 8) -> Path:
    for kind in ("checker", "dots", "stripes"):
        folder = root / kind
        folder.mkdir()
        for index in range(per_class):
            texture(kind, seed=index + 100 * len(kind)).save(folder / f"{kind}_{index:02d}.png")
    (root / ".hidden").mkdir()  # must be ignored
    (root / "__MACOSX").mkdir()
    return root


class SettingsAndDatasetTests(unittest.TestCase):
    def test_settings_reject_invalid_values(self) -> None:
        for kwargs in ({"detector": "orb"}, {"k": 1}, {"test_ratio": 0.9}, {"pyramid_levels": 3}, {"norm": "l3"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bm.BowSettings(**kwargs)

    def test_default_k_is_ten_per_class(self) -> None:
        self.assertEqual(bm.BowSettings().k_for(3), 30)
        self.assertEqual(bm.BowSettings(k=12).k_for(3), 12)

    def test_parse_k(self) -> None:
        self.assertIsNone(parse_k("  "))
        self.assertEqual(parse_k("40"), 40)
        for bad in ("abc", "1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_k(bad)

    def test_list_dataset_reads_class_folders_and_skips_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = bm.list_dataset(make_dataset(Path(tmp), per_class=3))
        self.assertEqual(list(dataset), ["checker", "dots", "stripes"])
        self.assertTrue(all(len(paths) == 3 for paths in dataset.values()))

    def test_list_dataset_needs_two_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "only").mkdir()
            texture("dots", 0).save(Path(tmp) / "only" / "a.png")
            with self.assertRaisesRegex(ValueError, "two class subfolders"):
                bm.list_dataset(tmp)

    def test_split_rejects_classes_left_with_one_training_image(self) -> None:
        dataset = {"big": [Path(f"big/{i}.png") for i in range(10)], "tiny": [Path("tiny/0.png"), Path("tiny/1.png")]}
        bm.split_dataset(dataset, 0.2, seed=0)  # 20 %: both tiny images stay in train
        with self.assertRaisesRegex(ValueError, "fewer than 2 training images: tiny"):
            bm.split_dataset(dataset, 0.5, seed=0)  # 50 %: one tiny image goes to test

    def test_default_model_path_does_not_depend_on_the_working_folder(self) -> None:
        self.assertTrue(bm.DEFAULT_MODEL_PATH.is_absolute())
        self.assertTrue(bm.DEFAULT_MODEL_PATH.is_relative_to(Path.home()))

    def test_split_is_stratified_disjoint_and_reproducible(self) -> None:
        dataset = {name: [Path(f"{name}/{i}.png") for i in range(n)] for name, n in (("a", 20), ("b", 10))}
        train_p, train_y, test_p, test_y = bm.split_dataset(dataset, 0.2, seed=3)
        self.assertFalse(set(train_p) & set(test_p))
        self.assertEqual(len(train_p) + len(test_p), 30)
        self.assertEqual(sorted(test_y), [0, 0, 0, 0, 1, 1])  # 20 % of each class
        self.assertEqual(bm.split_dataset(dataset, 0.2, seed=3)[2], test_p)


class FeatureTests(unittest.TestCase):
    def test_sift_features_are_128_dimensional(self) -> None:
        features, gray = bm.extract_features(Image.fromarray(data.camera()), bm.BowSettings(max_side=200))
        self.assertGreater(len(features), 10)
        self.assertEqual(features.descriptors.shape[1], 128)
        self.assertEqual(features.shape, (gray.height, gray.width))
        self.assertLessEqual(max(gray.size), 200)

    def test_dense_features_form_a_grid(self) -> None:
        features, _ = bm.extract_features(texture("dots", 1), FAST)
        self.assertEqual(features.descriptors.shape[1], bm.descriptor_size("dense"))
        self.assertEqual(len(np.unique(np.diff(np.unique(features.positions[:, 0])))), 1)

    def test_every_image_mode_is_converted(self) -> None:
        color = Image.fromarray(data.astronaut()).resize((160, 160))
        gray = np.asarray(Image.fromarray(data.camera()).resize((160, 160)))
        images = {mode: color.convert(mode) for mode in ("RGB", "RGBA", "P", "L", "LA", "1", "CMYK", "I")}
        images["I;16"] = Image.fromarray(gray.astype(np.uint16) * 256)
        images["F"] = Image.fromarray(gray.astype(np.float32) / 255)
        for mode, image in images.items():
            for detector in ("sift", "dense"):
                with self.subTest(mode=mode, detector=detector):
                    features, working = bm.extract_features(image, bm.BowSettings(detector=detector))
                    self.assertEqual(working.mode, "L")
                    self.assertGreater(len(features), 0)
                    self.assertEqual(features.descriptors.shape[1], bm.descriptor_size(detector))
                    self.assertEqual(features.shape, (working.height, working.width))
            with self.subTest(mode=mode, part="display"):
                shown = display_copy(image, working)
                self.assertEqual(shown.shape, (working.height, working.width, 3))
                self.assertEqual(shown.dtype, np.uint8)

    def test_flat_image_gives_no_features_without_crashing(self) -> None:
        features, _ = bm.extract_features(Image.new("L", (80, 80), 128), bm.BowSettings())
        self.assertEqual(len(features), 0)
        histogram = bm.encode(features, bm.assign_words(features.descriptors, np.zeros((4, 128))), levels=1)
        self.assertEqual(histogram.shape, (20,))
        self.assertFalse(np.isnan(bm.normalize(histogram)).any())


class VocabularyTests(unittest.TestCase):
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

    def test_codebook_shape_and_seed(self) -> None:
        rng = np.random.default_rng(0)
        descriptors = np.vstack([rng.normal(c, 0.5, (50, 2)) for c in self.centers]).astype(np.float32)
        first = bm.build_codebook(descriptors, 3, seed=1)
        self.assertEqual(first.shape, (3, 2))
        np.testing.assert_allclose(first, bm.build_codebook(descriptors, 3, seed=1))
        # every true cluster centre is found
        for centre in self.centers:
            self.assertLess(np.linalg.norm(first - centre, axis=1).min(), 0.5)

    def test_codebook_needs_enough_descriptors(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer than K"):
            bm.build_codebook(np.zeros((2, 4), np.float32), 3)

    def test_hard_assignment_picks_the_nearest_word(self) -> None:
        descriptors = np.array([[1.0, 1.0], [9.0, 1.0], [0.0, 8.0], [9.5, 0.0]])
        weights = bm.assign_words(descriptors, self.centers, "hard")
        np.testing.assert_array_equal(weights.argmax(axis=1), [0, 1, 2, 1])
        np.testing.assert_array_equal(weights.sum(axis=1), 1)

    def test_soft_assignment_rows_sum_to_one_and_favour_the_nearest(self) -> None:
        descriptors = np.array([[4.0, 0.0], [1.0, 1.0]])
        weights = bm.assign_words(descriptors, self.centers, "soft", sigma=5.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_array_equal(weights.argmax(axis=1), [0, 0])
        self.assertGreater(weights[0, 1], 0.2)  # a border descriptor also votes for its neighbour

    def test_pyramid_length_and_counts(self) -> None:
        features = bm.LocalFeatures(
            positions=np.array([[1.0, 1.0], [1.0, 9.0], [9.0, 9.0]]),
            scales=np.ones(3),
            descriptors=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
            shape=(10, 10),
        )
        weights = bm.assign_words(features.descriptors, self.centers)
        self.assertEqual(bm.encode(features, weights, levels=0).tolist(), [1, 1, 1])
        self.assertEqual(len(bm.encode(features, weights, levels=2)), bm.pyramid_length(3, 2))
        self.assertEqual(bm.pyramid_length(3, 2), 3 * 21)
        level1 = bm.encode(features, weights, levels=1)[3:].reshape(2, 2, 3)
        self.assertEqual(level1[0, 0].argmax(), 0)  # top-left feature is word 0
        self.assertEqual(level1[1, 1].argmax(), 2)  # bottom-right feature is word 2

    def test_idf_downweights_common_words_and_stays_positive(self) -> None:
        counts = np.array([[3, 1, 0], [2, 0, 0], [5, 0, 1], [1, 0, 0]])
        idf = bm.idf_weights(counts)
        self.assertLess(idf[0], idf[1])
        self.assertTrue((idf > 0).all())
        self.assertAlmostEqual(idf[0], np.log(5 / 5) + 1)
        self.assertAlmostEqual(idf[1], np.log(5 / 2) + 1)

    def test_normalisations(self) -> None:
        vector = np.array([3.0, 4.0, 0.0])
        self.assertAlmostEqual(bm.normalize(vector, "l1").sum(), 1)
        self.assertAlmostEqual(np.linalg.norm(bm.normalize(vector, "l2")), 1)
        np.testing.assert_array_equal(bm.normalize(np.zeros(3)), 0)

    def test_same_proportions_give_the_same_vector(self) -> None:
        settings = bm.BowSettings(weighting="tf")
        a = bm.to_vectors(np.array([[10.0, 20.0, 30.0]]), settings, None)
        b = bm.to_vectors(np.array([[20.0, 40.0, 60.0]]), settings, None)
        np.testing.assert_allclose(a, b)


class ClassifierTests(unittest.TestCase):
    def test_chi2_distance(self) -> None:
        distance = bm.chi2_distances(np.array([[0.5, 0.5]]), np.array([[1.0, 0.0], [0.5, 0.5]]))
        np.testing.assert_allclose(distance, [[0.25 / 1.5 + 0.25 / 0.5, 0.0]])

    def test_chi2_svm_separates_histograms_with_probabilities(self) -> None:
        rng = np.random.default_rng(0)
        prototypes = np.eye(3) * 0.7 + 0.1
        vectors = bm.normalize(np.vstack([rng.dirichlet(p * 30, 12) for p in prototypes]))
        labels = np.repeat([0, 1, 2], 12)
        classifier, cv_accuracy = bm.train_classifier(vectors, labels)
        self.assertGreater(cv_accuracy, 0.9)
        self.assertIn(classifier.c, bm.SVM_C_GRID)
        probabilities = classifier.predict_proba(bm.normalize(prototypes))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), [0, 1, 2])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "data"
        cls.root.mkdir()
        make_dataset(cls.root, per_class=8)
        cls.progress: list[tuple[int, int, str]] = []
        cls.model = bm.train_bow_model(cls.root, FAST, lambda *args: cls.progress.append(args))

    @classmethod
    def tearDownClass(cls) -> None:
        bm.clear_model()
        cls._tmp.cleanup()

    def tearDown(self) -> None:
        bm.clear_model()

    def test_model_summary(self) -> None:
        model = self.model
        self.assertEqual(model.classes, ["checker", "dots", "stripes"])
        self.assertEqual(model.centers.shape, (6, bm.descriptor_size("dense")))
        self.assertEqual(len(model.test_paths), 6)
        self.assertFalse(set(model.train_paths) & set(model.test_paths))
        self.assertGreaterEqual(model.metrics["accuracy"], 0.8)
        self.assertEqual(np.asarray(model.metrics["confusion"]).sum(), 6)
        self.assertEqual(len(model.exemplars), 6)
        self.assertTrue(self.progress)

    def test_codebook_uses_train_descriptors_only(self) -> None:
        train_count = sum(
            len(bm.extract_features(bm.load_image(Path(p)), FAST)[0]) for p in self.model.train_paths
        )
        self.assertEqual(self.model.metrics["train_descriptors"], train_count)

    def test_exemplars_come_from_training_images(self) -> None:
        train_names = {Path(p).name for p in self.model.train_paths}
        for exemplar in self.model.exemplars:
            self.assertTrue(set(exemplar.sources) <= train_names)
            self.assertEqual(exemplar.patches.shape[1:], (bm.PATCH_SIDE, bm.PATCH_SIDE))
            if exemplar.class_share.sum():
                self.assertAlmostEqual(exemplar.class_share.sum(), 1)

    def test_classifies_a_new_image(self) -> None:
        analysis = bm.describe_image(self.model, texture("stripes", seed=999))
        self.assertFalse(analysis.in_train)
        self.assertAlmostEqual(analysis.word_frequency.sum(), 1)
        self.assertEqual(len(analysis.vector), bm.pyramid_length(6, 1))
        ranked = bm.classify(self.model, analysis)
        self.assertEqual(ranked[0][0], "stripes")
        self.assertAlmostEqual(sum(p for _, p in ranked), 1)
        self.assertEqual(analysis.neighbours[0][1], "stripes")

    def test_detects_training_test_and_external_images(self) -> None:
        train, test = self.model.train_paths[0], self.model.test_paths[0]
        self.assertEqual(bm.image_split(self.model, train), "train")
        self.assertTrue(bm.describe_image(self.model, bm.load_image(Path(train)), train).in_train)
        self.assertEqual(bm.image_split(self.model, test), "test")
        self.assertEqual(bm.image_split(self.model, "/elsewhere/photo.png"), "external")
        self.assertEqual(bm.image_split(self.model, None), "external")
        # a relative path to a training file is still recognised
        # (Path.relative_to(walk_up=True) needs Python 3.12; os.path.relpath works on 3.10)
        try:
            relative = os.path.relpath(train)
        except ValueError:  # Windows: temp folder and working folder on different drives
            self.skipTest("no relative path between drives")
        self.assertEqual(bm.image_split(self.model, relative), "train")

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.model.save(Path(tmp) / "sub" / "model.joblib")
            loaded = bm.BowModel.load(path)
        image = texture("dots", seed=5)
        np.testing.assert_allclose(
            bm.classify(loaded, bm.describe_image(loaded, image))[0][1],
            bm.classify(self.model, bm.describe_image(self.model, image))[0][1],
        )

    def test_load_rejects_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.joblib"
            bm.joblib.dump({"version": -1}, path)
            with self.assertRaisesRegex(ValueError, "not a BoW model"):
                bm.BowModel.load(path)

    def test_choose_k_by_cv_returns_a_candidate(self) -> None:
        features = [bm.extract_features(bm.load_image(Path(p)), FAST)[0] for p in self.model.train_paths]
        best, results = bm.choose_k_by_cv(features, self.model.train_labels, [3, 6], FAST)
        self.assertIn(best, (3, 6))
        self.assertEqual([k for k, _ in results], [3, 6])

    def test_unreadable_image_names_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_dataset(root, per_class=8)
            (root / "dots" / "broken.png").write_bytes(b"not really a png")
            with self.assertRaisesRegex(ValueError, "Could not read .*broken.png"):
                bm.train_bow_model(root, FAST)

    def test_save_checks_what_was_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(bm.BowModel, "load", return_value=bm.BowModel.load(self.model.save(Path(tmp) / "a.joblib"))):
            other = bm.train_bow_model(self.root, bm.BowSettings(**{**FAST.__dict__, "k": 4}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                other.save(Path(tmp) / "b.joblib")

    def test_training_can_be_cancelled(self) -> None:
        with self.assertRaises(TaskCancelled):
            bm.train_bow_model(self.root, FAST, cancelled=lambda: True)

    def test_renders(self) -> None:
        image = texture("checker", seed=7)
        analysis = bm.describe_image(self.model, image)
        for rendered in (render_codebook(self.model), render_confusion(self.model),
                         render_word_histogram(image, analysis, self.model)):
            self.assertEqual(rendered.mode, "RGB")
            self.assertGreater(rendered.width, 100)
        self.assertEqual(word_tile(np.zeros((0, bm.PATCH_SIDE, bm.PATCH_SIDE), np.uint8)).shape,
                         (2 * (bm.PATCH_SIDE + 2), 3 * (bm.PATCH_SIDE + 2)))

    def test_model_details(self) -> None:
        details = model_details(self.model)
        self.assertEqual(details["Vocabulary size K"], 6)
        self.assertIn("Test accuracy", details)
        self.assertIn("Confusion stripes", details)

    # ----------------------------------------------------------------- tools

    def test_tools_without_model_explain_what_to_do(self) -> None:
        document = ImageDocument()
        document.current = texture("dots", 1)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(bm, "DEFAULT_MODEL_PATH", Path(tmp) / "missing.joblib"):
            for tool in (BowHistogramTool(), BowClassifyTool()):
                with self.subTest(tool=tool.tool_id), self.assertRaisesRegex(ValueError, "Build Codebook"):
                    tool.run(None, document)

    def test_histogram_and_classify_tools(self) -> None:
        bm.set_model(self.model)
        document = ImageDocument()
        document.current = texture("stripes", seed=321)
        histogram = BowHistogramTool().run(None, document)
        self.assertTrue(histogram.preview_only)
        self.assertIn("Top word #1", histogram.details)
        result = BowClassifyTool().run(None, document)
        self.assertEqual(result.details["Prediction"], "stripes")
        self.assertTrue(result.message.startswith("p(stripes) = "))
        self.assertTrue(result.preview_only)

    def test_classify_tool_warns_on_training_images(self) -> None:
        bm.set_model(self.model)
        document = ImageDocument()
        document.load(self.model.train_paths[0])
        result = BowClassifyTool().run(None, document)
        self.assertEqual(result.details["Warning"], TRAIN_WARNING)
        self.assertTrue(result.details["Data split"].startswith("TRAIN"))
        self.assertIn("optimistic", result.message)
        with patch("forensics_app.tools.bow_tools.show_classify_window", return_value=None) as window:
            BowClassifyTool().run(object(), document)
        self.assertEqual(window.call_args.args[4], TRAIN_WARNING)  # the window shows the warning

    def test_test_images_have_no_warning(self) -> None:
        bm.set_model(self.model)
        document = ImageDocument()
        document.load(self.model.test_paths[0])
        result = BowHistogramTool().run(None, document)
        self.assertNotIn("Warning", result.details)
        self.assertTrue(result.details["Data split"].startswith("TEST"))

    def test_images_without_features_are_not_classified(self) -> None:
        bm.set_model(self.model)
        document = ImageDocument()
        document.current = Image.new("L", (20, 20), 128)  # smaller than one DAISY descriptor
        analysis = bm.describe_image(self.model, document.current)
        self.assertEqual(len(analysis.features), 0)
        with self.assertRaisesRegex(ValueError, "cannot be classified"):
            bm.classify(self.model, analysis)
        histogram = BowHistogramTool().run(None, document)  # still shows the (empty) histogram
        self.assertIn("No local features", histogram.message)
        with patch("forensics_app.tools.bow_tools.show_classify_window") as window, \
             self.assertRaisesRegex(ValueError, "cannot be classified"):
            BowClassifyTool().run(object(), document)
        window.assert_not_called()  # the error comes before the window opens

    def test_evaluate_matches_classify(self) -> None:
        vectors = self.model.classifier.train_vectors[:5]
        metrics = bm.evaluate(self.model.classifier, vectors, self.model.train_labels[:5], self.model.classes)
        probabilities = self.model.classifier.predict_proba(vectors)
        expected = self.model.classifier.classes_[probabilities.argmax(axis=1)]
        self.assertAlmostEqual(metrics["accuracy"], float(np.mean(expected == self.model.train_labels[:5])))

    def test_classify_window_closed_means_cancel(self) -> None:
        bm.set_model(self.model)
        document = ImageDocument()
        document.current = texture("dots", seed=4)
        with patch("forensics_app.tools.bow_tools.show_classify_window", return_value=None):
            self.assertIsNone(BowClassifyTool().run(object(), document))

    def test_codebook_tool_trains_saves_and_sets_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
             patch("forensics_app.tools.bow_tools.filedialog.askdirectory", return_value=str(self.root)), \
             patch("forensics_app.tools.bow_tools.ask_bow_settings", return_value=FAST), \
             patch("forensics_app.tools.bow_tools.DEFAULT_MODEL_PATH", Path(tmp) / "m.joblib"):
            result = BowCodebookTool().run(None, ImageDocument())
            self.assertTrue((Path(tmp) / "m.joblib").is_file())
        self.assertTrue(result.preview_only)
        self.assertIn("Test accuracy", result.details)
        self.assertIs(bm.get_model(), bm._session["model"])

    def test_codebook_tool_keeps_the_model_when_saving_fails(self) -> None:
        with patch("forensics_app.tools.bow_tools.filedialog.askdirectory", return_value=str(self.root)), \
             patch("forensics_app.tools.bow_tools.ask_bow_settings", return_value=FAST), \
             patch.object(bm.BowModel, "save", side_effect=OSError("read-only folder")):
            result = BowCodebookTool().run(None, ImageDocument())
        self.assertIn("could not be saved", result.message)
        self.assertEqual(result.details["Model NOT saved"], "read-only folder")
        self.assertNotIn("Model saved to", result.details)
        self.assertEqual(bm.get_model().k, FAST.k)  # still usable in this session

    def test_codebook_tool_cancel(self) -> None:
        with patch("forensics_app.tools.bow_tools.filedialog.askdirectory", return_value=""):
            self.assertIsNone(BowCodebookTool().run(None, ImageDocument()))
        with patch("forensics_app.tools.bow_tools.filedialog.askdirectory", return_value=str(self.root)), \
             patch("forensics_app.tools.bow_tools.ask_bow_settings", return_value=None):
            self.assertIsNone(BowCodebookTool().run(None, ImageDocument()))

    def test_registry_contains_bow_tools(self) -> None:
        ids = [tool.tool_id for tool in build_tool_registry().all()]
        for tool_id in ("bow_codebook", "bow_histogram", "bow_classify"):
            self.assertIn(tool_id, ids)


class ReviewFixesTests(unittest.TestCase):
    """Regression tests for problems found while reviewing the BoW core."""

    def test_kmeans_finds_obvious_groups_for_every_seed(self) -> None:
        # MiniBatchKMeans split the group near (0, 0) and merged the other two with seed 0
        points = np.array([[0, 0], [1, 0], [0, 1], [9, 9], [9, 10], [10, 9], [0, 1], [20, 0], [19, 1]], np.float32)
        for seed in range(5):
            with self.subTest(seed=seed):
                centers = bm.build_codebook(points, 3, seed=seed)
                for expected in ([0.25, 0.5], [9.33, 9.33], [19.5, 0.5]):
                    self.assertLess(np.linalg.norm(centers - expected, axis=1).min(), 0.1)

    def test_gamma_ignores_the_zero_diagonal(self) -> None:
        distances = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
        self.assertAlmostEqual(bm.chi2_gamma(distances), 1 / 2.0)  # mean of 1, 3, 2 (not 12/9)
        self.assertAlmostEqual(bm.chi2_gamma(np.array([[2.0, 4.0]])), 1 / 3.0)  # not square: plain mean

    def test_idf_still_works_with_soft_assignment(self) -> None:
        centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        def image(descriptors: list[list[float]]) -> bm.LocalFeatures:
            d = np.array(descriptors, np.float32)
            return bm.LocalFeatures(np.full((len(d), 2), 5.0), np.ones(len(d)), d, (10, 10))
        # word 0 in every image, word 2 in only one
        features = [image([[0, 0], [10, 0]]), image([[1, 0], [9, 0]]), image([[0, 1], [0, 10]])]
        descriptors = np.vstack([f.descriptors for f in features])
        settings = bm.BowSettings(assignment="soft", soft_sigma=5.0, pyramid_levels=0)
        _, idf, _ = bm._vectors_for(features, centers, settings, descriptors)
        self.assertLess(idf[0], idf[2])  # soft votes reach every word, presence must not
        hard = bm._vectors_for(features, centers, bm.BowSettings(pyramid_levels=0), descriptors)[1]
        np.testing.assert_allclose(idf, hard)

    def test_tiny_images_give_no_features_instead_of_crashing(self) -> None:
        for size in ((1, 1), (5, 5), (3, 200), (15, 40)):
            for detector in ("sift", "dense"):
                with self.subTest(size=size, detector=detector):
                    image = Image.fromarray((np.random.default_rng(0).random(size) * 255).astype(np.uint8))
                    features, _ = bm.extract_features(image, bm.BowSettings(detector=detector))
                    self.assertEqual(len(features), 0)

    def test_crop_patch_at_the_border_stays_square(self) -> None:
        gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
        for row, col in ((9.6, 9.6), (-0.4, 0.0), (0.0, 9.9)):
            self.assertEqual(bm._crop_patch(gray, row, col, 3).shape, (bm.PATCH_SIDE, bm.PATCH_SIDE))

    def test_load_reports_unreadable_or_incomplete_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.joblib"
            broken.write_bytes(b"this is not a model")
            with self.assertRaisesRegex(ValueError, "Could not read the BoW model"):
                bm.BowModel.load(broken)
            partial = Path(tmp) / "partial.joblib"
            bm.joblib.dump({"version": bm.MODEL_VERSION, "classes": ["a"]}, partial)
            with self.assertRaisesRegex(ValueError, "not a complete BoW model"):
                bm.BowModel.load(partial)


if __name__ == "__main__":
    unittest.main()
