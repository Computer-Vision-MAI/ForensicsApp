"""Set 11: segmentation metrics, the tray dataset, the U-Net figures and tools (no PyTorch needed)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.tools import segmentation_metrics as sm
from forensics_app.tools import tray_dataset as td
from forensics_app.tools import unet_model as um
from forensics_app.tools import unet_tools as ut
from forensics_app.tools.unet_render import class_colors, error_map, overlay, render_comparison, render_passport
from forensics_app.tools.yolo_dialogs import Field, parse_fields


def make_dataset(root: Path, n_train: int = 12, n_test: int = 2, size=(32, 52), rgb_masks: bool = False) -> Path:
    """A tiny TrayDataset: a tray (1) with a food square (class 7) whose position depends on the image."""
    h, w = size
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("test", n_test)):
        xdir, ydir = (root / d for d in td.SPLIT_FOLDERS[split])
        xdir.mkdir(parents=True)
        ydir.mkdir(parents=True)
        for i in range(n):
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[4:h - 4, 4:w - 4] = 1
            top, left = 6 + i % 5, 8 + (2 * i) % 20
            mask[top:top + 8, left:left + 10] = 7
            mask[2, 2] = 2  # cutlery
            image = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
            image[mask == 1] = (200, 180, 140)
            image[mask == 7] = (220, 90, 30)
            name = f"{1000 + i}a{i:02d}" if split == "train" else f"{2000 + i}a"
            Image.fromarray(image).save(xdir / f"{name}.jpg", quality=95)
            stored = np.dstack([mask] * 3) if rgb_masks else mask
            Image.fromarray(stored).save(ydir / f"{name}.png")
    (root / "classes.csv").write_text(
        "﻿_id,_class,_name\n" + "\n".join(f"{i},c{i},{name}" for i, name in enumerate(td.DEFAULT_CLASS_NAMES)),
        encoding="utf-8")
    return root


class MetricTests(unittest.TestCase):
    def test_notebook_example(self) -> None:
        p = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        g = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(sm.binary_dice(p, g), 0.5)
        self.assertAlmostEqual(sm.iou_scores(p, g, 2)[1], 2 / 6)
        self.assertAlmostEqual(float(np.mean(p == g)), 0.75)

    def test_iou_dice_identity(self) -> None:
        for d in (0.0, 0.3, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(float(sm.dice_from_iou(sm.iou_from_dice(d))), d)
        rng = np.random.default_rng(1)
        p, g = rng.integers(0, 2, (20, 20)), rng.integers(0, 2, (20, 20))
        dice, iou = sm.dice_scores(p, g, 2)[1], sm.iou_scores(p, g, 2)[1]
        self.assertAlmostEqual(float(sm.iou_from_dice(dice)), iou)
        self.assertGreaterEqual(dice, iou)

    def test_empty_masks(self) -> None:
        zeros = np.zeros((4, 4), dtype=int)
        self.assertEqual(sm.binary_dice(zeros, zeros), 1.0)  # notebook: eps makes it 1
        self.assertTrue(np.isnan(sm.dice_scores(zeros, zeros, 3)[2]))
        ones = np.ones((4, 4), dtype=int)
        self.assertEqual(sm.binary_dice(ones, zeros), 0.0)

    def test_confusion_matrix(self) -> None:
        gt = np.array([0, 0, 1, 2, 2])
        pred = np.array([0, 1, 1, 2, 0])
        expected = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        np.testing.assert_array_equal(sm.confusion_matrix(pred, gt, 3), expected)

    def test_threshold_sweep_and_masks(self) -> None:
        probs = np.array([[[0.2, 0.6], [0.8, 0.4]]])
        gts = np.array([[[0, 1], [1, 0]]])
        np.testing.assert_array_equal(sm.threshold_masks(probs[0], 0.5), [[0, 1], [1, 0]])
        sweep = sm.threshold_sweep(probs, gts, [0.1, 0.5, 0.9])
        self.assertAlmostEqual(sweep[1], 1.0)
        self.assertLess(sweep[0], 1.0)
        self.assertEqual(sweep[2], 0.0)

    def test_evaluate_masks_binary_and_multiclass(self) -> None:
        gt = np.zeros((2, 4, 4), dtype=np.uint8)
        gt[:, :2, :2] = 1
        pred = gt.copy()
        pred[1, 3, 3] = 1
        result = sm.evaluate_masks(pred, gt, ["not food", "food"], ["a", "b"], 0.5)
        self.assertAlmostEqual(result.dice[0], 1.0)
        self.assertAlmostEqual(result.dice[1], 2 * 4 / 9)
        self.assertEqual(result.worst(1)[0], 1)
        self.assertEqual(result.confusion.sum(), 32)
        saved = result.to_dict()
        self.assertEqual(saved["images"], 2)
        back = sm.split_from_dict(saved, ["not food", "food"])
        np.testing.assert_allclose(back["dice"], result.dice, atol=1e-5)

        three = sm.evaluate_masks([np.array([[0, 1], [2, 2]])], [np.array([[0, 1], [2, 1]])], ["bg", "tray", "food"])
        self.assertAlmostEqual(three.class_dice[1], 2 / 3)
        self.assertAlmostEqual(three.class_dice[2], 2 / 3)
        self.assertAlmostEqual(three.dice[0], 2 / 3)  # mean of tray and food


class TrayDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = make_dataset(Path(self.tmp.name) / "TrayDataset", rgb_masks=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_files_classes_and_masks(self) -> None:
        self.assertTrue(td.is_dataset_root(self.root))
        names = td.read_class_names(self.root)
        self.assertEqual(len(names), 43)
        self.assertEqual(names[0], "background")  # the BOM is removed
        pairs = td.split_pairs(self.root, "train")
        self.assertEqual(len(pairs), 12)
        self.assertEqual(pairs[0][0].stem, pairs[0][1].stem)
        mask = td.read_mask(pairs[0][1])
        self.assertEqual(mask.ndim, 2)  # 3 identical channels -> 1
        self.assertEqual(set(np.unique(mask)), {0, 1, 2, 7})

    def test_tasks(self) -> None:
        mask = np.array([[0, 1, 2, 4, 5, 42]])
        np.testing.assert_array_equal(td.make_task("binary").map(mask), [[0, 0, 0, 0, 1, 1]])
        np.testing.assert_array_equal(td.make_task("three").map(mask), [[0, 1, 0, 0, 2, 2]])
        np.testing.assert_array_equal(td.make_task("all").map(mask), mask)
        with self.assertRaises(ValueError):
            td.make_task("nope")

    def test_resize_keeps_class_values(self) -> None:
        loaded = td.load_split(self.root, "train", (16, 24))
        self.assertEqual(loaded.images.shape, (12, 16, 24, 3))
        self.assertTrue(set(np.unique(loaded.masks)) <= {0, 1, 2, 7})  # nearest: no invented classes
        self.assertIs(td.load_split(self.root, "train", (16, 24)), loaded)  # cached

    def test_split_matches_the_notebook(self) -> None:
        train, val = td.train_val_indices(1241)
        self.assertEqual((len(train), len(val)), (1055, 186))
        expected = np.random.RandomState(42).permutation(1241)
        np.testing.assert_array_equal(val, expected[:186])
        self.assertEqual(len(set(train) | set(val)), 1241)

    def test_split_by_tray_keeps_trays_apart(self) -> None:
        names = [f"{1000 + t}a{k:02d}.jpg" for t in range(10) for k in range(5)]
        self.assertEqual(td.tray_id("1001a07.jpg"), "1001a")
        self.assertEqual(td.tray_id("4002a05.JPG"), "4002a")
        train, val = td.split_indices(names, 0.2, 42, by_tray=True)
        train_trays = {td.tray_id(names[i]) for i in train}
        val_trays = {td.tray_id(names[i]) for i in val}
        self.assertEqual(len(val_trays), 2)
        self.assertFalse(train_trays & val_trays)
        self.assertEqual(len(train) + len(val), 50)
        random_train, random_val = td.split_indices(names, 0.2, 42)
        np.testing.assert_array_equal(random_val, td.train_val_indices(50, 0.2, 42)[1])

    def test_find_mask_and_stats(self) -> None:
        image = td.split_pairs(self.root, "test")[0][0]
        self.assertEqual(td.find_mask_for(image).parent.name, "yTest")
        self.assertIsNone(td.find_mask_for(Path(self.tmp.name) / "other.jpg"))
        stats = td.dataset_stats(self.root, "train", 43)
        self.assertEqual(stats.images, 12)
        self.assertEqual(stats.images_with_class[7], 12)
        self.assertTrue(np.all(stats.food_fraction > 0))


def fake_passport(root: Path, folder: Path, task_key: str = "binary") -> um.UnetPassport:
    task = td.make_task(task_key)
    rng = np.random.default_rng(0)
    gts = rng.integers(0, task.num_classes, (5, 8, 8))
    preds = np.where(rng.random((5, 8, 8)) < 0.8, gts, 0)
    val = sm.evaluate_masks(preds, gts, task.class_names, [f"v{i}.jpg" for i in range(5)], 0.5)
    val.sweep_thresholds = np.array([0.3, 0.5, 0.7])
    val.sweep_dice = np.array([0.8, 0.9, 0.85])
    test = sm.evaluate_masks(preds[:2], gts[:2], task.class_names, ["t0.jpg", "t1.jpg"], 0.5)
    settings = um.TrainSettings(root=root, task=task_key, size=(16, 24), epochs=3)
    history = {"train_loss": [1.0, 0.6, 0.4], "val_loss": [0.9, 0.7, 0.6], "val_dice": [0.7, 0.85, 0.9],
               "val_iou": [0.55, 0.74, 0.82], "seconds": [1, 1, 1]}
    data = um.build_passport(folder.name, settings, task, 482737, history, 3, 3, False, 3.0, "cpu", val, test,
                             10, 5, 2)
    um.write_passport(folder, data)
    (folder / "model.pt").write_bytes(b"not used")
    return um.read_passport(folder)


class UnetPassportAndToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = make_dataset(base / "TrayDataset")
        self.models = base / "models"
        self.binary = fake_passport(self.root, self.models / "unet_binary")
        self.three = fake_passport(self.root, self.models / "unet_three", "three")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_passport_round_trip_and_listing(self) -> None:
        self.assertTrue(self.binary.binary)
        self.assertEqual(self.binary.size, (16, 24))
        self.assertEqual(self.three.class_names, ["background", "tray", "food"])
        self.assertEqual(self.binary.task.map(np.array([6])).tolist(), [1])
        listed = um.list_models(self.models)
        self.assertEqual({p.name for p in listed}, {"unet_binary", "unet_three"})
        self.assertIn("val Dice", self.binary.label)
        details = ut.passport_details(self.binary)
        self.assertEqual(details["Parameters"], "482,737")

    def test_names(self) -> None:
        self.assertEqual(um.default_model_name("binary", "bce_dice", False, (128, 208), 15),
                         "unet_binary_bce-dice_noskip_128x208_15e")
        self.assertEqual(um.safe_name("  my model/1 "), "my_model_1")
        self.assertIn("NO skip", um.architecture_summary(16, False, 1))

    def test_figures(self) -> None:
        for passport in (self.binary, self.three):
            n = passport.num_classes
            image = np.full((8, 8, 3), 120, dtype=np.uint8)
            gt = np.zeros((8, 8), dtype=np.uint8)
            gt[2:5, 2:5] = n - 1
            worst = [("v0.jpg", image, gt, np.zeros_like(gt), 0.0)]
            self.assertGreater(render_passport(passport, worst=worst).width, 1000)
        self.assertGreater(render_comparison([self.binary, self.three]).width, 1000)
        self.assertEqual(class_colors(43).shape, (43, 3))
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        pred, gt = np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]])
        errors = error_map(image, pred, gt)
        self.assertEqual(tuple(errors[0, 0]), (46, 158, 79))  # TP green
        self.assertEqual(tuple(errors[0, 1]), (209, 73, 91))  # FP red
        self.assertEqual(tuple(errors[1, 0]), (42, 122, 176))  # FN blue
        self.assertEqual(overlay(image, pred, 2).shape, (2, 2, 3))

    def test_segment_with_ground_truth(self) -> None:
        image_path, mask_path = td.split_pairs(self.root, "test")[0]
        image = td.read_image(image_path)
        truth = self.binary.task.map(td.read_mask(mask_path))

        def perfect(rgb):
            return truth, truth.astype(np.float32), np.zeros((16, 24, 3), np.uint8)

        settings = ut.SegmentSettings(self.binary, image_path.name)
        result = ut.segment(self.binary, image, image_path, settings, segment_fn=perfect)
        self.assertEqual(result.details["Dice"], "1.0000 (food)")
        self.assertIn("test image, unseen", result.details["Image"])
        self.assertTrue(result.preview_only)
        overlay_only = ut.segment(self.binary, image, image_path, ut.SegmentSettings(self.binary, "x", output=1),
                                  segment_fn=perfect)
        self.assertEqual(overlay_only.image.size, image.size)

    def test_segment_without_ground_truth(self) -> None:
        image = Image.new("RGB", (20, 10), "white")

        def nothing(rgb):
            return np.zeros((10, 20), np.uint8), np.zeros((10, 20), np.float32), np.zeros((16, 24, 3), np.uint8)

        result = ut.segment(self.three, image, None, ut.SegmentSettings(self.three, "working"), segment_fn=nothing)
        self.assertIn("no mask found", result.details["Ground truth"])

    def test_validation_names_and_rows(self) -> None:
        names = ut.validation_names(self.binary)
        self.assertEqual(len(names), int(0.15 * 12))
        rng = np.random.default_rng(2)
        gts = rng.integers(0, 2, (4, 6, 6))
        evaluation = sm.evaluate_masks(gts, gts, ["not food", "food"], list("abcd"), 0.5)
        rows = ut.evaluation_rows(evaluation, "Validation", self.binary)
        self.assertEqual(rows["Validation: Dice"], "1.0000 ± 0.0000")
        self.assertIn("Exercise 4", " ".join(rows))
        comparison = ut.comparison_result([self.binary, self.three])
        self.assertIn("unet_binary", comparison.details)

    def test_explore_dataset(self) -> None:
        result = ut.explore_dataset(self.root, "three")
        self.assertIn("Food (classes 5-42)", result.details)
        self.assertEqual(result.details["Images"], "train 12, test 2")

    def test_no_models_message(self) -> None:
        with patch.object(um, "list_models", return_value=[]):
            with self.assertRaises(ValueError):
                ut._models_or_raise()


class FormFieldTests(unittest.TestCase):
    def test_str_and_dir_fields(self) -> None:
        fields = [Field("name", "Name", "str", ""), Field("root", "Folder", "dir", "")]
        values = parse_fields(fields, {"name": "  model  ", "root": "/tmp"})
        self.assertEqual(values["name"], "model")
        self.assertEqual(values["root"], Path("/tmp"))
        self.assertIsNone(parse_fields(fields, {"name": "", "root": ""})["root"])


if __name__ == "__main__":
    unittest.main()
