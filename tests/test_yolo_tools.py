"""Tests for the Set 9 YOLO tools.

A fake model with the same ``predict`` interface as Ultralytics is used, so these tests
need neither Ultralytics, nor PyTorch, nor a network connection.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import yolo_model as ym
from forensics_app.tools import yolo_tools as yt
from forensics_app.tools.yolo_dataset import convert_coco_dataset
from forensics_app.tools.yolo_dialogs import Field, parse_fields
from forensics_app.tools.yolo_eval import evaluate_split


class _Array:
    """Mimics a torch tensor just enough: ``.cpu().numpy()``."""

    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def cpu(self) -> "_Array":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _Boxes:
    def __init__(self, xyxy, cls, conf) -> None:
        self.xyxy, self.cls, self.conf = _Array(xyxy), _Array(cls), _Array(conf)
        self._n = len(cls)

    def __len__(self) -> int:
        return self._n


class _Result:
    def __init__(self, names, boxes) -> None:
        self.names = names
        self.boxes = boxes
        self.speed = {"preprocess": 1.0, "inference": 5.0, "postprocess": 1.0}


class FakeYolo:
    """Returns fixed boxes (in pixels) whatever the image; records the calls."""

    def __init__(self, names: dict[int, str], boxes, classes, scores) -> None:
        self.names = names
        self.boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.classes = np.asarray(classes)
        self.scores = np.asarray(scores, dtype=float)
        self.calls: list[dict] = []

    def predict(self, source, **kwargs):
        self.calls.append({"source": source, **kwargs})
        keep = self.scores >= kwargs.get("conf", 0)
        return [_Result(self.names, _Boxes(self.boxes[keep], self.classes[keep], self.scores[keep]))]


CELLS = {0: "PC", 1: "RBC", 2: "WBC"}


def make_dataset(root: Path) -> Path:
    """A tiny converted blood-cell dataset: 2 train and 2 test images of 64 x 64."""
    categories = [{"id": 0, "name": "cells", "supercategory": "none"},
                  {"id": 1, "name": "PC", "supercategory": "cells"},
                  {"id": 2, "name": "RBC", "supercategory": "cells"},
                  {"id": 3, "name": "WBC", "supercategory": "cells"}]
    for split in ("train", "test"):
        folder = root / "coco" / split
        folder.mkdir(parents=True)
        images, anns = [], []
        for i in range(2):
            name = f"{split}{i}.jpg"
            Image.new("RGB", (64, 64), (230, 200, 200)).save(folder / name)
            images.append({"id": i, "file_name": name, "width": 64, "height": 64})
            anns.append({"id": 2 * i, "image_id": i, "category_id": 2, "bbox": [0, 0, 20, 20]})
            anns.append({"id": 2 * i + 1, "image_id": i, "category_id": 3, "bbox": [30, 30, 30, 30]})
        (folder / "_annotations.coco.json").write_text(
            json.dumps({"images": images, "annotations": anns, "categories": categories}))
    _, _, yaml_path = convert_coco_dataset(root / "coco", root / "yolo")
    return yaml_path


class FormTests(unittest.TestCase):
    fields = [Field("n", "Epochs", "int", 5, minimum=1), Field("c", "Confidence", "float", 0.25, minimum=0, maximum=1),
              Field("m", "Model", "choice", 0, ("a", "b")), Field("p", "Preview", "check", True),
              Field("f", "File", "file", ""), Field("x", "A note", "note")]

    def test_parses_values(self) -> None:
        values = parse_fields(self.fields, {"n": "12", "c": "0,5", "m": 1, "p": False, "f": "/tmp/a.pt"})
        self.assertEqual(values, {"n": 12, "c": 0.5, "m": 1, "p": False, "f": Path("/tmp/a.pt")})
        self.assertIsNone(parse_fields(self.fields, {"f": "  "})["f"])

    def test_rejects_bad_numbers(self) -> None:
        for raw in ({"n": "1.5"}, {"n": "0"}, {"c": "2"}, {"c": "abc"}, {"m": 5}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_fields(self.fields, raw)


class ModelHelperTests(unittest.TestCase):
    def test_to_rgb_handles_every_mode(self) -> None:
        for image in (Image.new("L", (4, 4), 9), Image.new("RGBA", (4, 4), (255, 0, 0, 0)),
                      Image.new("I;16", (4, 4), 3000), Image.new("F", (4, 4), 0.5), Image.new("P", (4, 4))):
            with self.subTest(mode=image.mode):
                rgb = ym.to_rgb(image)
                self.assertEqual(rgb.mode, "RGB")
                self.assertEqual(rgb.size, (4, 4))
        # a transparent pixel becomes white, not black
        self.assertEqual(ym.to_rgb(Image.new("RGBA", (1, 1), (255, 0, 0, 0))).getpixel((0, 0)), (255, 255, 255))
        original = Image.new("RGB", (2, 2))
        self.assertIsNot(ym.to_rgb(original), original)

    def test_installed_names(self) -> None:
        self.assertEqual(ym.installed_weights_for("yolo11n.pt").name, "blood_cell_yolo11n.pt")
        self.assertEqual(ym.installed_weights_for("yolo11n.yaml").name, "blood_cell_yolo11n_scratch.pt")

    def test_train_settings_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.yaml"
            data.write_text("nc: 1\n")
            ym.TrainSettings(data).validate()
            for bad in ({"epochs": 0}, {"imgsz": 100}, {"batch": 0}, {"start_from": "x.onnx"}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    ym.TrainSettings(data, **bad).validate()
        with self.assertRaises(FileNotFoundError):
            ym.TrainSettings(Path("/nonexistent/data.yaml")).validate()

    def test_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text("epoch,  train/box_loss,metrics/mAP50(B)\n1,1.5,0.2\n2,1.2,0.4\n")
            history = ym.read_results_csv(path)
            self.assertEqual(history["epoch"], [1.0, 2.0])
            self.assertEqual(history["train/box_loss"], [1.5, 1.2])
            self.assertEqual(ym.read_results_csv(Path(tmp) / "missing.csv"), {})

    def test_predict_with_our_nms(self) -> None:
        model = FakeYolo({0: "person"}, [[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], [0, 0, 0], [0.9, 0.8, 0.7])
        image = Image.new("RGB", (64, 64))
        plain = ym.predict(model, image, iou=0.5, device="cpu")
        self.assertEqual(len(plain.detections), 3)  # the fake does no NMS
        self.assertEqual(model.calls[-1]["iou"], 0.5)
        own = ym.predict(model, image, iou=0.5, device="cpu", own_nms=True)
        self.assertEqual(model.calls[-1]["iou"], 0.99)  # library NMS practically off
        self.assertEqual(len(own.raw), 3)
        self.assertEqual(len(own.detections), 2)
        with self.assertRaises(ValueError):
            ym.predict(model, image, conf=2)

    def test_nms_free_head(self) -> None:
        model = FakeYolo({0: "person"}, [[0, 0, 10, 10]], [0], [0.9])
        image = Image.new("RGB", (32, 32))
        with self.assertRaisesRegex(ValueError, "YOLO26"):
            ym.predict(model, image, device="cpu", nms_free=True)

        class Head:
            one2one_cv2 = object()

        class Inner:
            model = [Head()]

        model.model = Inner()
        model.predictor = "built for NMS"
        prediction = ym.predict(model, image, device="cpu", nms_free=True)
        self.assertIs(model.calls[-1]["nms"], False)
        self.assertIsNone(model.predictor)  # rebuilt, so Ultralytics picks the one-to-one head
        self.assertIn("NMS-free", prediction.nms)
        self.assertIsNone(prediction.raw)


class DetectToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.new("L", (120, 80), 128)  # also checks grey images work
        self.model = FakeYolo({0: "person", 5: "bus"}, [[10, 10, 60, 70], [12, 10, 62, 70], [0, 0, 120, 40]],
                              [0, 0, 5], [0.9, 0.6, 0.8])

    def test_detect_draws_and_reports(self) -> None:
        result = yt.detect(self.document, yt.DetectSettings("yolo11n.pt"), self.model)
        self.assertTrue(result.preview_only)
        self.assertEqual(result.image.size, (120, 80))
        self.assertEqual(result.details["Detections"], 3)
        self.assertEqual(result.details["Per class"], "2 person, 1 bus")
        self.assertIn("person 0.90", result.details["#1"])
        self.assertIn("BGR", result.details["Colours"])

    def test_detect_with_our_nms_shows_before_and_after(self) -> None:
        result = yt.detect(self.document, yt.DetectSettings("yolo11n.pt", iou=0.5, own_nms=True), self.model)
        self.assertEqual(result.details["Candidates before NMS"], 3)
        self.assertEqual(result.details["Removed by NMS"], 1)
        self.assertTrue(result.preview_only)
        self.assertGreater(result.image.width, 120)  # side-by-side figure

    def test_tool_needs_ultralytics(self) -> None:
        with patch.object(ym, "ultralytics_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "requirements-yolo"):
                yt.YoloDetectTool().run(None, self.document)

    def test_tools_are_registered(self) -> None:
        ids = [tool.tool_id for tool in build_tool_registry().all()]
        for tool_id in ("yolo_detect", "yolo_dataset", "yolo_train", "yolo_blood_cells", "yolo_evaluate"):
            self.assertIn(tool_id, ids)
        categories = dict(build_tool_registry().categories())
        self.assertEqual(len(categories[yt.CATEGORY]), 5)


class BloodCellToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.yaml_path = make_dataset(self.root)
        self.test_image = self.root / "yolo" / "test" / "images" / "test0.jpg"
        # one correct RBC, one false WBC; the real WBC is missed
        self.model = FakeYolo(CELLS, [[0, 0, 20, 20], [0, 40, 10, 50]], [1, 2], [0.9, 0.5])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_ground_truth_vs_prediction(self) -> None:
        image = Image.open(self.test_image).convert("RGB")
        result = yt.detect_cells(image, self.test_image, yt.CellSettings(Path("m.pt")), self.model)
        self.assertEqual(result.details["Matches at IoU ≥ 0.50"], "1 TP, 1 FP, 1 missed")
        self.assertEqual(result.details["Count WBC"], "predicted 1, real 1")
        self.assertEqual(result.details["Split"], "test (unseen)")
        self.assertTrue(result.preview_only)
        self.assertIn("1 correct", result.message)

    def test_without_ground_truth_only_counts(self) -> None:
        result = yt.detect_cells(Image.new("RGB", (64, 64)), None, yt.CellSettings(Path("m.pt")), self.model)
        self.assertEqual(result.details["Count RBC"], 1)
        self.assertIn("no label file", result.details["Ground truth"])

    def test_split_of_label(self) -> None:
        self.assertEqual(yt.split_of(Path("/d/train/labels/x.txt")), "train")
        self.assertIsNone(yt.split_of(Path("/d/x.txt")))
        self.assertIsNone(yt.split_of(None))

    def test_explore_dataset(self) -> None:
        result = yt.explore_dataset(self.yaml_path)
        self.assertEqual(result.details["train: images"], 2)
        self.assertEqual(result.details["test: per class"], "PC 0, RBC 2, WBC 2")
        self.assertEqual(result.details["Imbalance"], "PC 0.0%, RBC 50.0%, WBC 50.0%")
        self.assertIsNotNone(result.image)

    def test_evaluation_with_our_metrics(self) -> None:
        perfect = FakeYolo(CELLS, [[0, 0, 20, 20], [30, 30, 60, 60]], [1, 2], [0.9, 0.8])
        evaluation = evaluate_split(perfect, self.yaml_path, "test", device="cpu")
        self.assertEqual(evaluation.images, 2)
        self.assertAlmostEqual(evaluation.result.map, 1.0)
        result = yt.evaluation_result(evaluation, {"map50": 0.99, "map": 0.9}, Path("m.pt"))
        self.assertEqual(result.details["mAP@[.5:.95]"], "1.000")
        self.assertEqual(result.details["RBC"], "AP50 1.000, AP75 1.000, AP50-95 1.000 (2 objects)")
        self.assertIn("Ultralytics model.val mAP@0.5", result.details)

    def test_evaluation_rejects_a_model_of_other_classes(self) -> None:
        coco_model = FakeYolo({0: "person", 1: "bicycle"}, [], [], [])
        with self.assertRaisesRegex(ValueError, "trained on this dataset"):
            evaluate_split(coco_model, self.yaml_path, "test", device="cpu")

    def test_training_result(self) -> None:
        history = {"epoch": [1, 2], "train/box_loss": [1.2, 1.0], "val/box_loss": [1.3, 1.1],
                   "metrics/mAP50(B)": [0.5, 0.8], "metrics/mAP50-95(B)": [0.3, 0.6],
                   "metrics/precision(B)": [0.6, 0.8], "metrics/recall(B)": [0.5, 0.7]}
        outcome = ym.TrainOutcome(self.root / "run", self.root / "run/weights/best.pt", self.root / "best.pt",
                                  2, False, 90.0, history)
        settings = ym.TrainSettings(self.yaml_path, epochs=2, device="cpu")
        result = yt.training_result(outcome, settings)
        self.assertEqual(result.details["Best epoch (val mAP50-95)"], 2)
        self.assertEqual(result.details["Validation mAP@0.5"], "0.800")
        self.assertEqual(result.details["Time"], "1.5 min")
        self.assertIsNotNone(result.image)


if __name__ == "__main__":
    unittest.main()
