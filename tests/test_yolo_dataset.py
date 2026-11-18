"""Tests for the Set 9.2 COCO -> YOLO conversion and the dataset exploration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from forensics_app.tools import yolo_dataset as yd

# The Roboflow export: a parent category without boxes, then the three real classes
ROBOFLOW_CATEGORIES = [
    {"id": 0, "name": "1-2-3-SmhR", "supercategory": "none"},
    {"id": 1, "name": "PC", "supercategory": "1-2-3-SmhR"},
    {"id": 2, "name": "RBC", "supercategory": "1-2-3-SmhR"},
    {"id": 3, "name": "WBC", "supercategory": "1-2-3-SmhR"},
]


def write_split(folder: Path, images: list[tuple[str, int, int]], annotations: list[dict],
                categories: list[dict] = ROBOFLOW_CATEGORIES, create_files: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (name, width, height) in enumerate(images):
        entries.append({"id": index, "file_name": name, "width": width, "height": height})
        if create_files:
            Image.new("RGB", (width, height), (200, 150, 150)).save(folder / name)
    coco = {"images": entries, "annotations": annotations, "categories": categories}
    (folder / yd.ANNOTATIONS_FILE).write_text(json.dumps(coco), encoding="utf-8")


def ann(image_id: int, category: int, bbox: list[float]) -> dict:
    return {"id": 0, "image_id": image_id, "category_id": category, "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0}


class CocoClassesTests(unittest.TestCase):
    def test_roboflow_parent_category_is_dropped(self) -> None:
        coco = {"categories": ROBOFLOW_CATEGORIES, "annotations": [ann(0, 2, [0, 0, 1, 1])]}
        classes = yd.coco_classes(coco)
        self.assertEqual(classes.names, ["PC", "RBC", "WBC"])
        self.assertEqual(classes.coco_to_yolo, {1: 0, 2: 1, 3: 2})

    def test_plain_coco_keeps_every_category(self) -> None:
        # the dummy example of the notebook: person = 1, dog = 2
        coco = {"categories": [{"id": 2, "name": "dog"}, {"id": 1, "name": "person"}], "annotations": []}
        classes = yd.coco_classes(coco)
        self.assertEqual(classes.names, ["person", "dog"])
        self.assertEqual(classes.coco_to_yolo, {1: 0, 2: 1})

    def test_no_categories(self) -> None:
        with self.assertRaises(ValueError):
            yd.coco_classes({"categories": []})


class ConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.coco = self.root / "coco"
        self.yolo = self.root / "yolo"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_one_split_as_in_the_lab(self) -> None:
        write_split(
            self.coco / "train",
            [("a.jpg", 640, 480), ("b.jpg", 100, 100), ("gone.jpg", 10, 10)],
            [ann(0, 2, [100, 150, 200, 300]), ann(0, 3, [0, 0, 64, 48]), ann(0, 1, [5, 5, 0, 10])],
        )
        (self.coco / "train" / "gone.jpg").unlink()
        summary = yd.coco_to_yolo(self.coco / "train", self.yolo / "train")

        self.assertTrue((self.yolo / "train" / "images" / "a.jpg").is_file())
        lines = (self.yolo / "train" / "labels" / "a.txt").read_text().split()
        self.assertEqual(lines[:5], ["1", "0.312500", "0.625000", "0.312500", "0.625000"])
        self.assertEqual(lines[5:], ["2", "0.050000", "0.050000", "0.100000", "0.100000"])
        # an image without boxes still gets an (empty) label file: a background image
        self.assertEqual((self.yolo / "train" / "labels" / "b.txt").read_text(), "")
        self.assertEqual((summary.images, summary.boxes, summary.skipped_boxes, summary.missing_images), (2, 2, 1, 1))
        self.assertEqual(summary.per_class, {"PC": 0, "RBC": 1, "WBC": 1})

    def test_missing_json(self) -> None:
        (self.coco / "train").mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            yd.coco_to_yolo(self.coco / "train", self.yolo / "train")

    def test_whole_dataset_and_data_yaml(self) -> None:
        write_split(self.coco / "train", [("t.jpg", 64, 64)], [ann(0, 1, [0, 0, 8, 8]), ann(0, 2, [10, 10, 20, 20])])
        write_split(self.coco / "test", [("s.jpg", 64, 64)], [ann(0, 3, [0, 0, 32, 32])])
        names, summaries, yaml_path = yd.convert_coco_dataset(self.coco, self.yolo)

        self.assertEqual(names, ["PC", "RBC", "WBC"])
        self.assertEqual([s.split for s in summaries], ["train", "test"])
        data = yd.read_data_yaml(yaml_path)
        self.assertEqual(data["names"], ["PC", "RBC", "WBC"])
        self.assertEqual(int(data["nc"]), 3)
        self.assertEqual(yd.split_images_dir(data, yaml_path, "test"), (self.yolo / "test" / "images").resolve())
        with self.assertRaises(KeyError):
            yd.split_images_dir(data, yaml_path, "val")  # no valid split was converted

        stats = yd.dataset_stats(self.yolo / "train" / "images", 3, "train")
        self.assertEqual((stats.images, stats.boxes), (1, 2))
        np.testing.assert_array_equal(stats.per_class, [1, 1, 0])
        np.testing.assert_array_equal(stats.images_with_class, [1, 1, 0])
        np.testing.assert_allclose(stats.sizes, [[8 / 64, 8 / 64], [20 / 64, 20 / 64]])

        gt = yd.ground_truth(self.yolo / "test" / "images" / "s.jpg", 64, 64)
        np.testing.assert_allclose(gt.boxes, [[0, 0, 32, 32]])
        np.testing.assert_array_equal(gt.classes, [2])

    def test_needs_a_train_split(self) -> None:
        write_split(self.coco / "valid", [("v.jpg", 8, 8)], [])
        with self.assertRaises(FileNotFoundError):
            yd.convert_coco_dataset(self.coco, self.yolo)

    def test_relative_roboflow_yaml(self) -> None:
        (self.yolo / "valid" / "images").mkdir(parents=True)
        yaml_path = self.yolo / "data.yaml"
        yaml_path.write_text("train: ../train/images\nval: ../valid/images\nnc: 3\nnames: ['PC', 'RBC', 'WBC']\n")
        data = yd.read_data_yaml(yaml_path)
        self.assertEqual(data["names"], ["PC", "RBC", "WBC"])
        self.assertEqual(yd.split_images_dir(data, yaml_path, "valid").name, "images")


class LabelFileTests(unittest.TestCase):
    def test_label_path_follows_the_ultralytics_convention(self) -> None:
        self.assertEqual(yd.label_path_for(Path("/d/test/images/x.jpg")), Path("/d/test/labels/x.txt"))
        self.assertEqual(yd.label_path_for(Path("/d/x.png")), Path("/d/x.txt"))

    def test_read_labels_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.txt"
            path.write_text("1 0.5 0.5 0.2 0.2\n\n2 0.1 0.1 0.1 0.1\n")
            classes, boxes = yd.read_yolo_labels(path)
            np.testing.assert_array_equal(classes, [1, 2])
            self.assertEqual(boxes.shape, (2, 4))
            path.write_text("1 0.5 0.5\n")
            with self.assertRaises(ValueError):
                yd.read_yolo_labels(path)
            path.write_text("")
            self.assertEqual(len(yd.read_yolo_labels(path)[0]), 0)


if __name__ == "__main__":
    unittest.main()
