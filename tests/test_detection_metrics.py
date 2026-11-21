"""Tests for the Set 9 detection maths: box formats, IoU, NMS, matching, AP and mAP."""

from __future__ import annotations

import unittest

import numpy as np

from forensics_app.tools import detection_metrics as dm
from forensics_app.tools.detection_metrics import Detections


def dets(boxes, classes, scores=None) -> Detections:
    return Detections(np.asarray(boxes, dtype=float), np.asarray(classes), None if scores is None else np.asarray(scores))


class BoxFormatTests(unittest.TestCase):
    def test_coco_xywh_to_corners(self) -> None:
        np.testing.assert_allclose(dm.xywh_to_xyxy([[100, 150, 200, 300]]), [[100, 150, 300, 450]])

    def test_centre_format_round_trip(self) -> None:
        corners = np.array([[10, 20, 50, 80], [0, 0, 4, 2]], dtype=float)
        np.testing.assert_allclose(dm.cxcywh_to_xyxy(dm.xyxy_to_cxcywh(corners)), corners)

    def test_coco_box_to_yolo_is_normalised_centre(self) -> None:
        # the dummy COCO example of the notebook: 640 x 480 image, bbox [100, 150, 200, 300]
        xc, yc, w, h = dm.coco_to_yolo_box([100, 150, 200, 300], 640, 480)
        self.assertAlmostEqual(xc, 200 / 640)
        self.assertAlmostEqual(yc, 300 / 480)
        self.assertAlmostEqual(w, 200 / 640)
        self.assertAlmostEqual(h, 300 / 480)

    def test_coco_box_is_clipped_to_the_image(self) -> None:
        xc, yc, w, h = dm.coco_to_yolo_box([-10, 600, 50, 100], 640, 640)
        self.assertAlmostEqual(w, 40 / 640)
        self.assertAlmostEqual(h, 40 / 640)
        self.assertAlmostEqual(xc, 20 / 640)
        self.assertAlmostEqual(yc, 620 / 640)

    def test_coco_box_rejects_an_empty_image(self) -> None:
        with self.assertRaises(ValueError):
            dm.coco_to_yolo_box([0, 0, 1, 1], 0, 10)

    def test_yolo_to_pixels(self) -> None:
        np.testing.assert_allclose(dm.yolo_to_xyxy([[0.5, 0.5, 0.5, 0.25]], 640, 480), [[160, 180, 480, 300]])


class IouTests(unittest.TestCase):
    def test_identical_and_disjoint(self) -> None:
        iou = dm.box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]])
        np.testing.assert_allclose(iou, [[1.0, 0.0]])

    def test_known_overlap(self) -> None:
        # two 2 x 2 squares shifted by 1: intersection 2, union 6
        self.assertAlmostEqual(dm.box_iou([[0, 0, 2, 2]], [[1, 0, 3, 2]])[0, 0], 2 / 6)

    def test_matrix_shape_and_symmetry(self) -> None:
        rng = np.random.default_rng(0)
        a = np.sort(rng.uniform(0, 100, (5, 4)).reshape(5, 2, 2), axis=1).transpose(0, 2, 1).reshape(5, 4)
        b = np.sort(rng.uniform(0, 100, (3, 4)).reshape(3, 2, 2), axis=1).transpose(0, 2, 1).reshape(3, 4)
        iou = dm.box_iou(a, b)
        self.assertEqual(iou.shape, (5, 3))
        np.testing.assert_allclose(iou, dm.box_iou(b, a).T)
        self.assertTrue(np.all((iou >= 0) & (iou <= 1)))

    def test_empty_inputs(self) -> None:
        self.assertEqual(dm.box_iou(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape, (0, 1))

    def test_degenerate_box_has_zero_iou(self) -> None:
        self.assertEqual(dm.box_iou([[5, 5, 5, 5]], [[5, 5, 5, 5]])[0, 0], 0.0)


class NmsTests(unittest.TestCase):
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.6, 0.9, 0.8])

    def test_keeps_the_best_of_overlapping_boxes(self) -> None:
        keep = dm.nms(self.boxes, self.scores, 0.5)
        np.testing.assert_array_equal(keep, [1, 2])  # best score first

    def test_threshold_above_the_overlap_keeps_both(self) -> None:
        overlap = dm.box_iou(self.boxes[:1], self.boxes[1:2])[0, 0]
        keep = dm.nms(self.boxes, self.scores, overlap + 0.01)
        self.assertEqual(sorted(keep.tolist()), [0, 1, 2])

    def test_per_class_does_not_suppress_other_classes(self) -> None:
        classes = np.array([0, 1, 0])
        self.assertEqual(len(dm.batched_nms(self.boxes, self.scores, classes, 0.5)), 3)
        self.assertEqual(len(dm.batched_nms(self.boxes, self.scores, classes, 0.5, class_agnostic=True)), 2)

    def test_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            dm.nms(self.boxes, self.scores, 1.5)
        with self.assertRaises(ValueError):
            dm.nms(self.boxes, self.scores[:2], 0.5)

    def test_empty(self) -> None:
        self.assertEqual(len(dm.nms(np.zeros((0, 4)), np.zeros(0), 0.5)), 0)
        self.assertEqual(len(dm.batched_nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 0.5)), 0)


class MatchingTests(unittest.TestCase):
    gt = dets([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 1])

    def test_duplicate_detection_is_a_false_positive(self) -> None:
        pred = dets([[0, 0, 10, 10], [0, 0, 10, 9]], [0, 0], [0.7, 0.9])
        is_tp, matched = dm.match_detections(pred, self.gt, 0.5)
        # the higher score is matched first, the other one finds the object already taken
        np.testing.assert_array_equal(is_tp, [False, True])
        np.testing.assert_array_equal(matched, [-1, 0])

    def test_wrong_class_is_not_a_match(self) -> None:
        pred = dets([[20, 0, 30, 10]], [0], [0.9])
        is_tp, _ = dm.match_detections(pred, self.gt, 0.5)
        self.assertFalse(is_tp[0])

    def test_iou_threshold(self) -> None:
        pred = dets([[0, 0, 10, 6]], [0], [0.9])  # IoU 0.6 with the first box
        self.assertTrue(dm.match_detections(pred, self.gt, 0.6)[0][0])
        self.assertFalse(dm.match_detections(pred, self.gt, 0.65)[0][0])

    def test_compare_image_counts(self) -> None:
        pred = dets([[0, 0, 10, 10], [100, 100, 110, 110]], [0, 0], [0.9, 0.8])
        match = dm.compare_image(pred, self.gt, 0.5)
        self.assertEqual((match.true_positives, match.false_positives, match.false_negatives), (1, 1, 1))
        self.assertAlmostEqual(match.precision, 0.5)
        self.assertAlmostEqual(match.recall, 0.5)
        np.testing.assert_array_equal(match.gt_found, [True, False])

    def test_predictions_need_scores(self) -> None:
        with self.assertRaises(ValueError):
            dm.match_detections(self.gt, self.gt)


class AveragePrecisionTests(unittest.TestCase):
    def test_perfect_detector(self) -> None:
        precision, recall, _ = dm.precision_recall_curve(np.array([0.9, 0.8]), np.array([True, True]), 2)
        self.assertAlmostEqual(dm.average_precision(precision, recall), 1.0)

    def test_hand_computed_example(self) -> None:
        # ranked TP, FP, TP with 2 objects: P = 1, 1/2, 2/3 at R = 1/2, 1/2, 1
        precision, recall, _ = dm.precision_recall_curve(np.array([0.9, 0.8, 0.7]), np.array([True, False, True]), 2)
        np.testing.assert_allclose(precision, [1, 0.5, 2 / 3])
        np.testing.assert_allclose(recall, [0.5, 0.5, 1])
        self.assertAlmostEqual(dm.average_precision(precision, recall), 0.5 * 1 + 0.5 * (2 / 3))

    def test_missed_objects_lower_the_ap(self) -> None:
        precision, recall, _ = dm.precision_recall_curve(np.array([0.9]), np.array([True]), 4)
        self.assertAlmostEqual(dm.average_precision(precision, recall), 0.25)

    def test_no_detections(self) -> None:
        self.assertEqual(dm.average_precision(np.zeros(0), np.zeros(0)), 0.0)


class EvaluationTests(unittest.TestCase):
    gts = [dets([[0, 0, 10, 10], [20, 20, 40, 40]], [0, 1]), dets([[5, 5, 15, 15]], [0])]

    def test_perfect_predictions_give_map_one(self) -> None:
        preds = [Detections(g.boxes, g.classes, np.full(len(g), 0.9)) for g in self.gts]
        result = dm.evaluate_detections(preds, self.gts, ["a", "b"])
        self.assertAlmostEqual(result.map, 1.0)
        self.assertAlmostEqual(result.map_at(0.5), 1.0)
        np.testing.assert_array_equal(result.num_gt, [2, 1])
        np.testing.assert_array_equal(np.diag(result.confusion)[:2], [2, 1])

    def test_ap_depends_on_the_iou_threshold(self) -> None:
        # each box covers 60% of its object (IoU 0.6): found at 0.50-0.60, missed above.
        # The course notebook reused the 0.5 matches and reported the same AP everywhere.
        preds = [Detections(np.column_stack([g.boxes[:, :3], g.boxes[:, 1] + 0.6 * (g.boxes[:, 3] - g.boxes[:, 1])]),
                            g.classes, np.full(len(g), 0.9)) for g in self.gts]
        result = dm.evaluate_detections(preds, self.gts, ["a", "b"])
        self.assertAlmostEqual(result.map_at(0.5), 1.0)
        self.assertAlmostEqual(result.map_at(0.6), 1.0)
        self.assertAlmostEqual(result.map_at(0.65), 0.0)
        self.assertAlmostEqual(result.map, 3 / 10)

    def test_class_without_ground_truth_is_left_out_of_the_mean(self) -> None:
        preds = [Detections(g.boxes, g.classes, np.full(len(g), 0.9)) for g in self.gts]
        result = dm.evaluate_detections(preds, self.gts, ["a", "b", "never"])
        self.assertTrue(np.isnan(result.ap_per_class[2]))
        self.assertAlmostEqual(result.map, 1.0)

    def test_false_positive_and_missed_in_confusion_and_fppi(self) -> None:
        preds = [dets([[100, 100, 110, 110]], [0], [0.9]), Detections.empty(with_scores=True)]
        result = dm.evaluate_detections(preds, self.gts, ["a", "b"], iou_thresholds=(0.5,))
        background = 2
        self.assertEqual(result.confusion[background, 0], 1)  # false alarm
        self.assertEqual(result.confusion[:2, background].sum(), 3)  # every object missed
        np.testing.assert_allclose(result.fppi, [0.5])  # 1 false positive over 2 images
        np.testing.assert_allclose(result.miss_rate, [1.0])
        self.assertAlmostEqual(result.map, 0.0)

    def test_confusion_matrix_ignores_unconfident_boxes(self) -> None:
        preds = [dets([[100, 100, 110, 110]], [0], [0.01]), Detections.empty(with_scores=True)]
        result = dm.evaluate_detections(preds, self.gts, ["a", "b"], iou_thresholds=(0.5,), confusion_confidence=0.25)
        self.assertEqual(result.confusion[2, 0], 0)

    def test_wrong_class_shows_off_the_diagonal(self) -> None:
        matrix = dm.confusion_matrix([dets([[0, 0, 10, 10]], [1], [0.9])], [dets([[0, 0, 10, 10]], [0])], 2)
        self.assertEqual(matrix[0, 1], 1)

    def test_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            dm.evaluate_detections([], self.gts, ["a", "b"])

    def test_detections_validation(self) -> None:
        with self.assertRaises(ValueError):
            Detections(np.zeros((2, 4)), np.zeros(1))
        with self.assertRaises(ValueError):
            Detections(np.zeros((2, 4)), np.zeros(2), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
