"""Unit tests for Haar-like feature extraction and AdaBoost classification."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.dialogs import TaskCancelled
from forensics_app.tools.haar_adaboost import (
    CANONICAL_PATCH_SIZE,
    CASCADE_FORMAT_VERSION,
    CascadeStage,
    FeatureType,
    HaarLikeFeature,
    ViolaJonesCascade,
    activated_classifiers,
    compute_integral_image,
    draw_haar_overlay,
    ensemble_vote,
    learn_adaboost,
    load_cascade_model,
    load_haar_model,
    patch_to_normalized_integral,
    save_haar_model,
    sum_region,
    train_viola_jones_cascade,
)


def _write_patches(folder: Path, bright_center: bool, count: int, seed: int) -> None:
    """Write ``count`` 19x19 grayscale patches: faces get a bright centre, non-faces noise."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    for i in range(count):
        arr = rng.integers(0, 60, size=(19, 19)).astype(np.uint8)
        if bright_center:
            arr[5:14, 5:14] = 220
        Image.fromarray(arr).save(folder / f"{i:03d}.png")


class TestHaarAdaBoost(unittest.TestCase):
    """Test suite for Haar feature calculations, AdaBoost training, and overlays."""

    def test_compute_integral_image(self) -> None:
        """Verify that the integral image correctly computes 2D cumulative sums."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        ii = compute_integral_image(arr)
        # Integral image has shape (H+1, W+1)
        self.assertEqual(ii.shape, (3, 4))
        # First row and column must be zeros
        self.assertTrue(np.all(ii[0, :] == 0))
        self.assertTrue(np.all(ii[:, 0] == 0))
        # Bottom-right corner is total sum of all elements
        self.assertEqual(ii[2, 3], np.sum(arr))

    def test_sum_region(self) -> None:
        """Verify constant-time box sum calculation using corner values."""
        arr = np.ones((10, 10), dtype=np.float64)
        ii = compute_integral_image(arr)
        # Sum of a 4x5 sub-rectangle in an all-ones image should be exactly 20
        total = sum_region(ii, (2, 3), (6, 8))
        self.assertEqual(total, 20.0)

    def test_haar_feature_types(self) -> None:
        """Verify response scoring for different Haar feature geometries."""
        # Create an artificial test image with a bright top half and dark bottom half
        img = np.zeros((20, 20), dtype=np.float64)
        img[:10, :] = 1.0  # Top half bright
        ii = compute_integral_image(img)

        # TWO_HORIZONTAL: top half (+) minus bottom half (-)
        feat_h = HaarLikeFeature(
            feature_type=FeatureType.TWO_HORIZONTAL,
            position=(0, 0),
            width=20,
            height=20,
            threshold=0.0,
            polarity=1,
            weight=1.5,
        )
        score_h = feat_h.get_score(ii)
        # Top half sum is 200, bottom half is 0 -> score should be +200
        self.assertGreater(score_h, 0.0)

        # TWO_VERTICAL: left half (+) minus right half (-)
        feat_v = HaarLikeFeature(
            feature_type=FeatureType.TWO_VERTICAL,
            position=(0, 0),
            width=20,
            height=20,
            threshold=0.0,
            polarity=1,
        )
        score_v = feat_v.get_score(ii)
        # Image is symmetric horizontally, so left and right halves are equal -> score = 0
        self.assertAlmostEqual(score_v, 0.0)

    def test_haar_feature_voting(self) -> None:
        """Verify that weak classifier votes reflect threshold and polarity."""
        img = np.ones((20, 20), dtype=np.float64)
        ii = compute_integral_image(img)
        feat = HaarLikeFeature(
            feature_type=FeatureType.TWO_VERTICAL,
            position=(0, 0),
            width=10,
            height=10,
            threshold=10.0,
            polarity=1,
            weight=2.0,
        )
        # get_score is 0, which is < 10 -> votes positively (+weight)
        self.assertEqual(feat.get_vote(ii), 2.0)

    def test_learn_and_ensemble_vote(self) -> None:
        """Verify training AdaBoost classifiers on synthetic positive/negative samples."""
        # 5 positive images (bright center) and 5 negative images (uniform black)
        pos_samples = []
        neg_samples = []
        for _ in range(5):
            pos_img = np.zeros((25, 25), dtype=np.float64)
            pos_img[8:17, 8:17] = 1.0
            pos_samples.append(compute_integral_image(pos_img))

            neg_img = np.zeros((25, 25), dtype=np.float64)
            neg_samples.append(compute_integral_image(neg_img))

        clfs = learn_adaboost(pos_samples, neg_samples, num_classifiers=3, img_height=25, img_width=25)
        self.assertGreaterEqual(len(clfs), 1)

        # Ensemble vote on a positive sample
        is_face, score, _ = ensemble_vote(pos_samples[0], clfs)
        self.assertTrue(is_face)
        self.assertGreater(score, 0.0)

    def test_save_and_load_model(self) -> None:
        """Verify JSON round-trip serialization of trained Haar features."""
        feat = HaarLikeFeature(
            feature_type=FeatureType.FOUR,
            position=(2, 4),
            width=8,
            height=10,
            threshold=1.23,
            polarity=-1,
            weight=0.75,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir) / "haar_model.json"
            save_haar_model([feat], test_path)
            loaded = load_haar_model(test_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].type, FeatureType.FOUR)
            self.assertEqual(loaded[0].top_left, (2, 4))
            self.assertEqual(loaded[0].width, 8)
            self.assertEqual(loaded[0].height, 10)
            self.assertAlmostEqual(loaded[0].threshold, 1.23)
            self.assertEqual(loaded[0].polarity, -1)
            self.assertAlmostEqual(loaded[0].weight, 0.75)

    def test_viola_jones_cascade(self) -> None:
        """Verify multi-stage ViolaJonesCascade evaluation and prediction."""
        from forensics_app.tools.haar_adaboost import (
            CascadeStage,
            ViolaJonesCascade,
            load_cascade_model,
            save_cascade_model,
        )

        feat1 = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 6, 6, threshold=0.0, weight=1.0)
        feat2 = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 6, 6, threshold=0.0, weight=1.0)
        stage1 = CascadeStage(classifiers=[feat1], threshold=0.0)
        stage2 = CascadeStage(classifiers=[feat2], threshold=0.0)
        cascade = ViolaJonesCascade(stages=[stage1, stage2])

        dummy_ii = np.ones((20, 20), dtype=np.float64)
        is_face, passed, scores = cascade.predict(dummy_ii)
        self.assertIsInstance(is_face, bool)
        self.assertIsInstance(passed, int)
        self.assertEqual(len(scores), passed + 1 if not is_face else 2)

        # Test save & load cascade
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_cascade_path = Path(tmp_dir) / "cascade.json"
            save_cascade_model(cascade, tmp_cascade_path)
            loaded = load_cascade_model(tmp_cascade_path)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.stages), 2)

    def test_cascade_accepts_window_passing_every_stage(self) -> None:
        """A window that clears every stage threshold is a face with all stages passed."""
        feat = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 6, 6, threshold=0.0, weight=1.0)
        cascade = ViolaJonesCascade(
            stages=[CascadeStage([feat], threshold=-2.0), CascadeStage([feat], threshold=-2.0)]
        )
        is_face, passed, scores = cascade.predict(np.ones((20, 20), dtype=np.float64))
        self.assertTrue(is_face)
        self.assertEqual(passed, 2)
        self.assertEqual(len(scores), 2)

    def test_learn_adaboost_does_not_modify_candidates(self) -> None:
        """Selected classifiers are weighted copies; the candidate list keeps weight 1."""
        pos = [compute_integral_image(np.pad(np.ones((9, 9)), 8)) for _ in range(4)]
        neg = [compute_integral_image(np.zeros((25, 25))) for _ in range(4)]
        candidates = [
            HaarLikeFeature(FeatureType.TWO_VERTICAL, (8, 4), 9, 10),
            HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (4, 8), 10, 9),
        ]
        clfs = learn_adaboost(pos, neg, num_classifiers=2, features=candidates, img_height=25, img_width=25)
        self.assertTrue(all(c.weight == 1.0 for c in candidates))
        self.assertTrue(all(c not in candidates for c in clfs))

    def test_train_cascade_reports_progress_and_can_be_cancelled(self) -> None:
        """Cascade training on synthetic patches reports progress and honours cancel."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pos_dir = Path(tmp_dir) / "faces"
            neg_dir = Path(tmp_dir) / "non_faces"
            _write_patches(pos_dir, bright_center=True, count=20, seed=1)
            _write_patches(neg_dir, bright_center=False, count=30, seed=2)

            calls: list[tuple[int, int, str]] = []
            cascade = train_viola_jones_cascade(
                stage_configs=[1, 2],
                max_pos=20,
                max_neg=30,
                train_pos_dir=pos_dir,
                train_neg_dir=neg_dir,
                progress=lambda done, total, label: calls.append((done, total, label)),
            )
            self.assertEqual(len(cascade.stages), 2)
            self.assertEqual([c[0] for c in calls], [0, 1, 2])
            self.assertTrue(all(c[1] == 3 for c in calls))
            # Stumps carry learned thresholds, not the 0.0 placeholder
            self.assertTrue(any(c.threshold != 0.0 for c in cascade.get_all_classifiers()))
            # Training faces pass the cascade
            face_ii = patch_to_normalized_integral(Image.open(next(pos_dir.glob("*.png"))))
            self.assertTrue(cascade.predict(face_ii)[0])

            with self.assertRaises(TaskCancelled):
                train_viola_jones_cascade(
                    stage_configs=[1],
                    train_pos_dir=pos_dir,
                    train_neg_dir=neg_dir,
                    should_cancel=lambda: True,
                )

    def test_polarity_inverts_the_stump(self) -> None:
        """Polarity +1 votes face below the threshold, polarity -1 above it."""
        img = np.zeros((20, 20), dtype=np.float64)
        img[:, :10] = 1.0  # bright 20x10 left half -> TWO_VERTICAL score = +200
        ii = compute_integral_image(img)
        for threshold, polarity, expected in [(250.0, 1, 1), (150.0, 1, -1), (250.0, -1, -1), (150.0, -1, 1)]:
            with self.subTest(threshold=threshold, polarity=polarity):
                feat = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 20, 20, threshold, polarity)
                self.assertEqual(feat.get_score(ii), 200.0)
                self.assertEqual(feat.predict(ii), expected)

    def test_get_score_matches_explicit_lobe_sums(self) -> None:
        """Scores from ``lobes()`` equal the hand-written rectangle formulas."""
        ii = compute_integral_image(np.random.default_rng(5).random((19, 19)))
        tl, w, h = (2, 3), 12, 12
        br = (tl[0] + w, tl[1] + h)
        sr = lambda a, b: sum_region(ii, a, b)  # noqa: E731 - short alias for readability
        expected = {
            FeatureType.TWO_VERTICAL: sr(tl, (br[0], tl[1] + h // 2)) - sr((tl[0], tl[1] + h // 2), br),
            FeatureType.TWO_HORIZONTAL: sr(tl, (tl[0] + w // 2, br[1])) - sr((tl[0] + w // 2, tl[1]), br),
            FeatureType.THREE_HORIZONTAL: sr(tl, (tl[0] + w // 3, br[1]))
            - sr((tl[0] + w // 3, tl[1]), (tl[0] + 2 * w // 3, br[1]))
            + sr((tl[0] + 2 * w // 3, tl[1]), br),
            FeatureType.THREE_VERTICAL: sr(tl, (br[0], tl[1] + h // 3))
            - sr((tl[0], tl[1] + h // 3), (br[0], tl[1] + 2 * h // 3))
            + sr((tl[0], tl[1] + 2 * h // 3), br),
            FeatureType.FOUR: sr(tl, (tl[0] + w // 2, tl[1] + h // 2))
            - sr((tl[0] + w // 2, tl[1]), (br[0], tl[1] + h // 2))
            - sr((tl[0], tl[1] + h // 2), (tl[0] + w // 2, br[1]))
            + sr((tl[0] + w // 2, tl[1] + h // 2), br),
        }
        for ftype, value in expected.items():
            with self.subTest(feature=ftype):
                self.assertEqual(HaarLikeFeature(ftype, tl, w, h).get_score(ii), value)

    def test_adaboost_learns_threshold_and_polarity(self) -> None:
        """A single stump separates faces with *high* scores, which needs polarity -1."""
        pos, neg = [], []
        for level in (0.8, 0.9, 1.0):
            img = np.zeros((19, 19)); img[:, :9] = level  # bright left: high score
            pos.append(compute_integral_image(img))
        for level in (0.1, 0.2, 0.3):
            img = np.zeros((19, 19)); img[:, :9] = level
            neg.append(compute_integral_image(img))
        feat = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 18, 18)
        (clf,) = learn_adaboost(pos, neg, num_classifiers=1, features=[feat])
        self.assertEqual(clf.polarity, -1)
        pos_scores = [feat.get_score(ii) for ii in pos]
        neg_scores = [feat.get_score(ii) for ii in neg]
        self.assertGreater(clf.threshold, max(neg_scores))
        self.assertLess(clf.threshold, min(pos_scores))
        self.assertTrue(all(clf.predict(ii) == 1 for ii in pos))
        self.assertTrue(all(clf.predict(ii) == -1 for ii in neg))

    def test_old_format_cascade_is_not_loaded(self) -> None:
        """Cascades saved before the stump fix are reported missing so they get retrained."""
        import json

        cascade = ViolaJonesCascade([CascadeStage([HaarLikeFeature(FeatureType.FOUR, (0, 0), 6, 6)], 0.0)])
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cascade.json"
            data = cascade.to_dict()
            self.assertEqual(data["format_version"], CASCADE_FORMAT_VERSION)
            data.pop("format_version")
            path.write_text(json.dumps(data))
            self.assertIsNone(load_cascade_model(path))

    def test_overlay_colors_follow_polarity(self) -> None:
        """Lobes the stump expects darker are red, lighter ones green; polarity swaps them."""
        img = Image.new("RGB", (19, 19), color=(0, 0, 0))
        for polarity, left_red in ((1, True), (-1, False)):
            with self.subTest(polarity=polarity):
                feat = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 19, 18, polarity=polarity)
                arr = np.array(draw_haar_overlay(img, [feat], alpha=1.0, patch_size=(19, 19), draw_box=False))
                left, right = arr[9, 3], arr[9, 14]
                self.assertEqual(left[0] > left[1], left_red)  # red channel dominates when red
                self.assertEqual(right[0] > right[1], not left_red)

    def test_patch_to_normalized_integral_modes(self) -> None:
        """Every image mode gives a (20, 20) integral image normalized to [0, 1]."""
        rng = np.random.default_rng(0)
        rgb = Image.fromarray(rng.integers(0, 255, size=(30, 40, 3)).astype(np.uint8))
        images = {
            "L": rgb.convert("L"),
            "RGB": rgb,
            "RGBA": rgb.convert("RGBA"),
            "P": rgb.convert("P"),
        }
        for mode, img in images.items():
            with self.subTest(mode=mode):
                ii = patch_to_normalized_integral(img)
                self.assertEqual(ii.shape, (CANONICAL_PATCH_SIZE[0] + 1, CANONICAL_PATCH_SIZE[1] + 1))
                self.assertAlmostEqual(float(ii.max()), 1.0, places=5)
                self.assertGreaterEqual(float(ii.min()), 0.0)

    def test_patch_to_normalized_integral_all_black(self) -> None:
        """An all-black patch gives an all-zero integral image instead of NaNs."""
        ii = patch_to_normalized_integral(Image.new("L", CANONICAL_PATCH_SIZE, 0))
        self.assertTrue(np.all(ii == 0.0))

    def test_patch_to_normalized_integral_matches_previous_formula(self) -> None:
        """The helper reproduces the original inline preprocessing exactly."""
        rng = np.random.default_rng(3)
        for size in [(19, 19), (37, 50)]:
            with self.subTest(size=size):
                img = Image.fromarray(rng.integers(0, 255, size=(size[1], size[0], 3)).astype(np.uint8))
                patch = img.convert("L").resize(CANONICAL_PATCH_SIZE, Image.Resampling.BILINEAR)
                arr = np.array(patch, dtype=np.float64)
                ii = compute_integral_image(arr / (arr.max() + 1e-6))
                expected = ii / (ii.max() + 1e-6)
                np.testing.assert_array_equal(patch_to_normalized_integral(img), expected)

    def test_draw_haar_overlay(self) -> None:
        """Verify blending Haar features with alpha 0.5."""
        img = Image.new("RGB", (60, 60), color=(128, 128, 128))
        feat = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 10, 10)
        overlay = draw_haar_overlay(img, [feat], alpha=0.5, patch_size=(25, 25))
        self.assertEqual(overlay.size, (60, 60))
        self.assertEqual(overlay.mode, "RGB")

    def test_draw_haar_overlay_input_modes(self) -> None:
        """L, RGBA and P inputs come back as RGB images of the same size."""
        base = Image.new("RGB", (40, 30), color=(128, 64, 32))
        feat = HaarLikeFeature(FeatureType.FOUR, (2, 2), 8, 8)
        for mode in ("L", "RGBA", "P"):
            with self.subTest(mode=mode):
                overlay = draw_haar_overlay(base.convert(mode), [feat], patch_size=(19, 19))
                self.assertEqual(overlay.mode, "RGB")
                self.assertEqual(overlay.size, (40, 30))

    def test_draw_haar_overlay_ndarray_input(self) -> None:
        """Grayscale and RGB arrays are accepted and drawn as RGB images."""
        feat = HaarLikeFeature(FeatureType.THREE_VERTICAL, (0, 0), 9, 9)
        for arr in (np.full((30, 30), 100, np.uint8), np.full((30, 30, 3), 100, np.uint8)):
            with self.subTest(ndim=arr.ndim):
                overlay = draw_haar_overlay(arr, [feat], patch_size=(19, 19))
                self.assertEqual(overlay.mode, "RGB")
                self.assertEqual(overlay.size, (30, 30))

    def test_draw_haar_overlay_target_box_clamped_to_borders(self) -> None:
        """A box sticking out of the image is clamped and leaves the size unchanged."""
        img = Image.new("RGB", (50, 50), color=(0, 0, 0))
        feat = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 19, 19)
        overlay = draw_haar_overlay(img, [feat], target_box=(-10, 35, 40, 40), draw_box=True)
        self.assertEqual(overlay.size, (50, 50))
        arr = np.array(overlay)
        self.assertTrue(np.any(arr[35:, :40] != 0))  # drawn inside the clamped box
        self.assertTrue(np.all(arr[:30, 45:] == 0))  # nothing outside it

    def test_draw_haar_overlay_deduplication(self) -> None:
        """Verify deduplication of identical Haar features across stages."""
        img = Image.new("RGB", (60, 60), color=(128, 128, 128))
        # Create 3 identical features and 1 distinct feature
        feat1 = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 10, 10)
        feat1.weight = 0.8
        feat2 = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 10, 10)
        feat2.weight = 0.5
        feat3 = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 10, 10)
        feat3.weight = 0.2
        feat4 = HaarLikeFeature(FeatureType.TWO_VERTICAL, (4, 4), 6, 8)
        feat4.weight = 0.7

        overlay = draw_haar_overlay(
            img,
            [feat1, feat2, feat3, feat4],
            alpha=0.5,
            patch_size=(25, 25),
            max_features=10,
            draw_box=True,
            box_label="Test Face",
        )
        self.assertEqual(overlay.size, (60, 60))
        # Verify that pixels changed due to feature drawing & box
        arr = np.array(overlay)
        # Bounding box border should have green color component (0, 255, 0)
        self.assertTrue(np.any(arr[:, :, 1] > 200))


    def test_activated_classifiers_keeps_only_face_votes(self) -> None:
        """Only stumps that vote +1 on the patch are returned, in their original order."""
        arr = np.zeros((19, 19), dtype=np.float64)
        arr[:, :9] = 1.0  # bright left half -> TWO_VERTICAL score > 0
        ii = compute_integral_image(arr)
        fires = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 18, 18, threshold=0.0, polarity=-1, weight=0.8)
        silent = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 18, 18, threshold=0.0, polarity=1, weight=0.9)
        also_fires = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 18, 18, threshold=1.0, polarity=1, weight=0.1)

        active = activated_classifiers(ii, [fires, silent, also_fires])

        self.assertEqual(active, [fires, also_fires])
        self.assertEqual(activated_classifiers(ii, []), [])


if __name__ == "__main__":
    unittest.main()
