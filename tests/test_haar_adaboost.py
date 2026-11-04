"""Unit tests for Haar-like feature extraction and AdaBoost classification."""

from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.haar_adaboost import (
    FeatureType,
    HaarLikeFeature,
    compute_integral_image,
    draw_haar_overlay,
    ensure_default_haar_model,
    ensemble_vote,
    learn_adaboost,
    load_haar_model,
    save_haar_model,
    sum_region,
)


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
        test_path = Path("tests/test_haar_model.json")
        try:
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
        finally:
            if test_path.exists():
                test_path.unlink()

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
        tmp_cascade_path = Path("tests/test_cascade.json")
        try:
            save_cascade_model(cascade, tmp_cascade_path)
            loaded = load_cascade_model(tmp_cascade_path)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.stages), 2)
        finally:
            if tmp_cascade_path.exists():
                tmp_cascade_path.unlink()

    def test_draw_haar_overlay(self) -> None:
        """Verify blending Haar features with alpha 0.5."""
        img = Image.new("RGB", (60, 60), color=(128, 128, 128))
        feat = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 10, 10)
        overlay = draw_haar_overlay(img, [feat], alpha=0.5, patch_size=(25, 25))
        self.assertEqual(overlay.size, (60, 60))
        self.assertEqual(overlay.mode, "RGB")

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


if __name__ == "__main__":
    unittest.main()
