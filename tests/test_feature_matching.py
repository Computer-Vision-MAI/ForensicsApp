"""Tests for Set 5.3: matching keypoints with a second image (+ RANSAC)."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import data, transform

from forensics_app.core import ImageDocument
from forensics_app.tools import FeatureMatchingTool
from forensics_app.tools.dialogs import run_with_progress
from forensics_app.tools.matching import (
    GAP,
    INLIER_COLOR,
    ImageFeatures,
    MatchResult,
    MatchSettings,
    align_to_reference,
    apply_feature_matching,
    compare_features,
    detect_features,
    full_resolution_matrix,
    match_features,
    ransac_filter,
    render_matches,
    working_image,
)
from forensics_app.tools.matching_dialog import parse_match_settings

# Same homography as the RANSAC slides: the astronaut's corners moved inwards/outwards
SLIDE_TRANSFORM = transform.ProjectiveTransform.from_estimate(
    np.array([[0, 0], [512, 0], [512, 512], [0, 512]]),
    np.array([[50, 50], [500, -50], [500, 500], [-50, 500]]),
)

# Small and fast settings so the whole suite stays quick
FAST_ORB = MatchSettings(method="orb", n_keypoints=300, max_side=256)


def astronaut() -> Image.Image:
    return Image.fromarray(data.astronaut())


def warped_astronaut() -> Image.Image:
    warped = transform.warp(data.astronaut(), SLIDE_TRANSFORM.inverse)
    return Image.fromarray(np.round(warped * 255).astype(np.uint8))


def noise_image(seed: int = 3, size: int = 200) -> Image.Image:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (size // 8, size // 8), dtype=np.uint8)
    return Image.fromarray(blocks).resize((size, size), Image.Resampling.NEAREST)


class MatchSettingsTests(unittest.TestCase):
    def test_defaults_are_valid_and_use_sift(self) -> None:
        settings = MatchSettings()
        self.assertEqual(settings.method, "sift")
        self.assertEqual(settings.metric, "euclidean")

    def test_orb_uses_hamming_distance(self) -> None:
        self.assertEqual(MatchSettings(method="orb").metric, "hamming")

    def test_min_samples_depends_on_model(self) -> None:
        self.assertEqual(MatchSettings(ransac_model="affine").min_samples, 3)
        self.assertEqual(MatchSettings(ransac_model="projective").min_samples, 4)

    def test_rejects_invalid_values(self) -> None:
        invalid = [
            {"method": "surf"},
            {"n_keypoints": 0},
            {"n_octaves": 0},
            {"n_scales": 0},
            {"sigma": 0.5},
            {"max_side": 10},
            {"max_ratio": 0.0},
            {"max_ratio": 1.5},
            {"ransac_model": "similarity"},
            {"residual_threshold": 0.0},
            {"max_trials": 0},
        ]
        for kwargs in invalid:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MatchSettings(**kwargs)

    def test_full_size_working_size_is_allowed(self) -> None:
        self.assertEqual(MatchSettings(max_side=0).max_side, 0)

    def test_describe_lists_only_relevant_parameters(self) -> None:
        orb = MatchSettings(method="orb", use_ransac=False).describe()
        self.assertIn("n_keypoints", orb)
        self.assertNotIn("n_octaves", orb)
        self.assertEqual(orb["RANSAC"], "off")
        sift = MatchSettings().describe()
        self.assertIn("n_octaves", sift)
        self.assertEqual(sift["RANSAC model"], "affine")


class ParseMatchSettingsTests(unittest.TestCase):
    def test_parses_text_fields(self) -> None:
        settings = parse_match_settings(
            {
                "method": "ORB",
                "n_keypoints": " 800 ",
                "max_ratio": "0.75",
                "residual_threshold": "3",
                "cross_check": False,
                "use_ransac": True,
                "ransac_model": "projective",
            }
        )
        self.assertEqual(settings.method, "orb")
        self.assertEqual(settings.n_keypoints, 800)
        self.assertAlmostEqual(settings.max_ratio, 0.75)
        self.assertEqual(settings.residual_threshold, 3.0)
        self.assertFalse(settings.cross_check)
        self.assertEqual(settings.ransac_model, "projective")

    def test_missing_fields_keep_defaults(self) -> None:
        self.assertEqual(parse_match_settings({}), MatchSettings())

    def test_bad_number_gives_readable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Number of keypoints must be a whole number"):
            parse_match_settings({"n_keypoints": "many"})
        with self.assertRaisesRegex(ValueError, "Max ratio must be a number"):
            parse_match_settings({"max_ratio": "0,8x"})

    def test_out_of_range_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_match_settings({"max_ratio": "2"})


class WorkingImageTests(unittest.TestCase):
    def test_large_image_is_reduced_keeping_aspect(self) -> None:
        work, scale = working_image(Image.new("RGB", (1000, 500)), 200)
        self.assertEqual(work.size, (200, 100))
        self.assertEqual(work.mode, "L")
        self.assertAlmostEqual(scale, 0.2)

    def test_small_image_and_full_size_are_untouched(self) -> None:
        self.assertEqual(working_image(Image.new("RGB", (100, 50)), 200)[1], 1.0)
        work, scale = working_image(Image.new("RGB", (1000, 500)), 0)
        self.assertEqual((work.size, scale), ((1000, 500), 1.0))

    def test_accepts_every_common_mode(self) -> None:
        base = astronaut().resize((64, 64))
        for mode in ("RGB", "RGBA", "L", "LA", "P", "1", "CMYK", "I", "F", "I;16"):
            with self.subTest(mode=mode):
                image = base.convert("L").convert(mode) if mode in ("I", "F", "I;16") else base.convert(mode)
                work, _ = working_image(image, 640)
                self.assertEqual((work.mode, work.size), ("L", (64, 64)))

    def test_empty_image_raises(self) -> None:
        with self.assertRaises(ValueError):
            working_image(Image.new("L", (0, 10)), 640)


class DetectAndMatchTests(unittest.TestCase):
    def test_orb_and_sift_find_keypoints(self) -> None:
        for settings in (FAST_ORB, MatchSettings(max_side=256)):
            with self.subTest(method=settings.method):
                features = detect_features(astronaut(), settings)
                self.assertGreater(len(features), 50)
                self.assertEqual(len(features.descriptors), len(features))
                self.assertAlmostEqual(features.scale, 0.5)
                self.assertEqual(features.image.size, (256, 256))

    def test_uniform_image_has_no_keypoints_and_no_matches(self) -> None:
        empty = detect_features(Image.new("L", (100, 100), 128), FAST_ORB)
        self.assertEqual(len(empty), 0)
        self.assertIsNone(empty.descriptors)
        full = detect_features(astronaut(), FAST_ORB)
        self.assertEqual(match_features(empty, full, FAST_ORB).shape, (0, 2))
        result = compare_features(empty, full, FAST_ORB)
        self.assertEqual((result.n_matches, result.n_inliers, result.score), (0, 0, 0))
        self.assertIsNone(result.model)

    def test_identical_images_match_at_the_same_place(self) -> None:
        features = detect_features(astronaut(), FAST_ORB)
        matches = match_features(features, features, FAST_ORB)
        self.assertGreater(len(matches), 100)
        np.testing.assert_array_equal(matches[:, 0], matches[:, 1])

    def test_ratio_test_and_cross_check_reduce_matches(self) -> None:
        f1 = detect_features(astronaut(), FAST_ORB)
        f2 = detect_features(warped_astronaut(), FAST_ORB)
        loose = MatchSettings(method="orb", n_keypoints=300, max_side=256, cross_check=False, max_ratio=1.0)
        self.assertGreater(len(match_features(f1, f2, loose)), len(match_features(f1, f2, FAST_ORB)))


class RansacTests(unittest.TestCase):
    def test_recovers_the_slide_homography(self) -> None:
        settings = MatchSettings(ransac_model="projective", residual_threshold=2.0)
        f1 = detect_features(astronaut(), settings)
        f2 = detect_features(warped_astronaut(), settings)
        result = compare_features(f1, f2, settings)
        self.assertIsNotNone(result.model)
        self.assertGreater(result.n_inliers, 50)
        self.assertLess(result.n_inliers, result.n_matches + 1)
        estimated = full_resolution_matrix(result.model, f1.scale, f2.scale)
        estimated = estimated / estimated[2, 2]
        np.testing.assert_allclose(estimated[:2, :2], SLIDE_TRANSFORM.params[:2, :2], atol=0.02)
        np.testing.assert_allclose(estimated[:2, 2], SLIDE_TRANSFORM.params[:2, 2], atol=2.0)

    def test_inliers_mark_a_subset_of_matches(self) -> None:
        f1 = detect_features(astronaut(), FAST_ORB)
        f2 = detect_features(warped_astronaut(), FAST_ORB)
        result = compare_features(f1, f2, FAST_ORB)
        self.assertEqual(result.inliers.shape, (result.n_matches,))
        self.assertEqual(result.score, result.n_inliers)

    def test_without_ransac_the_score_is_the_number_of_matches(self) -> None:
        settings = MatchSettings(method="orb", n_keypoints=300, max_side=256, use_ransac=False)
        f1 = detect_features(astronaut(), settings)
        f2 = detect_features(warped_astronaut(), settings)
        result = compare_features(f1, f2, settings)
        self.assertIsNone(result.inliers)
        self.assertIsNone(result.n_inliers)
        self.assertEqual(result.score, result.n_matches)

    def test_too_few_matches_give_no_model(self) -> None:
        keypoints = np.array([[0.0, 0.0], [10.0, 0.0]])
        matches = np.array([[0, 0], [1, 1]])
        model, inliers = ransac_filter(keypoints, keypoints, matches, MatchSettings())
        self.assertIsNone(model)
        np.testing.assert_array_equal(inliers, [False, False])

    def test_random_matches_give_no_model(self) -> None:
        rng = np.random.default_rng(7)
        keypoints1 = rng.random((40, 2)) * 500
        keypoints2 = rng.random((40, 2)) * 500
        matches = np.column_stack((np.arange(40), np.arange(40)))
        settings = MatchSettings(residual_threshold=1.0)
        model, inliers = ransac_filter(keypoints1, keypoints2, matches, settings)
        self.assertIsNone(model)
        self.assertFalse(inliers.any())

    def test_ransac_is_reproducible(self) -> None:
        f1 = detect_features(astronaut(), FAST_ORB)
        f2 = detect_features(warped_astronaut(), FAST_ORB)
        first = compare_features(f1, f2, FAST_ORB)
        second = compare_features(f1, f2, FAST_ORB)
        np.testing.assert_array_equal(first.inliers, second.inliers)


class GeometryTests(unittest.TestCase):
    def test_full_resolution_matrix_undoes_both_scales(self) -> None:
        model = transform.AffineTransform(translation=(10, 20))
        matrix = full_resolution_matrix(model, scale1=0.5, scale2=0.25)
        # original point (100, 100) -> working (50, 50) -> (60, 70) -> original (240, 280)
        mapped = matrix @ np.array([100.0, 100.0, 1.0])
        np.testing.assert_allclose(mapped[:2] / mapped[2], [240.0, 280.0])

    def test_align_to_reference_undoes_the_warp(self) -> None:
        reference = astronaut()
        aligned = align_to_reference(reference, warped_astronaut(), SLIDE_TRANSFORM.params)
        self.assertEqual((aligned.size, aligned.mode), (reference.size, "RGB"))
        diff = np.abs(np.asarray(aligned, dtype=float) - np.asarray(reference, dtype=float))
        # The centre is fully covered by the warped image, so it must be almost identical
        self.assertLess(diff[150:350, 150:350].mean(), 12.0)

    def test_align_keeps_grayscale_images_gray(self) -> None:
        gray = astronaut().convert("L")
        self.assertEqual(align_to_reference(gray, gray, np.eye(3)).mode, "L")


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.f1 = detect_features(astronaut(), FAST_ORB)
        self.f2 = detect_features(warped_astronaut().resize((256, 128)), FAST_ORB)

    def test_images_are_placed_side_by_side_at_the_same_height(self) -> None:
        result = compare_features(self.f1, self.f2, FAST_ORB)
        canvas = render_matches(self.f1, self.f2, result)
        self.assertEqual(canvas.mode, "RGB")
        # 256x256 next to 256x128 scaled to 512x256
        self.assertEqual(canvas.size, (256 + GAP + 512, 256))

    def test_title_adds_a_banner(self) -> None:
        result = compare_features(self.f1, self.f2, FAST_ORB)
        plain = render_matches(self.f1, self.f2, result)
        titled = render_matches(self.f1, self.f2, result, title="Reference <---> Best")
        self.assertEqual(titled.width, plain.width)
        self.assertGreater(titled.height, plain.height)

    def test_inliers_are_drawn_in_green(self) -> None:
        f2 = detect_features(warped_astronaut(), FAST_ORB)
        result = compare_features(self.f1, f2, FAST_ORB)
        self.assertGreater(result.n_inliers, 0)
        pixels = np.asarray(render_matches(self.f1, f2, result, show_keypoints=False))
        self.assertTrue(np.all(pixels == INLIER_COLOR, axis=-1).any())

    def test_no_keypoints_still_renders(self) -> None:
        empty = ImageFeatures(Image.new("L", (50, 40)), np.empty((0, 2)), None, 1.0)
        result = MatchResult(np.empty((0, 2), dtype=int), None, None)
        self.assertEqual(render_matches(empty, empty, result).size, (50 + GAP + 50, 40))


class ApplyFeatureMatchingTests(unittest.TestCase):
    def test_matches_output(self) -> None:
        image, result, f1, f2 = apply_feature_matching(astronaut(), warped_astronaut(), FAST_ORB)
        self.assertEqual(image.width, f1.image.width + GAP + f2.image.width)
        self.assertGreater(result.n_matches, 0)

    def test_aligned_output_has_the_reference_size(self) -> None:
        reference = astronaut().resize((400, 300))
        image, *_ = apply_feature_matching(reference, warped_astronaut(), FAST_ORB, output="aligned")
        self.assertEqual(image.size, (400, 300))

    def test_transforming_the_first_image_recreates_the_second(self) -> None:
        # The second image is the first deformed by a known matrix X (the slides' one).
        # RANSAC estimates X from the matches, so applying it to the first gives the second.
        settings = MatchSettings(ransac_model="projective", residual_threshold=2.0)
        second = warped_astronaut()
        image, result, *_ = apply_feature_matching(astronaut(), second, settings, output="transformed")
        self.assertEqual(image.size, second.size)
        diff = np.abs(np.asarray(image, dtype=float) - np.asarray(second, dtype=float))
        self.assertLess(diff[150:350, 150:350].mean(), 12.0)

    def test_transformed_output_takes_the_second_image_size(self) -> None:
        second = warped_astronaut().resize((300, 200))
        image, *_ = apply_feature_matching(astronaut(), second, FAST_ORB, output="transformed")
        self.assertEqual(image.size, (300, 200))

    def test_both_directions_are_inverse(self) -> None:
        settings = MatchSettings(ransac_model="projective", residual_threshold=2.0)
        first = astronaut()
        transformed, *_ = apply_feature_matching(first, warped_astronaut(), settings, output="transformed")
        back, *_ = apply_feature_matching(first, transformed, settings, output="aligned")
        diff = np.abs(np.asarray(back, dtype=float) - np.asarray(first, dtype=float))
        self.assertLess(diff[150:350, 150:350].mean(), 15.0)

    def test_warp_outputs_need_ransac(self) -> None:
        settings = MatchSettings(method="orb", use_ransac=False)
        for output in ("transformed", "aligned"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "needs RANSAC"):
                apply_feature_matching(astronaut(), astronaut(), settings, output=output)

    def test_aligned_output_needs_ransac(self) -> None:
        settings = MatchSettings(method="orb", use_ransac=False)
        with self.assertRaisesRegex(ValueError, "needs RANSAC"):
            apply_feature_matching(astronaut(), astronaut(), settings, output="aligned")

    def test_aligned_output_fails_clearly_without_a_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "could not find a reliable transform"):
            apply_feature_matching(astronaut(), Image.new("L", (100, 100)), FAST_ORB, output="aligned")

    def test_unknown_output_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_feature_matching(astronaut(), astronaut(), FAST_ORB, output="gif")


class RunWithProgressTests(unittest.TestCase):
    def test_runs_inline_without_parent(self) -> None:
        calls = []
        value = run_with_progress(None, "t", lambda report, cancelled: calls.append(cancelled()) or 42)
        self.assertEqual((value, calls), (42, [False]))

    def test_errors_propagate(self) -> None:
        def task(report, cancelled):
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            run_with_progress(None, "t", task)


class FeatureMatchingToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = astronaut()
        self.tool = FeatureMatchingTool()

    @patch("forensics_app.tools.feature_matching.load_image_file", return_value=None)
    def test_cancelling_the_file_dialog_returns_none(self, _load) -> None:
        self.assertIsNone(self.tool.run(None, self.document))

    @patch("forensics_app.tools.feature_matching.ask_match_settings", return_value=None)
    @patch("forensics_app.tools.feature_matching.load_image_file")
    def test_cancelling_the_settings_returns_none(self, load, _ask) -> None:
        load.return_value = (Path("second.png"), warped_astronaut())
        self.assertIsNone(self.tool.run(None, self.document))

    @patch("forensics_app.tools.feature_matching.ask_match_settings")
    @patch("forensics_app.tools.feature_matching.load_image_file")
    def test_matches_result(self, load, ask) -> None:
        load.return_value = (Path("second.png"), warped_astronaut())
        ask.return_value = (FAST_ORB, {"output": "matches"})
        result = self.tool.run(None, self.document)
        self.assertIsNotNone(result.image)
        self.assertFalse(result.preview_only)
        self.assertEqual(result.details["Second image"], "second.png")
        self.assertIn("RANSAC inliers", result.details)
        self.assertIn("Transform (current → second)", result.details)
        self.assertIn("second.png", result.message)
        # The tool never touches the document itself
        self.assertEqual(self.document.current.size, (512, 512))

    @patch("forensics_app.tools.feature_matching.ask_match_settings")
    @patch("forensics_app.tools.feature_matching.load_image_file")
    def test_transformed_result(self, load, ask) -> None:
        second = warped_astronaut().resize((300, 200))
        load.return_value = (Path("second.png"), second)
        ask.return_value = (FAST_ORB, {"output": "transformed"})
        result = self.tool.run(None, self.document)
        self.assertEqual(result.image.size, (300, 200))
        self.assertTrue(result.message.startswith("Transformed the current image into second.png"))

    @patch("forensics_app.tools.feature_matching.ask_match_settings")
    @patch("forensics_app.tools.feature_matching.load_image_file")
    def test_aligned_result(self, load, ask) -> None:
        load.return_value = (Path("second.png"), warped_astronaut())
        ask.return_value = (FAST_ORB, {"output": "aligned"})
        result = self.tool.run(None, self.document)
        self.assertEqual(result.image.size, (512, 512))
        self.assertTrue(result.message.startswith("Transformed second.png back"))


if __name__ == "__main__":
    unittest.main()
