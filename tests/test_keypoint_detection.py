"""Tests for Set 5.2: Keypoint Detection and Descriptor Extraction (ORB and SIFT)."""

import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import KeypointDetectionTool
from forensics_app.tools.keypoint_detection import (
    apply_keypoint_detection,
    convert_keypoint_orientations,
    extract_keypoints,
    scale_keypoint_radii,
)


def make_checkerboard_image(size: int = 64, cell: int = 8) -> Image.Image:
    """Create a high-contrast checkerboard image that produces consistent keypoints."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, cell):
        for j in range(0, size, cell):
            if (i // cell + j // cell) % 2 == 0:
                arr[i : i + cell, j : j + cell] = 255
    return Image.fromarray(arr, mode="L").convert("RGB")


class KeypointDetectionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = make_checkerboard_image(64, 8)
        self.tool = KeypointDetectionTool()
        self.parent = None  # Mock parent for Tkinter

    @patch("forensics_app.tools.keypoint_detection.ask_choice")
    def test_returns_none_on_choice_cancellation(self, mock_ask_choice) -> None:
        mock_ask_choice.return_value = None
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.simpledialog.askinteger")
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=0)
    def test_returns_none_on_orb_params_cancellation(
        self, mock_ask_choice, mock_askint
    ) -> None:
        mock_askint.return_value = None
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.simpledialog.askinteger")
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=1)
    def test_returns_none_on_sift_octaves_cancellation(
        self, mock_ask_choice, mock_askint
    ) -> None:
        mock_askint.return_value = None
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.simpledialog.askinteger")
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=1)
    def test_returns_none_on_sift_scales_cancellation(
        self, mock_ask_choice, mock_askint
    ) -> None:
        # First call is n_octaves (4), second call is n_scales (None -> cancelled)
        mock_askint.side_effect = [4, None]
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.simpledialog.askfloat")
    @patch("tkinter.simpledialog.askinteger")
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=1)
    def test_returns_none_on_sift_sigma_cancellation(
        self, mock_ask_choice, mock_askint, mock_askfloat
    ) -> None:
        mock_askint.side_effect = [4, 3]
        mock_askfloat.return_value = None
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.simpledialog.askfloat")
    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("tkinter.simpledialog.askinteger", return_value=50)
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=0)
    def test_returns_none_on_alpha_cancellation(
        self, mock_ask_choice, mock_askint, mock_askyesno, mock_askfloat
    ) -> None:
        mock_askfloat.return_value = None
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askinteger", return_value=50)
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=0)
    def test_orb_binary_mask(
        self, mock_ask_choice, mock_askint, mock_askyesno
    ) -> None:
        source = self.document.current
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = source.convert(mode)
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, source.size)
                self.assertEqual(set(np.unique(result.image)), {0, 255})
                self.assertEqual(result.details["Method"], "ORB")
                self.assertFalse(result.details["Superimposed"])
                self.assertEqual(result.details["n_keypoints"], 50)
                self.assertGreater(result.details["Keypoints Detected"], 0)

    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askfloat", return_value=1.6)
    @patch("tkinter.simpledialog.askinteger", side_effect=[4, 3])
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=1)
    def test_sift_binary_mask(
        self, mock_ask_choice, mock_askint, mock_askfloat, mock_askyesno
    ) -> None:
        source = self.document.current
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = source.convert(mode)
                mock_askint.side_effect = [4, 3]
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, source.size)
                self.assertEqual(set(np.unique(result.image)), {0, 255})
                self.assertEqual(result.details["Method"], "SIFT")
                self.assertFalse(result.details["Superimposed"])
                self.assertEqual(result.details["n_octaves"], 4)
                self.assertEqual(result.details["n_scales"], 3)
                self.assertEqual(result.details["sigma"], 1.6)
                self.assertGreater(result.details["Keypoints Detected"], 0)

    @patch("tkinter.simpledialog.askfloat", return_value=0.5)
    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("tkinter.simpledialog.askinteger", return_value=50)
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=0)
    def test_orb_superimposed(
        self, mock_ask_choice, mock_askint, mock_askyesno, mock_askfloat
    ) -> None:
        source = self.document.current
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = source.convert(mode)
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                expected_mode = "RGBA" if mode == "RGBA" else "RGB"
                self.assertEqual(result.image.mode, expected_mode)
                self.assertEqual(result.image.size, source.size)
                self.assertTrue(result.details["Superimposed"])
                self.assertEqual(result.details["Alpha"], 0.5)

    @patch("tkinter.simpledialog.askfloat", side_effect=[1.6, 0.5])
    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("tkinter.simpledialog.askinteger", side_effect=[4, 3])
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=1)
    def test_sift_superimposed(
        self, mock_ask_choice, mock_askint, mock_askyesno, mock_askfloat
    ) -> None:
        result = self.tool.run(self.parent, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, self.document.current.size)
        self.assertTrue(result.details["Superimposed"])
        self.assertEqual(result.details["Alpha"], 0.5)

    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("tkinter.simpledialog.askinteger", return_value=20)
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=0)
    @patch("tkinter.simpledialog.askfloat")
    def test_superimposed_alpha_extremes(
        self, mock_askfloat, mock_ask_choice, mock_askint, mock_askyesno
    ) -> None:
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                mock_askfloat.return_value = alpha
                result = self.tool.run(self.parent, self.document)
                self.assertIsNotNone(result)
                arr = np.array(result.image)
                if alpha == 0.0:
                    np.testing.assert_array_equal(arr, np.array(self.document.current))
                else:
                    # At least one pixel has keypoint red [255, 0, 0]
                    red_pixels = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0)
                    self.assertTrue(np.any(red_pixels))

    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askinteger", return_value=50)
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=0)
    def test_blank_image_reports_zero_detected_keypoints(
        self, mock_ask_choice, mock_askint, mock_askyesno
    ) -> None:
        self.document.current = Image.new("RGB", (64, 64), color=(128, 128, 128))

        result = self.tool.run(self.parent, self.document)

        self.assertIsNotNone(result)
        self.assertEqual(result.details["Keypoints Detected"], 0)

    @patch("tkinter.messagebox.showerror")
    @patch(
        "forensics_app.tools.keypoint_detection.extract_keypoints",
        side_effect=ValueError("Test Extraction Error"),
    )
    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askinteger", return_value=50)
    @patch("forensics_app.tools.keypoint_detection.ask_choice", return_value=0)
    def test_processing_error_is_reported(
        self, mock_ask_choice, mock_askint, mock_askyesno, mock_extract, mock_error
    ) -> None:
        self.assertIsNone(self.tool.run(self.parent, self.document))
        mock_error.assert_called_once()


class KeypointDetectionFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.img = make_checkerboard_image(64, 8)

    def test_extract_keypoints_orb(self) -> None:
        keypoints, descriptors = extract_keypoints(self.img, method="orb", n_keypoints=30)
        self.assertGreater(len(keypoints), 0)
        self.assertLessEqual(len(keypoints), 30)
        self.assertEqual(keypoints.ndim, 2)
        self.assertEqual(keypoints.shape[1], 2)
        self.assertIsNotNone(descriptors)
        self.assertEqual(descriptors.shape[0], len(keypoints))
        self.assertEqual(descriptors.shape[1], 256)

    def test_extract_keypoints_sift(self) -> None:
        keypoints, descriptors = extract_keypoints(
            self.img, method="sift", n_octaves=4, n_scales=3, sigma=1.6
        )
        self.assertGreater(len(keypoints), 0)
        self.assertEqual(keypoints.ndim, 2)
        self.assertEqual(keypoints.shape[1], 2)
        self.assertIsNotNone(descriptors)
        self.assertEqual(descriptors.shape[0], len(keypoints))
        self.assertEqual(descriptors.shape[1], 128)

    def test_invalid_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_keypoints(self.img, method="invalid_method")

    def test_invalid_orb_parameters_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_keypoints(self.img, method="orb", n_keypoints=0)
        with self.assertRaises(ValueError):
            extract_keypoints(self.img, method="orb", n_keypoints=-10)

    def test_invalid_sift_parameters_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_keypoints(self.img, method="sift", n_octaves=0)
        with self.assertRaises(ValueError):
            extract_keypoints(self.img, method="sift", n_scales=0)
        with self.assertRaises(ValueError):
            extract_keypoints(self.img, method="sift", sigma=-0.5)
        with self.assertRaises(ValueError):
            extract_keypoints(self.img, method="sift", sigma=0.5)

    def test_blank_image_returns_empty(self) -> None:
        blank = Image.new("RGB", (64, 64), color=(128, 128, 128))
        keypoints, descriptors = extract_keypoints(blank, method="orb")
        self.assertEqual(len(keypoints), 0)
        self.assertIsNone(descriptors)

        keypoints_s, descriptors_s = extract_keypoints(blank, method="sift")
        self.assertEqual(len(keypoints_s), 0)
        self.assertIsNone(descriptors_s)

        mask = apply_keypoint_detection(blank, method="orb", superimpose=False)
        np.testing.assert_array_equal(np.asarray(mask), np.zeros((64, 64), dtype=np.uint8))

        overlay = apply_keypoint_detection(blank, method="orb", superimpose=True)
        self.assertEqual(overlay.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(overlay), np.asarray(blank))

    def test_apply_keypoint_detection_direct(self) -> None:
        mask = apply_keypoint_detection(self.img, method="orb", n_keypoints=20, superimpose=False)
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, (64, 64))
        self.assertEqual(set(np.unique(mask)), {0, 255})

        overlay = apply_keypoint_detection(
            self.img, method="sift", n_octaves=4, n_scales=3, sigma=1.6, superimpose=True, alpha=0.5
        )
        self.assertEqual(overlay.mode, "RGB")
        self.assertEqual(overlay.size, (64, 64))

    def test_marker_radius(self) -> None:
        # Default marker_radius=0 vs marker_radius=1
        mask0 = apply_keypoint_detection(
            self.img, method="orb", n_keypoints=10, superimpose=False, marker_radius=0
        )
        mask1 = apply_keypoint_detection(
            self.img, method="orb", n_keypoints=10, superimpose=False, marker_radius=1
        )
        arr0 = np.array(mask0)
        arr1 = np.array(mask1)
        self.assertGreater((arr1 == 255).sum(), (arr0 == 255).sum())

    def test_high_depth_image(self) -> None:
        arr = np.zeros((64, 64), dtype=np.float32)
        arr[16:48, 16:48] = 100.0
        img_f = Image.fromarray(arr, mode="F")

        kp, desc = extract_keypoints(img_f, method="orb", n_keypoints=10)
        self.assertIsInstance(kp, np.ndarray)

        out = apply_keypoint_detection(img_f, method="orb", n_keypoints=10, superimpose=True)
        self.assertEqual(out.mode, "RGB")

    def test_extract_keypoints_return_meta_orb(self) -> None:
        kp, desc, scales, orientations = extract_keypoints(
            self.img, method="orb", n_keypoints=20, return_meta=True
        )
        self.assertEqual(len(kp), len(scales))
        self.assertEqual(len(kp), len(orientations))
        if len(kp) > 0:
            self.assertEqual(scales.ndim, 1)
            self.assertEqual(orientations.ndim, 1)

    def test_extract_keypoints_return_meta_sift(self) -> None:
        kp, desc, scales, orientations = extract_keypoints(
            self.img, method="sift", n_octaves=4, n_scales=3, sigma=1.6, return_meta=True
        )
        self.assertEqual(len(kp), len(scales))
        self.assertEqual(len(kp), len(orientations))
        if len(kp) > 0:
            self.assertEqual(scales.ndim, 1)
            self.assertEqual(orientations.ndim, 1)

    def test_apply_keypoint_detection_rich_keypoints(self) -> None:
        out = apply_keypoint_detection(
            self.img, method="sift", draw_rich_keypoints=True, superimpose=True
        )
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, self.img.size)

    def test_scale_keypoint_radii(self) -> None:
        self.assertIsNone(scale_keypoint_radii(None))
        arr = np.array([1.0, 2.0, 3.0])
        orb_scaled = scale_keypoint_radii(arr, method="orb", orb_radius_multiplier=4.0)
        np.testing.assert_array_almost_equal(orb_scaled, [4.0, 8.0, 12.0])

        sift_scaled = scale_keypoint_radii(arr, method="sift")
        np.testing.assert_array_equal(sift_scaled, arr)

    def test_convert_keypoint_orientations(self) -> None:
        self.assertIsNone(convert_keypoint_orientations(None))
        arr = np.array([0.0, np.pi / 2.0])
        # SIFT convention: 0 rad points down, pi/2 rad points right
        # Cartesian renderer convention: 0 rad points right, pi/2 rad points down
        converted = convert_keypoint_orientations(arr, method="sift")
        np.testing.assert_array_almost_equal(converted, [np.pi / 2.0, 0.0])

        # ORB convention: already matches Cartesian
        orb_converted = convert_keypoint_orientations(arr, method="orb")
        np.testing.assert_array_equal(orb_converted, arr)

    def test_sift_ray_direction_rendered(self) -> None:
        from forensics_app.tools.image_utils import render_keypoint_overlay

        im = Image.new("RGB", (30, 30), (0, 0, 0))
        coords = np.array([[15.0, 15.0]])
        scales = np.array([5.0])
        # In SIFT, angle 0 is down (gradient_row > 0)
        sift_angle = np.array([0.0])
        cartesian_angle = convert_keypoint_orientations(sift_angle, method="sift")

        mask = render_keypoint_overlay(
            im, coords, scales=scales, orientations=cartesian_angle, superimpose=False
        )
        arr = np.array(mask)
        # Ray should point down: intermediate pixels inside the circle along row axis are drawn
        self.assertEqual(arr[18, 15], 255)
        # Intermediate pixels along col axis (pointing right) are NOT drawn (circle is hollow)
        self.assertEqual(arr[15, 18], 0)


if __name__ == "__main__":
    unittest.main()
