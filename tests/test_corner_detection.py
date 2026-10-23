"""Tests for Set 5.1: Corner Detection (Harris and FAST)."""

import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from forensics_app.tools import CornerDetectionTool
from forensics_app.tools.corner_detection import apply_corner_detection, detect_corners


class CornerDetectionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        # Create a simple test image: a 10x10 grayscale image with a white square in the center.
        self.document.current = Image.new("RGB", (10, 10), color=(0, 0, 0))
        for i in range(3, 7):
            for j in range(3, 7):
                self.document.current.putpixel((i, j), (255, 255, 255))

        self.tool = CornerDetectionTool()
        self.parent = None  # Mock parent for Tkinter

    @patch("forensics_app.tools.corner_detection.ask_choice")
    def test_returns_none_on_choice_cancellation(self, mock_ask_choice) -> None:
        mock_ask_choice.return_value = None
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.simpledialog.askfloat")
    @patch("forensics_app.tools.corner_detection.ask_choice")
    def test_returns_none_on_threshold_cancellation(
        self, mock_ask_choice, mock_askfloat
    ) -> None:
        mock_ask_choice.return_value = 0  # Harris
        mock_askfloat.return_value = None
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.simpledialog.askfloat")
    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("forensics_app.tools.corner_detection.ask_choice", return_value=0)
    def test_returns_none_on_alpha_cancellation(
        self, mock_ask_choice, mock_askyesno, mock_askfloat
    ) -> None:
        # First call is threshold (0.1), second call is alpha (None -> cancelled)
        mock_askfloat.side_effect = [0.1, None]
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askfloat", return_value=0.1)
    @patch("forensics_app.tools.corner_detection.ask_choice", return_value=0)
    def test_harris_binary_mask(
        self, mock_ask_choice, mock_askfloat, mock_askyesno
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
                self.assertEqual(result.details["Method"], "HARRIS")
                self.assertFalse(result.details["Superimposed"])
                self.assertEqual(result.details["Corners Detected"], 4)

    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askfloat", return_value=0.1)
    @patch("forensics_app.tools.corner_detection.ask_choice", return_value=1)
    def test_fast_binary_mask(
        self, mock_ask_choice, mock_askfloat, mock_askyesno
    ) -> None:
        # Create a 30x30 image with square from 10 to 19 for FAST detection
        img = Image.new("RGB", (30, 30), color=(0, 0, 0))
        for i in range(10, 20):
            for j in range(10, 20):
                img.putpixel((i, j), (255, 255, 255))
        self.document.current = img

        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = img.convert(mode)
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, img.size)
                self.assertEqual(set(np.unique(result.image)), {0, 255})
                self.assertEqual(result.details["Method"], "FAST")
                self.assertFalse(result.details["Superimposed"])

    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("tkinter.simpledialog.askfloat", side_effect=[0.1, 0.5])
    @patch("forensics_app.tools.corner_detection.ask_choice", return_value=0)
    def test_harris_superimposed(
        self, mock_ask_choice, mock_askfloat, mock_askyesno
    ) -> None:
        source = self.document.current
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = source.convert(mode)
                mock_askfloat.side_effect = [0.1, 0.5]
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                expected_mode = "RGBA" if mode == "RGBA" else "RGB"
                self.assertEqual(result.image.mode, expected_mode)
                self.assertEqual(result.image.size, source.size)
                self.assertTrue(result.details["Superimposed"])
                self.assertEqual(result.details["Alpha"], 0.5)

    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askfloat", return_value=0.1)
    @patch("forensics_app.tools.corner_detection.ask_choice", return_value=0)
    def test_binary_mask_corner_locations(
        self, mock_ask_choice, mock_askfloat, mock_askyesno
    ) -> None:
        result = self.tool.run(self.parent, self.document)
        self.assertIsNotNone(result)
        arr = np.array(result.image)

        # Center of the white square (5,5) and corner of image (0,0) should NOT be corners
        self.assertEqual(arr[5, 5], 0)
        self.assertEqual(arr[0, 0], 0)

        # The four corners of the white square should be 255
        self.assertEqual(arr[3, 3], 255)
        self.assertEqual(arr[3, 6], 255)
        self.assertEqual(arr[6, 3], 255)
        self.assertEqual(arr[6, 6], 255)

    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("tkinter.simpledialog.askfloat", side_effect=[0.1, 0.5])
    @patch("forensics_app.tools.corner_detection.ask_choice", return_value=0)
    def test_superimposed_ponderated_colors(
        self, mock_ask_choice, mock_askfloat, mock_askyesno
    ) -> None:
        source = Image.new("RGB", (10, 10), (10, 30, 50))
        source.paste((200, 180, 160), (3, 3, 7, 7))
        self.document.current = source

        result = self.tool.run(self.parent, self.document)
        self.assertIsNotNone(result)
        arr = np.array(result.image)
        self.assertEqual(result.image.mode, "RGB")

        # Non-corner pixels must remain untouched
        np.testing.assert_array_equal(arr[5, 5], [200, 180, 160])
        np.testing.assert_array_equal(arr[0, 0], [10, 30, 50])

        # Corner pixel (3, 3) must be blended with red [255, 0, 0] at alpha 0.5:
        # 200 * 0.5 + 255 * 0.5 = 227.5 -> uint8 truncation gives 227
        # 180 * 0.5 + 0 = 90
        # 160 * 0.5 + 0 = 80
        np.testing.assert_array_equal(arr[3, 3], [227, 90, 80])

    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("tkinter.simpledialog.askfloat")
    @patch("forensics_app.tools.corner_detection.ask_choice", return_value=0)
    def test_superimposed_alpha_extremes(
        self, mock_ask_choice, mock_askfloat, mock_askyesno
    ) -> None:
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                mock_askfloat.side_effect = [0.1, alpha]
                result = self.tool.run(self.parent, self.document)
                self.assertIsNotNone(result)
                arr = np.array(result.image)
                if alpha == 0.0:
                    np.testing.assert_array_equal(arr, np.array(self.document.current))
                else:
                    np.testing.assert_array_equal(arr[3, 3], [255, 0, 0])

    @patch("tkinter.messagebox.showerror")
    @patch("forensics_app.tools.corner_detection.detect_corners", side_effect=ValueError("Test Error"))
    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askfloat", return_value=0.1)
    @patch("forensics_app.tools.corner_detection.ask_choice", return_value=0)
    def test_processing_error_is_reported(
        self, mock_ask_choice, mock_askfloat, mock_askyesno, mock_detect, mock_error
    ) -> None:
        self.assertIsNone(self.tool.run(self.parent, self.document))
        mock_error.assert_called_once()


class CornerDetectionFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        for i in range(3, 7):
            for j in range(3, 7):
                self.img.putpixel((i, j), (255, 255, 255))

    def test_detect_corners_harris(self) -> None:
        coords = detect_corners(self.img, method="harris", threshold=0.1)
        self.assertEqual(len(coords), 4)
        expected = {(3, 3), (3, 6), (6, 3), (6, 6)}
        actual = {tuple(p) for p in coords}
        self.assertEqual(expected, actual)

    def test_detect_corners_fast(self) -> None:
        # 30x30 image with square from 10 to 19
        img = Image.new("RGB", (30, 30), color=(0, 0, 0))
        for i in range(10, 20):
            for j in range(10, 20):
                img.putpixel((i, j), (255, 255, 255))
        coords = detect_corners(img, method="fast", threshold=0.1)
        self.assertEqual(len(coords), 4)
        expected = {(10, 10), (10, 19), (19, 10), (19, 19)}
        actual = {tuple(p) for p in coords}
        self.assertEqual(expected, actual)

    def test_invalid_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_corners(self.img, method="unsupported_method")

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_corners(self.img, threshold=-0.1)
        with self.assertRaises(ValueError):
            detect_corners(self.img, threshold=1.5)

        expected = {(3, 3), (3, 6), (6, 3), (6, 6)}
        threshold_zero = detect_corners(self.img, threshold=0.0)
        threshold_point_one = detect_corners(self.img, threshold=0.1)

        self.assertTrue(expected.issubset({tuple(point) for point in threshold_zero}))
        self.assertGreaterEqual(len(threshold_zero), 4)
        self.assertGreaterEqual(len(threshold_zero), len(threshold_point_one))
        threshold_one = detect_corners(self.img, threshold=1.0)
        self.assertEqual(len(threshold_one), 4)
        self.assertEqual({tuple(point) for point in threshold_one}, expected)

    def test_zero_threshold_excludes_negative_harris_response(self) -> None:
        # Straight edge produces negative Harris responses along the boundary.
        # threshold=0.0 (passing threshold_rel=0.0) suppresses negative responses.
        im = np.zeros((15, 15), dtype=np.uint8)
        im[:, 7:] = 255
        edge_img = Image.fromarray(im, mode="L")
        coords = detect_corners(edge_img, method="harris", threshold=0.0)

        # Any detected corner must have a strictly positive Harris response
        gray = np.array(edge_img) / 255.0
        resp = feature.corner_harris(gray)
        for r, c in coords:
            self.assertGreater(resp[r, c], 0.0)

    def test_apply_corner_detection_direct(self) -> None:
        mask = apply_corner_detection(self.img, method="harris", threshold=0.1, superimpose=False)
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, (10, 10))
        self.assertEqual(set(np.unique(mask)), {0, 255})

        overlay = apply_corner_detection(
            self.img, method="harris", threshold=0.1, superimpose=True, alpha=0.6
        )
        self.assertEqual(overlay.mode, "RGB")
        self.assertEqual(overlay.size, (10, 10))

    def test_marker_radius_dilation(self) -> None:
        # With marker_radius=1, neighboring pixels around corner (3, 3) should also be active
        mask = apply_corner_detection(
            self.img, method="harris", threshold=0.1, superimpose=False, marker_radius=1
        )
        arr = np.array(mask)
        self.assertEqual(arr[3, 3], 255)
        self.assertEqual(arr[3, 2], 255)
        self.assertEqual(arr[2, 3], 255)

    def test_high_depth_image(self) -> None:
        # Test mode I and F
        arr = np.zeros((10, 10), dtype=np.int32)
        arr[3:7, 3:7] = 50000
        img_i = Image.fromarray(arr, mode="I")

        coords = detect_corners(img_i, method="harris", threshold=0.1)
        self.assertEqual(len(coords), 4)

        overlay = apply_corner_detection(img_i, method="harris", threshold=0.1, superimpose=True)
        self.assertEqual(overlay.mode, "RGB")

        # Test mode F
        arr_f = np.zeros((10, 10), dtype=np.float32)
        arr_f[3:7, 3:7] = 50000.0
        img_f = Image.fromarray(arr_f, mode="F")

        coords_f = detect_corners(img_f, method="harris", threshold=0.1)
        self.assertEqual(len(coords_f), 4)

        overlay_f = apply_corner_detection(img_f, method="harris", threshold=0.1, superimpose=True)
        self.assertEqual(overlay_f.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
