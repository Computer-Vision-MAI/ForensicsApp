"""Unit tests for the clean WebcamCaptureTool."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.webcam_capture import (
    WebcamCaptureTool,
    WebcamPhotoDialog,
    compute_lfw_crop_box,
)


class TestComputeLfwCropBox(unittest.TestCase):
    """The LFW crop keeps a 50:37 shape and stays inside the frame."""

    IMAGE = (640, 480)

    def _assert_inside(self, box: tuple[int, int, int, int]) -> None:
        """Assert that the crop box is non-empty and lies inside the image."""
        x1, y1, x2, y2 = box
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, self.IMAGE[0])
        self.assertLessEqual(y2, self.IMAGE[1])
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_centered_face_matches_previous_behaviour(self) -> None:
        """An interior face gives the same box as the original formula."""
        bx, by, bw, bh = 250, 150, 120, 160
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        crop_h = bh * 1.35
        crop_w = crop_h * (37.0 / 50.0)
        expected = (
            int(round(cx - crop_w / 2.0)),
            int(round(cy - crop_h / 2.0)),
            int(round(cx + crop_w / 2.0)),
            int(round(cy + crop_h / 2.0)),
        )
        self.assertEqual(compute_lfw_crop_box((bx, by, bw, bh), self.IMAGE), expected)

    def test_faces_at_each_border_keep_the_aspect_ratio(self) -> None:
        """Crops near a border are shifted inside instead of being cut."""
        faces = {
            "left": (0, 160, 100, 140),
            "right": (560, 160, 80, 140),
            "top": (270, 0, 100, 140),
            "bottom": (270, 400, 100, 80),
        }
        for name, face in faces.items():
            with self.subTest(border=name):
                box = compute_lfw_crop_box(face, self.IMAGE)
                self._assert_inside(box)
                x1, y1, x2, y2 = box
                self.assertAlmostEqual((y2 - y1) / (x2 - x1), 50.0 / 37.0, delta=0.03)
                self.assertAlmostEqual(y2 - y1, face[3] * 1.35, delta=1.0)

    def test_face_larger_than_frame_is_shrunk_to_fit(self) -> None:
        """A crop bigger than the frame is scaled down, keeping the LFW shape."""
        box = compute_lfw_crop_box((0, 0, 640, 480), self.IMAGE)
        self._assert_inside(box)
        x1, y1, x2, y2 = box
        self.assertEqual(y2 - y1, 480)
        self.assertAlmostEqual((y2 - y1) / (x2 - x1), 50.0 / 37.0, delta=0.01)

    def test_one_pixel_box(self) -> None:
        """A degenerate 1x1 detection still gives a non-empty box inside the frame."""
        for face in [(0, 0, 1, 1), (639, 479, 1, 1), (320, 240, 1, 1)]:
            with self.subTest(face=face):
                self._assert_inside(compute_lfw_crop_box(face, self.IMAGE))


class TestWebcamCapture(unittest.TestCase):
    """Test suite for clean webcam capture and tool integration."""

    def test_tool_metadata(self) -> None:
        """Verify WebcamCaptureTool properties and registration attributes."""
        tool = WebcamCaptureTool()
        self.assertEqual(tool.tool_id, "webcam_capture")
        self.assertEqual(tool.category, "Face Analysis")
        self.assertFalse(tool.requires_image)

    @patch("forensics_app.tools.webcam_capture.WebcamPhotoDialog")
    def test_tool_run_cancelled(self, mock_dialog_cls: MagicMock) -> None:
        """Verify behavior when user closes the webcam window without capturing."""
        mock_dlg = MagicMock()
        mock_dlg.captured_image = None
        mock_dialog_cls.return_value = mock_dlg

        doc = ImageDocument()
        tool = WebcamCaptureTool()
        result = tool.run(parent=MagicMock(), document=doc)
        self.assertIsNone(result)

    @patch("forensics_app.tools.webcam_capture.WebcamPhotoDialog")
    def test_tool_run_full_frame(self, mock_dialog_cls: MagicMock) -> None:
        """Verify capturing full frame and initializing document."""
        mock_dlg = MagicMock()
        fake_photo = Image.new("RGB", (640, 480), color=(100, 150, 200))
        mock_dlg.captured_image = fake_photo
        mock_dlg.capture_mode = "Full Frame"
        mock_dlg.saved_path = None
        mock_dialog_cls.return_value = mock_dlg

        doc = ImageDocument()
        tool = WebcamCaptureTool()
        result = tool.run(parent=MagicMock(), document=doc)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, (640, 480))
        self.assertIn("Full Frame", result.message)
        self.assertEqual(doc.current.size, (640, 480))
        self.assertTrue(result.preview_only)
        self.assertIn("Source", result.details)

    @patch("forensics_app.tools.webcam_capture.WebcamPhotoDialog")
    def test_tool_run_with_loaded_document(self, mock_dialog_cls: MagicMock) -> None:
        """With an image already open, the photo is applied as a new step (not a preview)."""
        mock_dlg = MagicMock()
        mock_dlg.captured_image = Image.new("RGB", (320, 240), color=(10, 20, 30))
        mock_dlg.capture_mode = "Full Frame"
        mock_dlg.saved_path = None
        mock_dialog_cls.return_value = mock_dlg

        for mode in ("RGB", "RGBA", "L", "P"):
            with self.subTest(mode=mode):
                doc = ImageDocument()
                existing = Image.new("RGB", (50, 40), color=(200, 100, 50)).convert(mode)
                doc.original = existing.copy()
                doc.current = existing.copy()

                result = WebcamCaptureTool().run(parent=MagicMock(), document=doc)

                self.assertFalse(result.preview_only)
                self.assertEqual(result.image.size, (320, 240))
                self.assertEqual(doc.current.mode, mode)
                self.assertEqual(doc.current.size, (50, 40))

    @patch("forensics_app.tools.webcam_capture.WebcamPhotoDialog")
    def test_tool_run_cropped_face(self, mock_dialog_cls: MagicMock) -> None:
        """Verify capturing cropped face for face recognition."""
        mock_dlg = MagicMock()
        fake_face = Image.new("RGB", (185, 250), color=(180, 120, 100))
        mock_dlg.captured_image = fake_face
        mock_dlg.capture_mode = "Cropped Face (LFW 50:37 Format)"
        mock_dlg.saved_path = Path("/tmp/my_face.png")
        mock_dialog_cls.return_value = mock_dlg

        doc = ImageDocument()
        tool = WebcamCaptureTool()
        result = tool.run(parent=MagicMock(), document=doc)

        self.assertIsNotNone(result)
        self.assertEqual(result.image.size, (185, 250))
        self.assertIn("Cropped Face", result.message)
        self.assertTrue(result.preview_only)
        self.assertIn("Saved to Disk", result.details)


if __name__ == "__main__":
    unittest.main()
