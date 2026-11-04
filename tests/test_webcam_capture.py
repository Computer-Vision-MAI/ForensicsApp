"""Unit tests for the clean WebcamCaptureTool."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.webcam_capture import WebcamCaptureTool, WebcamPhotoDialog


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
        self.assertIn("Source", result.details)

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
        self.assertIn("Saved to Disk", result.details)


if __name__ == "__main__":
    unittest.main()
