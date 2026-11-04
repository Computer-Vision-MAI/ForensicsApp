"""Unit tests for the FaceDetectionTool and YuNet/Haar face detectors."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.face_detection import (
    FaceDetectionDialog,
    FaceDetectionTool,
    detect_faces_yunet,
    ensure_yunet_model,
)


class TestFaceDetection(unittest.TestCase):
    """Test suite for face detection algorithms and tool integration."""

    def test_ensure_yunet_model(self) -> None:
        """Verify that the YuNet ONNX weights exist locally."""
        model_path = ensure_yunet_model()
        self.assertTrue(model_path.exists())
        self.assertGreater(model_path.stat().st_size, 100_000)

    def test_detect_faces_yunet_no_faces(self) -> None:
        """Verify that YuNet handles images with no faces gracefully."""
        blank_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        annotated, faces = detect_faces_yunet(blank_bgr)
        self.assertEqual(len(faces), 0)
        self.assertEqual(annotated.shape, (100, 100, 3))

    def test_compute_face_drawing_params(self) -> None:
        """Verify dynamic scaling of line thickness, font, and landmark radius."""
        from forensics_app.tools.face_detection import compute_face_drawing_params

        # Tiny face
        tiny = compute_face_drawing_params(20, 25)
        self.assertEqual(tiny["line_thickness"], 1)
        self.assertLessEqual(tiny["font_scale"], 0.4)
        self.assertEqual(tiny["landmark_radius"], 2)

        # Large face
        large = compute_face_drawing_params(300, 350)
        self.assertGreaterEqual(large["line_thickness"], 4)
        self.assertGreaterEqual(large["font_scale"], 1.0)
        self.assertGreaterEqual(large["landmark_radius"], 6)

    def test_detect_faces_yunet_structure(self) -> None:
        """Verify that YuNet returns the correct dictionary structure (box, conf, landmarks)."""
        # Test on a small synthetic face or mock detect output
        blank_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        with patch("cv2.FaceDetectorYN.create") as mock_create:
            mock_detector = MagicMock()
            # 1 face: x=20, y=30, w=50, h=60, 5 landmarks (10 floats), conf=0.92
            face_row = np.array(
                [20, 30, 50, 60, 35, 45, 55, 45, 45, 60, 35, 75, 55, 75, 0.92],
                dtype=np.float32,
            )
            mock_detector.detect.return_value = (1, np.array([face_row]))
            mock_create.return_value = mock_detector

            annotated, faces = detect_faces_yunet(blank_bgr)
            self.assertEqual(len(faces), 1)
            f0 = faces[0]
            self.assertIn("box", f0)
            self.assertIn("confidence", f0)
            self.assertIn("landmarks", f0)
            self.assertAlmostEqual(f0["confidence"], 0.92, places=2)
            self.assertEqual(len(f0["landmarks"]), 5)
            self.assertIn("Right eye", f0["landmarks"])
            self.assertIn("Nose", f0["landmarks"])

    @patch("forensics_app.tools.face_detection.FaceDetectionDialog")
    def test_face_detection_tool_haar(self, mock_dialog_cls: MagicMock) -> None:
        """Verify FaceDetectionTool executing with Viola-Jones Haar cascade."""
        mock_dlg = MagicMock()
        mock_dlg.user_confirmed = True
        mock_dlg.selected_method = "haar"
        mock_dlg.selected_source = "current"
        mock_dlg.cascade_mode = "pretrained"
        mock_dlg.max_haar_features = 10
        mock_dlg.preserve_history = True
        mock_dialog_cls.return_value = mock_dlg

        doc = ImageDocument()
        img = Image.new("RGB", (50, 50), color=(150, 150, 150))
        doc.original = img.copy()
        doc.current = img.copy()

        tool = FaceDetectionTool()
        with patch("forensics_app.tools.face_detection.messagebox.showinfo"):
            result = tool.run(parent=MagicMock(), document=doc)

        self.assertIsNotNone(result)
        self.assertIn("Viola-Jones", result.message)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, (50, 50))
        self.assertIn("Verdict", result.details)
        self.assertIn("Stages Passed", result.details)

    @patch("forensics_app.tools.face_detection.FaceDetectionDialog")
    def test_face_detection_tool_yunet(self, mock_dialog_cls: MagicMock) -> None:
        """Verify FaceDetectionTool executing with YuNet and custom thresholds."""
        mock_dlg = MagicMock()
        mock_dlg.user_confirmed = True
        mock_dlg.selected_method = "yunet"
        mock_dlg.selected_source = "current"
        mock_dlg.score_threshold = 0.60
        mock_dlg.nms_threshold = 0.30
        mock_dlg.top_k = 5000
        mock_dlg.preserve_history = True
        mock_dialog_cls.return_value = mock_dlg

        doc = ImageDocument()
        img = Image.new("RGB", (100, 100), color=(100, 100, 100))
        doc.original = img.copy()
        doc.current = img.copy()

        tool = FaceDetectionTool()
        result = tool.run(parent=MagicMock(), document=doc)

        self.assertIsNotNone(result)
        self.assertIn("YuNet", result.message)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.image.size, (100, 100))
        self.assertIn("Score Threshold", result.details)


if __name__ == "__main__":
    unittest.main()
