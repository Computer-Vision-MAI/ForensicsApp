"""Unit tests for the FaceDetectionTool and YuNet/Haar face detectors.

The YuNet weights and the Viola-Jones cascade live in the git-ignored ``models/``
folder, so every test mocks them: nothing is downloaded or written to the repository.
"""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.dialogs import TaskCancelled
from forensics_app.tools.face_detection import (
    FaceDetectionTool,
    detect_faces_with_detector,
    detect_faces_yunet,
    ensure_yunet_model,
    scale_face_results,
)

from face_fixtures import build_simple_cascade

FD = "forensics_app.tools.face_detection"

# 1 face: x=20, y=30, w=50, h=60, 5 landmarks (10 floats), conf=0.92
FACE_ROW = np.array(
    [20, 30, 50, 60, 35, 45, 55, 45, 45, 60, 35, 75, 55, 75, 0.92],
    dtype=np.float32,
)


def _fake_detector(faces: np.ndarray | None) -> MagicMock:
    """Return a stand-in for cv2.FaceDetectorYN whose detect() yields ``faces``."""
    detector = MagicMock()
    detector.detect.return_value = (0 if faces is None else len(faces), faces)
    return detector


def _haar_dialog(**overrides: object) -> MagicMock:
    """Return a confirmed FaceDetectionDialog mock configured for the Haar method."""
    dlg = MagicMock()
    dlg.user_confirmed = True
    dlg.selected_method = "haar"
    dlg.selected_source = "current"
    dlg.cascade_mode = "pretrained"
    dlg.max_haar_features = 10
    dlg.preserve_history = True
    for name, value in overrides.items():
        setattr(dlg, name, value)
    return dlg


def _document(size: tuple[int, int], color: tuple[int, int, int]) -> ImageDocument:
    """Return a loaded document holding a flat RGB image."""
    doc = ImageDocument()
    img = Image.new("RGB", size, color=color)
    doc.original = img.copy()
    doc.current = img.copy()
    return doc


class TestEnsureYunetModel(unittest.TestCase):
    """The YuNet download is atomic and skipped when the weights already exist."""

    def setUp(self) -> None:
        """Use a temporary model path for each test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_path = Path(self._tmp.name) / "models" / "yunet.onnx"

    def test_downloads_into_place(self) -> None:
        """A missing model is downloaded and written to the final path."""
        payload = b"x" * 150_000
        with patch(f"{FD}.urllib.request.urlopen", return_value=io.BytesIO(payload)) as urlopen:
            result = ensure_yunet_model(self.model_path)
        self.assertEqual(result, self.model_path)
        self.assertEqual(self.model_path.read_bytes(), payload)
        self.assertIn("timeout", urlopen.call_args.kwargs)
        self.assertEqual(list(self.model_path.parent.glob("*.part")), [])

    def test_existing_model_is_not_downloaded_again(self) -> None:
        """A complete local model is reused without any network access."""
        self.model_path.parent.mkdir(parents=True)
        self.model_path.write_bytes(b"y" * 150_000)
        with patch(f"{FD}.urllib.request.urlopen") as urlopen:
            ensure_yunet_model(self.model_path)
        urlopen.assert_not_called()

    def test_failed_download_keeps_previous_file_and_cleans_up(self) -> None:
        """An interrupted download leaves no partial file and keeps the old one."""
        self.model_path.parent.mkdir(parents=True)
        self.model_path.write_bytes(b"old")  # too small: triggers a download
        with patch(f"{FD}.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(OSError):
                ensure_yunet_model(self.model_path)
        self.assertEqual(self.model_path.read_bytes(), b"old")
        self.assertEqual(list(self.model_path.parent.glob("*.part")), [])


@patch(f"{FD}.ensure_yunet_model")
class TestYunetDetection(unittest.TestCase):
    """YuNet result parsing and annotation, with the OpenCV detector mocked."""

    def test_detect_faces_yunet_no_faces(self, _ensure: MagicMock) -> None:
        """Verify that YuNet handles images with no faces gracefully."""
        blank_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch(f"{FD}.cv2.FaceDetectorYN.create", return_value=_fake_detector(None)):
            annotated, faces = detect_faces_yunet(blank_bgr)
        self.assertEqual(len(faces), 0)
        self.assertEqual(annotated.shape, (100, 100, 3))

    def test_detect_faces_yunet_structure(self, _ensure: MagicMock) -> None:
        """Verify that YuNet returns the correct dictionary structure (box, conf, landmarks)."""
        blank_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        with patch(f"{FD}.cv2.FaceDetectorYN.create", return_value=_fake_detector(np.array([FACE_ROW]))):
            annotated, faces = detect_faces_yunet(blank_bgr)
        self.assertEqual(len(faces), 1)
        f0 = faces[0]
        self.assertEqual(f0["box"], (20, 30, 50, 60))
        self.assertAlmostEqual(f0["confidence"], 0.92, places=2)
        self.assertEqual(len(f0["landmarks"]), 5)
        self.assertIn("Right eye", f0["landmarks"])
        self.assertIn("Nose", f0["landmarks"])
        self.assertTrue(np.any(annotated != 0))  # box/landmarks were drawn

    def test_reused_detector_follows_frame_size(self, _ensure: MagicMock) -> None:
        """One detector serves frames of different sizes by updating its input size."""
        detector = _fake_detector(None)
        detect_faces_with_detector(detector, np.zeros((120, 160, 3), dtype=np.uint8))
        detect_faces_with_detector(detector, np.zeros((60, 80, 3), dtype=np.uint8))
        self.assertEqual(
            [c.args[0] for c in detector.setInputSize.call_args_list], [(160, 120), (80, 60)]
        )

    @patch(f"{FD}.FaceDetectionDialog")
    def test_face_detection_tool_yunet(self, mock_dialog_cls: MagicMock, _ensure: MagicMock) -> None:
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

        doc = _document((100, 100), (100, 100, 100))
        with patch(f"{FD}.cv2.FaceDetectorYN.create", return_value=_fake_detector(None)):
            result = FaceDetectionTool().run(parent=MagicMock(), document=doc)

        self.assertIsNotNone(result)
        self.assertIn("YuNet", result.message)
        self.assertEqual(result.image.size, (100, 100))
        self.assertIn("Score Threshold", result.details)


class TestScaleFaceResults(unittest.TestCase):
    """Webcam detections are reported at the scale of the captured preview."""

    def test_scales_boxes_and_landmarks(self) -> None:
        """Coordinates are multiplied by the scale; other fields are kept."""
        faces = [{"box": (100, 50, 40, 60), "confidence": 0.9, "landmarks": {"Nose": (120, 80)}}]
        scaled = scale_face_results(faces, 0.5)
        self.assertEqual(scaled[0]["box"], (50, 25, 20, 30))
        self.assertEqual(scaled[0]["landmarks"], {"Nose": (60, 40)})
        self.assertEqual(scaled[0]["confidence"], 0.9)
        self.assertEqual(faces[0]["box"], (100, 50, 40, 60))  # input untouched

    def test_scale_one_is_identity(self) -> None:
        """No resize means the detections are returned unchanged."""
        faces = [{"box": (1, 2, 3, 4), "confidence": 0.5, "landmarks": {}}]
        self.assertIs(scale_face_results(faces, 1.0), faces)


class TestComputeFaceDrawingParams(unittest.TestCase):
    """Annotation sizes scale with the face size."""

    def test_compute_face_drawing_params(self) -> None:
        """Verify dynamic scaling of line thickness, font, and landmark radius."""
        from forensics_app.tools.face_detection import compute_face_drawing_params

        tiny = compute_face_drawing_params(20, 25)
        self.assertEqual(tiny["line_thickness"], 1)
        self.assertLessEqual(tiny["font_scale"], 0.4)
        self.assertEqual(tiny["landmark_radius"], 2)

        large = compute_face_drawing_params(300, 350)
        self.assertGreaterEqual(large["line_thickness"], 4)
        self.assertGreaterEqual(large["font_scale"], 1.0)
        self.assertGreaterEqual(large["landmark_radius"], 6)


@patch(f"{FD}.messagebox.showinfo")
@patch(f"{FD}.FaceDetectionDialog")
class TestViolaJonesTool(unittest.TestCase):
    """FaceDetectionTool with the Haar cascade, using a small hand-built cascade."""

    def test_standalone_image_without_candidates(self, mock_dialog_cls: MagicMock, _info: MagicMock) -> None:
        """With no YuNet candidate the whole image is evaluated as one window."""
        mock_dialog_cls.return_value = _haar_dialog()
        doc = _document((50, 50), (150, 150, 150))
        with patch(f"{FD}.load_cascade_model", return_value=build_simple_cascade()), patch(
            f"{FD}.detect_faces_yunet", return_value=(np.zeros((50, 50, 3), np.uint8), [])
        ):
            result = FaceDetectionTool().run(parent=MagicMock(), document=doc)

        self.assertIn("Viola-Jones", result.message)
        self.assertEqual(result.image.size, (50, 50))
        self.assertIn("Verdict", result.details)
        self.assertIn("Stages Passed", result.details)

    def test_candidate_boxes_are_checked_by_the_cascade(self, mock_dialog_cls: MagicMock, _info: MagicMock) -> None:
        """Each YuNet candidate gets a cascade verdict in the details."""
        mock_dialog_cls.return_value = _haar_dialog()
        doc = _document((120, 120), (90, 90, 90))
        candidate = {"box": (20, 20, 60, 60), "confidence": 0.8, "landmarks": {}}
        with patch(f"{FD}.load_cascade_model", return_value=build_simple_cascade()), patch(
            f"{FD}.detect_faces_yunet", return_value=(np.zeros((120, 120, 3), np.uint8), [candidate])
        ):
            result = FaceDetectionTool().run(parent=MagicMock(), document=doc)

        self.assertIn("Candidate #1", result.details)
        self.assertIn("Viola-Jones Confirmed", result.details)
        self.assertNotIn("Verdict", result.details)
        self.assertEqual(result.image.size, (120, 120))

    def test_missing_cascade_is_trained_behind_a_progress_window(
        self, mock_dialog_cls: MagicMock, _info: MagicMock
    ) -> None:
        """Without a saved cascade, training runs through run_with_progress."""
        mock_dialog_cls.return_value = _haar_dialog()
        doc = _document((50, 50), (150, 150, 150))

        def run_now(parent, title, task):
            """Run the progress task immediately, without a progress window."""
            return task(lambda *args: None, lambda: False)

        with patch(f"{FD}.load_cascade_model", return_value=None), patch(
            f"{FD}.run_with_progress", side_effect=run_now
        ) as progress, patch(
            f"{FD}.ensure_default_cascade_model", return_value=build_simple_cascade()
        ) as ensure, patch(f"{FD}.detect_faces_yunet", return_value=(None, [])):
            result = FaceDetectionTool().run(parent=MagicMock(), document=doc)

        progress.assert_called_once()
        ensure.assert_called_once()
        self.assertIsNotNone(result)

    def test_cancelled_training_cancels_the_tool(self, mock_dialog_cls: MagicMock, _info: MagicMock) -> None:
        """Cancelling the training progress window returns None (tool cancelled)."""
        mock_dialog_cls.return_value = _haar_dialog(cascade_mode="train_new", cascade_stages_str="2, 5", train_samples=20)
        doc = _document((50, 50), (150, 150, 150))
        with patch(f"{FD}.run_with_progress", side_effect=TaskCancelled()):
            self.assertIsNone(FaceDetectionTool().run(parent=MagicMock(), document=doc))


if __name__ == "__main__":
    unittest.main()
