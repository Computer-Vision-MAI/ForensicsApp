"""Unit tests for the Face Recognition tool (Set 6.3)."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.face_recognition import (
    FaceRecognitionDialog,
    FaceRecognitionTool,
)
from forensics_app.tools.face_recognition_core import (
    classify_face,
    load_face_model,
    make_recognition_gallery,
    prepare_lfw_dataset,
    project_face_image,
    train_eigenfaces_model,
)

from face_fixtures import N_COMPONENTS, build_lfw_like_dataset


class TestFaceRecognition(unittest.TestCase):
    """Test suite for projection, classification strategies, and tool workflow."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build a synthetic dataset and train a model once for the test suite.

        ``data/`` and ``models/`` are git-ignored (absent in CI), so everything lives
        in a temporary directory and the tool's module-level defaults are patched.
        """
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.data_dir = root / "data" / "lfw_faces"
        cls.model_dir = root / "models" / "eigenfaces"
        build_lfw_like_dataset(cls.data_dir)
        train_eigenfaces_model(n_components=N_COMPONENTS, data_dir=cls.data_dir, model_dir=cls.model_dir)

        cls.meta = prepare_lfw_dataset(cls.data_dir)
        cls.model = load_face_model(cls.model_dir)
        assert cls.model is not None, "Model should be trained before running tests."

        cls._patches = [
            patch("forensics_app.tools.face_recognition.DEFAULT_DATA_DIR", cls.data_dir),
            patch("forensics_app.tools.face_recognition.DEFAULT_MODEL_DIR", cls.model_dir),
        ]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls) -> None:
        for p in cls._patches:
            p.stop()
        cls._tmp.cleanup()

    def test_project_face_image(self) -> None:
        """Verify query image projection into k-dimensional subspace."""
        h, w = self.model.img_shape
        dummy_img = Image.new("L", (w, h), color=120)
        k = 15
        w_proj = project_face_image(dummy_img, self.model, k=k)
        self.assertEqual(w_proj.shape, (k,))
        self.assertEqual(w_proj.dtype, np.float64)

    def test_classify_face_majority_voting(self) -> None:
        """Verify m-NN majority voting strategy."""
        k = 20
        m = 3
        test_item = self.meta["test"][0]
        with Image.open(test_item["path"]) as img:
            w_test = project_face_image(img, self.model, k=k)

        res = classify_face(
            test_w=w_test,
            model=self.model,
            k=k,
            m=m,
            strategy="majority",
            use_lda=False,
        )
        self.assertIn("predicted_label", res)
        self.assertIn("predicted_name", res)
        self.assertEqual(len(res["top_matches"]), m)
        self.assertEqual(res["strategy_info"]["type"], "Majority Voting (m-NN)")

    def test_classify_face_distance_weighted(self) -> None:
        """Verify distance-weighted voting strategy."""
        k = 20
        m = 5
        test_item = self.meta["test"][0]
        with Image.open(test_item["path"]) as img:
            w_test = project_face_image(img, self.model, k=k)

        res = classify_face(
            test_w=w_test,
            model=self.model,
            k=k,
            m=m,
            strategy="distance_weighted",
            use_lda=False,
        )
        self.assertEqual(res["strategy_info"]["type"], "Distance-Weighted Voting")
        self.assertIn("weights", res["strategy_info"])

    def test_classify_face_centroids(self) -> None:
        """Verify class centroids distance strategy."""
        k = 20
        m = 3
        test_item = self.meta["test"][0]
        with Image.open(test_item["path"]) as img:
            w_test = project_face_image(img, self.model, k=k)

        res = classify_face(
            test_w=w_test,
            model=self.model,
            k=k,
            m=m,
            strategy="centroid",
            use_lda=False,
        )
        self.assertEqual(res["strategy_info"]["type"], "Class Centroids")
        self.assertIn("centroid_distances", res["strategy_info"])

    def test_classify_face_lda_fisherfaces(self) -> None:
        """Verify Fisherfaces (LDA) classification."""
        k = 25
        m = 3
        test_item = self.meta["test"][0]
        with Image.open(test_item["path"]) as img:
            w_test = project_face_image(img, self.model, k=k)

        res = classify_face(
            test_w=w_test,
            model=self.model,
            k=k,
            m=m,
            strategy="majority",
            use_lda=True,
        )
        self.assertTrue(res["use_lda"])
        self.assertIn("predicted_name", res)

    def test_make_recognition_gallery(self) -> None:
        """Verify creation of side-by-side comparison gallery."""
        test_img = Image.new("RGB", (37, 50), color=(100, 100, 100))
        top_matches = [
            {
                "rank": 1,
                "distance": 2.45,
                "label": 0,
                "name": "George W Bush",
                "image_path": self.meta["train"][0]["path"],
            },
            {
                "rank": 2,
                "distance": 3.12,
                "label": 0,
                "name": "George W Bush",
                "image_path": self.meta["train"][1]["path"],
            },
        ]
        gallery = make_recognition_gallery(
            test_image=test_img,
            top_matches=top_matches,
            predicted_name="George W Bush",
            strategy="Majority Voting",
            true_name="George W Bush",
        )
        self.assertIsInstance(gallery, Image.Image)
        self.assertEqual(gallery.mode, "RGB")
        self.assertGreater(gallery.width, 300)

    @patch("forensics_app.tools.face_recognition.FaceRecognitionDialog")
    def test_face_recognition_tool_run(self, mock_dialog_cls: MagicMock) -> None:
        """Verify end-to-end execution of FaceRecognitionTool."""
        mock_dlg = MagicMock()
        mock_dlg.user_confirmed = True
        mock_dlg.k = 20
        mock_dlg.m = 3
        mock_dlg.strategy = "majority"
        mock_dlg.method = "eigenfaces"
        mock_dlg.source = "test_set"
        mock_dlg.selected_test_file = self.meta["test"][0]["path"]
        mock_dialog_cls.return_value = mock_dlg

        doc = ImageDocument()
        tool = FaceRecognitionTool()
        result = tool.run(parent=MagicMock(), document=doc)

        self.assertIsNotNone(result)
        self.assertTrue(result.preview_only)
        self.assertIn("Recognized as", result.message)
        self.assertIn("Predicted Identity", result.details)
        self.assertIn("Top-1 Match", result.details)


if __name__ == "__main__":
    unittest.main()
