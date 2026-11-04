"""Unit tests for the Eigenfaces builder tool (Set 6.2)."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.eigenfaces_builder import (
    EigenfacesBuilderTool,
    EigenfacesDialog,
)
from forensics_app.tools.face_recognition_core import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    FaceModel,
    load_face_model,
    make_eigenfaces_grid,
    prepare_lfw_dataset,
    train_eigenfaces_model,
)


class TestEigenfacesBuilder(unittest.TestCase):
    """Test suite for Eigenface model training, serialization, and grid visualization."""

    def test_prepare_lfw_dataset(self) -> None:
        """Verify that LFW dataset files are extracted and indexed properly."""
        meta = prepare_lfw_dataset(DEFAULT_DATA_DIR)
        self.assertIn("train", meta)
        self.assertIn("test", meta)
        self.assertIn("target_names", meta)
        self.assertGreater(len(meta["train"]), 100)
        self.assertGreater(len(meta["test"]), 50)
        self.assertEqual(meta["n_classes"], len(meta["target_names"]))

        # Check that physical image files exist on disk
        first_train = Path(meta["train"][0]["path"])
        self.assertTrue(first_train.exists())

    def test_train_eigenfaces_model(self) -> None:
        """Verify PCA model training, mean face saving, and array shapes."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model = train_eigenfaces_model(n_components=20, data_dir=DEFAULT_DATA_DIR, model_dir=tmp_path)
            self.assertIsInstance(model, FaceModel)
            self.assertEqual(model.eigenfaces.shape[0], 20)
            self.assertEqual(model.X_train_pca.shape[1], 20)
            self.assertEqual(len(model.explained_variance_ratio), 20)

            # Check global and class mean faces on disk
            mean_dir = tmp_path / "mean_faces"
            self.assertTrue((mean_dir / "mean_face_global.png").exists())
            self.assertTrue(len(list(mean_dir.glob("mean_face_*.png"))) >= 2)

    def test_load_face_model(self) -> None:
        """Verify loading pre-trained FaceModel from disk."""
        model = load_face_model(DEFAULT_MODEL_DIR)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(len(model.eigenfaces), 20)
        self.assertEqual(model.mean_face.shape, model.img_shape)

    def test_make_eigenfaces_grid(self) -> None:
        """Verify rendering K eigenfaces into a grid image with square auto-layout."""
        dummy_eigenfaces = np.random.randn(16, 50, 37)
        # Test explicit columns
        grid_explicit = make_eigenfaces_grid(dummy_eigenfaces, k=12, n_cols=4)
        self.assertIsInstance(grid_explicit, Image.Image)
        self.assertEqual(grid_explicit.mode, "RGB")
        self.assertGreater(grid_explicit.width, 300)
        self.assertGreater(grid_explicit.height, 300)

        # Test square auto-layout (n_cols=None -> ceil(sqrt(16)) = 4x4)
        grid_square = make_eigenfaces_grid(dummy_eigenfaces, k=16, n_cols=None)
        self.assertIsInstance(grid_square, Image.Image)
        self.assertEqual(grid_square.mode, "RGB")

    @patch("forensics_app.tools.eigenfaces_builder.EigenfacesDialog")
    def test_eigenfaces_builder_tool_run(self, mock_dialog_cls: MagicMock) -> None:
        """Verify EigenfacesBuilderTool execution and ToolResult generation."""
        mock_dlg = MagicMock()
        mock_dlg.user_confirmed = True
        mock_dlg.k = 8
        mock_dlg.n_components = 20
        mock_dlg.retrain = False
        mock_dialog_cls.return_value = mock_dlg

        doc = ImageDocument()
        tool = EigenfacesBuilderTool()
        result = tool.run(parent=MagicMock(), document=doc)

        self.assertIsNotNone(result)
        self.assertTrue(result.preview_only)
        self.assertIsNotNone(result.image)
        self.assertIn("Visualized Eigenfaces (K)", result.details)
        self.assertIn("8", result.details["Visualized Eigenfaces (K)"])
        self.assertIn("Active Subspace Components", result.details)
        self.assertIn("Total Components Stored", result.details)


if __name__ == "__main__":
    unittest.main()
