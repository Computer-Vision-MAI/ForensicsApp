import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import CannyEdgeTool
from forensics_app.tools.canny_edge import apply_canny_edge


class CannyEdgeToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        # Create a simple test image: a 10x10 grayscale image with a white square in the center.
        self.document.current = Image.new("RGB", (10, 10), color=(0, 0, 0))
        for i in range(3, 7):
            for j in range(3, 7):
                self.document.current.putpixel((i, j), (255, 255, 255))
        
        self.tool = CannyEdgeTool()
        self.parent = None  # Mock parent for Tkinter
    
    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.simpledialog.askfloat")
    def test_returns_none_on_cancellation(self, mock_askfloat, mock_askyesno) -> None:
        values = [1.0, 10.0, 50., 0.5]
        # Simulate that the user chooses not to superimpose edges (False)
        mock_askyesno.return_value = True
        for i in range(4):
            with self.subTest(msg=f"Cancellation test {i+1}"):
                mock_askfloat.side_effect = [values[j] if j != i else None for j in range(4)]
                result = self.tool.run(self.parent, self.document)
                self.assertIsNone(result)     

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.simpledialog.askfloat")
    def test_canny_edge_binary_mask(self, mock_askfloat, mock_askyesno) -> None:
        source = self.document.current
        mock_askyesno.return_value = False
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = source.convert(mode)
                mock_askfloat.side_effect = [1.0, 10.0, 50.0]
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, source.size)
                self.assertEqual(set(np.unique(result.image)), {0, 255})

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.simpledialog.askfloat")
    def test_canny_edge_superimposed(self, mock_askfloat, mock_askyesno) -> None:
        source = self.document.current
        mock_askyesno.return_value = True
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = source.convert(mode)
                mock_askfloat.side_effect = [1.0, 10.0, 50.0, 0.5]
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.image.size, source.size)

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.simpledialog.askfloat")
    def test_binary_mask_edge_locations(self, mock_askfloat, mock_askyesno) -> None:
        # sigma, low_thresh, high_thresh
        mock_askfloat.side_effect = [1.0, 10.0, 50.0]
        mock_askyesno.return_value = False
        
        result = self.tool.run(self.parent, self.document)
        arr = np.array(result.image)
        
        # 1. Center of the white square (5,5) should NOT be an edge, hence 0 in the binary mask.
        self.assertEqual(arr[5, 5], 0)
        
        # 2. The corners of the white square (3,3), (3,6), (6,3), (6,6) should be edges, hence 255 in the binary mask.
        self.assertEqual(arr[3, 3], 255)
        self.assertEqual(arr[0, 0], 0)
        

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.simpledialog.askfloat")
    def test_superimposed_ponderated_colors(self, mock_askfloat, mock_askyesno) -> None:
        # Use source colors with different channel values to catch desaturation.
        source = Image.new("RGB", (10, 10), (10, 30, 50))
        source.paste((200, 180, 160), (3, 3, 7, 7))
        self.document.current = source
        mock_askfloat.side_effect = [0.0, 10.0, 50.0, 0.5]
        mock_askyesno.return_value = True

        result = self.tool.run(self.parent, self.document)
        arr = np.array(result.image)
        self.assertEqual(result.image.mode, "RGB")
        np.testing.assert_array_equal(arr[5, 5], [200, 180, 160])
        np.testing.assert_array_equal(arr[0, 0], [10, 30, 50])
        np.testing.assert_array_equal(arr[3, 3], [227, 90, 80])

    @patch("tkinter.messagebox.askyesno", return_value=True)
    @patch("tkinter.simpledialog.askfloat")
    def test_superimposed_alpha_extremes(self, mock_askfloat, mock_askyesno) -> None:
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                mock_askfloat.side_effect = [1.0, 10.0, 50.0, alpha]
                result = self.tool.run(self.parent, self.document)
                arr = np.array(result.image)
                np.testing.assert_array_equal(arr[5, 5], [255, 255, 255])
                if alpha == 0.0:
                    np.testing.assert_array_equal(arr, np.array(self.document.current))
                else:
                    np.testing.assert_array_equal(arr[3, 3], [255, 0, 0])

    @patch("tkinter.messagebox.showerror")
    @patch("tkinter.messagebox.askyesno", return_value=False)
    @patch("tkinter.simpledialog.askfloat", side_effect=[1.0, 50.0, 10.0])
    def test_processing_error_is_reported(self, mock_askfloat, mock_askyesno, mock_error) -> None:
        self.assertIsNone(self.tool.run(self.parent, self.document))
        mock_error.assert_called_once()


class CannyEdgeFunctionTests(unittest.TestCase):
    def test_mask_and_overlay_without_dialogs(self) -> None:
        source = Image.new("RGB", (10, 10), (10, 30, 50))
        source.paste((200, 180, 160), (3, 3, 7, 7))
        original = np.array(source)
        mask = apply_canny_edge(source, 0.0, 10.0, 50.0)
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, source.size)
        self.assertEqual(set(np.unique(mask)), {0, 255})
        edges = np.array(mask) != 0
        self.assertTrue(edges[3, 3])
        self.assertFalse(edges[5, 5])

        overlay = apply_canny_edge(source, 0.0, 10.0, 50.0, True, 0.5)
        self.assertEqual(overlay.mode, "RGB")
        self.assertEqual(overlay.size, source.size)
        actual = np.array(overlay)
        np.testing.assert_array_equal(actual[~edges], original[~edges])
        expected_edges = (original[edges] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        np.testing.assert_array_equal(actual[edges], expected_edges)
        np.testing.assert_array_equal(np.array(source), original)


if __name__ == "__main__":
    unittest.main()