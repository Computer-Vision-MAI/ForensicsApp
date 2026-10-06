import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import CannyEdgeTool


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
        # Simulate user input for sigma, low_thresh, high_thresh
        mock_askfloat.side_effect = [1.0, 10.0, 50.0]
        # Simulate that the user chooses not to superimpose edges (False)
        mock_askyesno.return_value = False
        
        result = self.tool.run(self.parent, self.document)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.image.mode, "L")
        
        # Check that the output is a binary mask (only 0 and 255 values)
        arr = np.array(result.image)
        unique_vals = np.unique(arr)
        for val in unique_vals:
            self.assertIn(val, [0, 255])

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.simpledialog.askfloat")
    def test_canny_edge_superimposed(self, mock_askfloat, mock_askyesno) -> None:
        # sigma, low_thresh, high_thresh, alpha
        mock_askfloat.side_effect = [1.0, 10.0, 50.0, 0.5]
        # Simulate that the user chooses to superimpose edges (True)
        mock_askyesno.return_value = True
        
        result = self.tool.run(self.parent, self.document)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, (10, 10))

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
        # sigma, low_thresh, high_thresh, alpha
        mock_askfloat.side_effect = [0.0, 10.0, 50.0, 0.5]
        mock_askyesno.return_value = True
        
        result = self.tool.run(self.parent, self.document)
        arr = np.array(result.image)
        
        # 1. Verify that the image is in RGB mode
        self.assertEqual(result.image.mode, "RGB")
        
        # 2. Check a pixel in the center of the white square (5,5) which is not an edge.
        # The original pixel value is (255, 255, 255), and since it's
        # not an edge, it should be weighted with alpha=0.5 getting 127.5 value.
        
        center_pixel = arr[5, 5]
        self.assertTrue(126 <= center_pixel[0] <= 128)
        self.assertEqual(center_pixel[0], center_pixel[1])
        self.assertEqual(center_pixel[1], center_pixel[2])
        
        # 3. Check a pixel in the background that is not an edge (original=0).
        # Formula: 0 * (1 - 0.5) = 0.
        corner_pixel = arr[0, 0]
        print(f"Corner pixel value: {corner_pixel}")
        self.assertTrue(np.array_equal(corner_pixel, [0, 0, 0]))
        
        # 4. Check a pixel that is an edge (e.g., (3,3)). The original pixel value is (255, 255, 255), 
        # and the edge color is (255, 0, 0).
        edge_found = False
        for r in range(10):
            for c in range(10):
                pixel = arr[r, c]
                # Check if the pixel has a red channel greater than the green and blue channels, indicating an edge.
                if pixel[0] > pixel[1] and pixel[0] > pixel[2]:
                    edge_found = True
                    # Check that the red channel is greater than the green and blue channels, indicating an edge.
                    self.assertTrue(pixel[0] > pixel[1])
                    self.assertTrue(pixel[0] > pixel[2])
                    # Check that the green and blue channels are equal (since the edge color is red).
                    self.assertEqual(pixel[1], pixel[2])
                    break
            if edge_found:
                break
                
        self.assertTrue(edge_found, "Not found edge pixels.")

    @patch("tkinter.messagebox.askyesno")
    @patch("tkinter.simpledialog.askfloat")
    def test_superimposed_alpha_extremes(self, mock_askfloat, mock_askyesno) -> None:
        

        # We check the behavior of the ponderated sum with extreme alpha values like 1.0.
        # 
        mock_askfloat.side_effect = [1.0, 10.0, 50.0, 1.0]
        mock_askyesno.return_value = True
        
        result = self.tool.run(self.parent, self.document)
        arr = np.array(result.image)
        
        # Center pixel (5,5) should be 0 as it's not an edge.
        self.assertTrue(np.array_equal(arr[5, 5], [0, 0, 0]))
        
        # Edge pixels should be fully red due to alpha=1.0, meaning the original color is completely overridden.
        edge_pixels = arr[(arr[:, :, 0] == 255)]
        self.assertTrue(len(edge_pixels) > 0)
        for p in edge_pixels:
            self.assertTrue(np.array_equal(p, [255, 0, 0]))


if __name__ == "__main__":
    unittest.main()