import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.mask import MaskTool

# Constants for patching dependencies
ASK_OPEN_FILENAME = "forensics_app.tools.mask.filedialog.askopenfilename"
ASK_YES_NO = "forensics_app.tools.mask.askyesno"
SHOW_ERROR = "forensics_app.tools.mask.messagebox.showerror"
IMAGE_OPEN = "forensics_app.tools.mask.Image.open"


class ApplyMaskTests(unittest.TestCase):
    """Unit tests for the apply_mask method of MaskTool."""
    def setUp(self) -> None:
        self.tool = MaskTool()
        # Create a 2x2 black base image
        self.base_image = Image.new("RGB", (2, 2), color=(0, 0, 0))
        
        # Create a mask where only the top-left pixel is active (white)
        self.mask_img = Image.new("RGB", (2, 2), color=(0, 0, 0))
        self.mask_img.putpixel((0, 0), (255, 255, 255))

    def test_apply_mask_without_texture(self) -> None:
        # Should replace the masked pixel with the mask's own color
        result = self.tool.apply_mask(self.base_image, self.mask_img, None)
        result_array = np.array(result)

        self.assertTrue(np.all(result_array[0, 0] == [255, 255, 255]))  # From mask
        self.assertTrue(np.all(result_array[1, 1] == [0, 0, 0]))        # Kept from base

    def test_apply_mask_with_texture(self) -> None:
        # Should replace the masked pixel with the texture's color
        texture_img = Image.new("RGB", (2, 2), color=(10, 200, 50))
        
        result = self.tool.apply_mask(self.base_image, self.mask_img, texture_img)
        result_array = np.array(result)

        self.assertTrue(np.all(result_array[0, 0] == [10, 200, 50]))  # From texture
        self.assertTrue(np.all(result_array[1, 1] == [0, 0, 0]))      # Kept from base

    def test_resizes_mask_and_texture_automatically(self) -> None:
        # Provide images of different sizes than the base image
        large_mask = Image.new("RGB", (10, 10), color=(255, 255, 255))
        large_texture = Image.new("RGB", (5, 5), color=(100, 100, 100))

        result = self.tool.apply_mask(self.base_image, large_mask, large_texture)
        
        # Output must be the size of the base image
        self.assertEqual(result.size, (2, 2))


class MaskToolLogicTests(unittest.TestCase):
    """Unit tests for the MaskTool's run method and its interactions with file dialogs and user prompts."""
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (4, 4), color="blue")
        self.tool = MaskTool()
        
        # Mock images to return during file opening
        self.mock_mask = Image.new("RGB", (4, 4), color="white")
        self.mock_texture = Image.new("RGB", (4, 4), color="red")

    @patch(ASK_OPEN_FILENAME, return_value="")
    def test_run_returns_none_when_mask_dialog_cancelled(self, mock_ask: MagicMock) -> None:
        self.assertIsNone(self.tool.run(None, self.document))

    @patch(ASK_YES_NO, return_value=True)
    @patch(ASK_OPEN_FILENAME, side_effect=["fake_mask.png", ""])
    @patch(IMAGE_OPEN)
    def test_run_returns_none_when_texture_dialog_cancelled(
        self, mock_open: MagicMock, mock_ask_file: MagicMock, mock_ask_yesno: MagicMock
    ) -> None:
        # Mock the context manager for Image.open to return the mask successfully
        mock_img_context = MagicMock()
        # Mock the copy and convert methods to return the mock mask (the return value inside the context manager)
        mock_img_context.copy.return_value.convert.return_value = self.mock_mask

        # We want to mock image from with Image.open(source) as image.
        # mock_open is the mock of the internal Image.open function, which is used as a context manager.
        # So we need to set the return value of the context manager's __enter__ method to our mock image context. 
        mock_open.return_value.__enter__.return_value = mock_img_context

        # The first call to askopenfilename will return "fake_mask.png" (for the mask), and the second call will return "" (for the texture), simulating a cancellation.
        self.assertIsNone(self.tool.run(None, self.document))

    @patch(ASK_YES_NO, return_value=False)
    @patch(ASK_OPEN_FILENAME, return_value="fake_mask.png")
    @patch(IMAGE_OPEN)
    def test_run_processes_successfully_without_texture(
        self, mock_open: MagicMock, mock_ask_file: MagicMock, mock_ask_yesno: MagicMock
    ) -> None:
        # Mock the context manager for Image.open to return the mask successfully
        mock_img_context = MagicMock()
        mock_img_context.copy.return_value.convert.return_value = self.mock_mask
        mock_open.return_value.__enter__.return_value = mock_img_context

        result = self.tool.run(None, self.document)
        # Verify that the result is not None and contains the expected details
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Mask used"], "fake_mask.png")
        self.assertEqual(result.details["Texture applied"], "No Texture")

    @patch(ASK_YES_NO, return_value=True)
    @patch(ASK_OPEN_FILENAME, side_effect=["fake_mask.png", "fake_texture.png"])
    @patch(IMAGE_OPEN)
    def test_run_processes_successfully_with_texture(
        self, mock_open: MagicMock, mock_ask_file: MagicMock, mock_ask_yesno: MagicMock
    ) -> None:
        # Set up sequential returns for Image.open() to yield mask, then texture
        mask_context = MagicMock()
        mask_context.copy.return_value.convert.return_value = self.mock_mask
        
        texture_context = MagicMock()
        texture_context.copy.return_value.convert.return_value = self.mock_texture

        # The mock_open needs to return different context managers for the two calls to Image.open
        mock_open.side_effect = [
            # First call returns the mask context manager, second call returns the texture context manager
            MagicMock(__enter__=MagicMock(return_value=mask_context)),
            MagicMock(__enter__=MagicMock(return_value=texture_context)),
        ]

        result = self.tool.run(None, self.document)
        # Verify that the result is not None and contains the expected details
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Mask used"], "fake_mask.png")
        self.assertEqual(result.details["Texture applied"], "fake_texture.png")

    @patch(ASK_YES_NO, return_value=True)
    @patch(ASK_OPEN_FILENAME, side_effect=["mask.png", "texture.jpg"])
    @patch(IMAGE_OPEN)
    def test_run_converts_mixed_modes_safely(
        self, mock_open: MagicMock, mock_ask_file: MagicMock, mock_ask_yesno: MagicMock
    ) -> None:
        # Overwrite the base image for this specific test to use RGBA (4 channels)
        self.document.current = Image.new("RGBA", (4, 4), color=(10, 20, 30, 255))
        
        # 1. Simulate the user loading a pure black and white mask ("1" mode)
        mask_img = Image.new("1", (4, 4), color=1)
        
        # 2. Simulate the user loading a print-format texture ("CMYK" mode)
        texture_img = Image.new("CMYK", (4, 4), color=(10, 20, 30, 40))

        # Configure the Image.open mock to return these real images
        # when entering the 'with' context blocks
        mock_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mask_img)),
            MagicMock(__enter__=MagicMock(return_value=texture_img)),
        ]

        # 3. Execute the full flow
        result = self.tool.run(None, self.document)

        # 4. Assertions
        self.assertIsNotNone(result)
        # Verify that the result did not crash in apply_mask and kept the base mode
        self.assertEqual(result.image.mode, "RGBA")
        self.assertEqual(result.details["Output mode"], "RGBA")


class LoadImageTests(unittest.TestCase):
    """Unit tests for the load_image method of MaskTool."""
    def setUp(self) -> None:
        self.tool = MaskTool()

    @patch(ASK_OPEN_FILENAME, return_value="")
    def test_load_image_handles_cancellation(self, mock_ask: MagicMock) -> None:
        # Test that load_image returns None when the user cancels the file dialog (similar to the run method's behavior)
        self.assertIsNone(self.tool.load_image(None, "Open", "RGB"))

    @patch(SHOW_ERROR)
    @patch(IMAGE_OPEN, side_effect=UnidentifiedImageError("Invalid format"))
    @patch(ASK_OPEN_FILENAME, return_value="broken_image.txt")
    def test_load_image_handles_corrupt_files(
        self, mock_ask: MagicMock, mock_open: MagicMock, mock_show_error: MagicMock
    ) -> None:
        result = self.tool.load_image(None, "Open", "RGB")
        # Verify that the result is None and that an error message was shown
        self.assertIsNone(result)
        mock_show_error.assert_called_once()
        self.assertIn("Could not open image", mock_show_error.call_args[0][0])

    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="valid_image.png")
    def test_load_image_converts_mode(self, mock_ask: MagicMock, mock_open: MagicMock) -> None:
        mock_img = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_img

        self.tool.load_image(None, "Open", "RGBA")

        # Verify the requested mode is explicitly passed to convert()
        mock_img.copy.return_value.convert.assert_called_once_with("RGBA")


if __name__ == "__main__":
    unittest.main()