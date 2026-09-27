import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.mask import MaskTool, apply_mask

# Constants for patching dependencies
ASK_OPEN_FILENAME = "forensics_app.tools.mask.filedialog.askopenfilename"
ASK_YES_NO = "forensics_app.tools.mask.askyesno"
SHOW_ERROR = "forensics_app.tools.mask.messagebox.showerror"
IMAGE_OPEN = "forensics_app.tools.mask.Image.open"


class ApplyMaskTests(unittest.TestCase):
    """Unit tests for the apply_mask function."""
    def setUp(self) -> None:
        # Create a 2x2 black base image
        self.base_image = Image.new("RGB", (2, 2), color=(0, 0, 0))
        
        # Use a single-channel-positive pixel (Red only) to detect channel-wise copying issues
        self.mask_img = Image.new("RGB", (2, 2), color=(0, 0, 0))
        self.mask_img.putpixel((0, 0), (255, 0, 0))

    def test_apply_mask_without_texture(self) -> None:
        # Should replace the masked pixel with the mask's own color
        result = apply_mask(self.base_image, self.mask_img, None)
        result_array = np.array(result)

        # Assert complete output pixel explicitly to ensure channel-wise consistency
        self.assertTrue(np.all(result_array[0, 0] == [255, 0, 0]))  # From mask
        self.assertTrue(np.all(result_array[1, 1] == [0, 0, 0]))    # Kept from base

    def test_apply_mask_with_texture(self) -> None:
        # Should replace the masked pixel with the texture's color
        texture_img = Image.new("RGB", (2, 2), color=(10, 200, 50))
        
        result = apply_mask(self.base_image, self.mask_img, texture_img)
        result_array = np.array(result)

        self.assertTrue(np.all(result_array[0, 0] == [10, 200, 50]))  # From texture
        self.assertTrue(np.all(result_array[1, 1] == [0, 0, 0]))      # Kept from base

    def test_resizes_mask_and_texture_automatically(self) -> None:
        # Provide images of different sizes than the base image
        large_mask = Image.new("RGB", (10, 10), color=(255, 0, 0))
        large_texture = Image.new("RGB", (5, 5), color=(100, 100, 100))

        result = apply_mask(self.base_image, large_mask, large_texture)
        
        # Output must be the size of the base image
        self.assertEqual(result.size, (2, 2))

    def test_apply_mask_unselected_rgba_base_rgb_mask(self) -> None:
        # Base with 4 channels (RGBA)
        base_rgba = Image.new("RGBA", (2, 2), color=(10, 20, 30, 255))
        # Mask with 3 channels (RGB) using a single-channel-positive pixel (Green)
        mask_rgb = Image.new("RGB", (2, 2), color=(0, 0, 0))
        mask_rgb.putpixel((0, 0), (0, 255, 0))
        
        result = apply_mask(base_rgba, mask_rgb, None)
        result_array = np.array(result)
        
        # Output must maintain the RGBA mode
        self.assertEqual(result.mode, "RGBA")
        # Active pixel should take the mask's color padded with full alpha
        self.assertTrue(np.all(result_array[0, 0] == [0, 255, 0, 255]))
        # Unselected pixel must retain the exact base RGBA color without degradation
        self.assertTrue(np.all(result_array[1, 1] == [10, 20, 30, 255]))

    def test_apply_mask_grayscale_base(self) -> None:
        # Base with 1 channel (L)
        base_l = Image.new("L", (2, 2), color=50)
        mask_rgb = Image.new("RGB", (2, 2), color=(0, 0, 0))
        mask_rgb.putpixel((0, 0), (200, 200, 200))
        
        result = apply_mask(base_l, mask_rgb, None)
        
        # Output must maintain the L mode
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.getpixel((0, 0)), 200)  # Modifed by mask
        self.assertEqual(result.getpixel((1, 1)), 50)   # Intact from base

    def test_apply_mask_palette_base(self) -> None:
        # Create a base P image with a custom palette
        base_p = Image.new("P", (2, 2), color=0)
        # Palette: Index 0 -> Black, Index 1 -> Blue, Index 2 -> Red
        palette = [0, 0, 0, 0, 0, 255, 255, 0, 0] + [0] * 759
        base_p.putpalette(palette)
        
        # Fill base image with Index 1 (Blue)
        base_p.paste(1, (0, 0, 2, 2))
        
        # Create an RGB mask with a Red active pixel
        mask_rgb = Image.new("RGB", (2, 2), color=(0, 0, 0))
        mask_rgb.putpixel((0, 0), (255, 0, 0))
        
        result = apply_mask(base_p, mask_rgb, None)
        
        # Output must maintain P mode and preserve the exact original palette
        self.assertEqual(result.mode, "P")
        self.assertEqual(result.getpalette(), palette)
        
        # Unselected pixel must remain the exact base color index (1 -> Blue)
        self.assertEqual(result.getpixel((1, 1)), 1)
        # The selected pixel should be replaced with the closest color in the palette (Red -> Index 2)
        self.assertEqual(result.getpixel((0, 0)), 2)


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
        mock_img_context = MagicMock()
        mock_img_context.copy.return_value = self.mock_mask
        mock_open.return_value.__enter__.return_value = mock_img_context

        self.assertIsNone(self.tool.run(None, self.document))

    @patch(ASK_YES_NO, return_value=False)
    @patch(ASK_OPEN_FILENAME, return_value="fake_mask.png")
    @patch(IMAGE_OPEN)
    def test_run_processes_successfully_without_texture(
        self, mock_open: MagicMock, mock_ask_file: MagicMock, mock_ask_yesno: MagicMock
    ) -> None:
        mock_img_context = MagicMock()
        mock_img_context.copy.return_value = self.mock_mask
        mock_open.return_value.__enter__.return_value = mock_img_context

        result = self.tool.run(None, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result.details["Mask used"], "fake_mask.png")
        self.assertEqual(result.details["Texture applied"], "No Texture")

    @patch(ASK_YES_NO, return_value=True)
    @patch(ASK_OPEN_FILENAME, side_effect=["fake_mask.png", "fake_texture.png"])
    @patch(IMAGE_OPEN)
    def test_run_processes_successfully_with_texture(
        self, mock_open: MagicMock, mock_ask_file: MagicMock, mock_ask_yesno: MagicMock
    ) -> None:
        mask_context = MagicMock()
        mask_context.copy.return_value = self.mock_mask
        
        texture_context = MagicMock()
        texture_context.copy.return_value.convert.return_value = self.mock_texture

        mock_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mask_context)),
            MagicMock(__enter__=MagicMock(return_value=texture_context)),
        ]

        result = self.tool.run(None, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result.details["Mask used"], "fake_mask.png")
        self.assertEqual(result.details["Texture applied"], "fake_texture.png")


class LoadImageTests(unittest.TestCase):
    """Unit tests for the load_image method of MaskTool."""
    def setUp(self) -> None:
        self.tool = MaskTool()

    @patch(ASK_OPEN_FILENAME, return_value="")
    def test_load_image_handles_cancellation(self, mock_ask: MagicMock) -> None:
        self.assertIsNone(self.tool.load_image(None, "Open"))

    @patch(SHOW_ERROR)
    @patch(IMAGE_OPEN, side_effect=UnidentifiedImageError("Invalid format"))
    @patch(ASK_OPEN_FILENAME, return_value="broken_image.txt")
    def test_load_image_handles_corrupt_files(
        self, mock_ask: MagicMock, mock_open: MagicMock, mock_show_error: MagicMock
    ) -> None:
        result = self.tool.load_image(None, "Open")
        self.assertIsNone(result)
        mock_show_error.assert_called_once()
        self.assertIn("Could not open image", mock_show_error.call_args[0][0])

    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="valid_image.png")
    def test_load_image_converts_mode_based_on_base_image(self, mock_ask: MagicMock, mock_open: MagicMock) -> None:
        mock_img = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_img

        base_img = Image.new("RGBA", (2, 2))
        self.tool.load_image(None, "Open", base_image=base_img)

        # Verify the base image mode triggers the correct explicit conversion
        mock_img.copy.return_value.convert.assert_called_once_with("RGBA")
        
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="valid_image.png")
    def test_load_image_quantizes_palette_based_on_base_image(self, mock_ask: MagicMock, mock_open: MagicMock) -> None:
        mock_img = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_img

        base_img = Image.new("P", (2, 2))
        self.tool.load_image(None, "Open", base_image=base_img)

        # Verify the palette mode triggers RGB conversion followed by quantization matching the base image
        mock_img.copy.return_value.convert.assert_called_once_with("RGB")
        mock_img.copy.return_value.convert.return_value.quantize.assert_called_once_with(palette=base_img)


if __name__ == "__main__":
    unittest.main()
