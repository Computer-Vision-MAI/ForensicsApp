from __future__ import annotations
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools import TemplateMatchingTool
from forensics_app.tools.template_matching import (
    apply_template_matching,
    prepare_matched_arrays,
    render_template_result,
    resize_template,
    rotate_template_array,
    to_unit_color_array,
    to_unit_gray_array,
)

ASK_OPEN_FILENAME = "forensics_app.tools.dialogs.filedialog.askopenfilename"
ASK_YES_NO = "forensics_app.tools.template_matching.messagebox.askyesno"
SHOW_ERROR = "forensics_app.tools.template_matching.messagebox.showerror"
DIALOG_SHOW_ERROR = "forensics_app.tools.dialogs.messagebox.showerror"
IMAGE_OPEN = "forensics_app.tools.dialogs.Image.open"
ASK_FLOAT = "forensics_app.tools.template_matching.simpledialog.askfloat"
ASK_CHOICE = "forensics_app.tools.template_matching.ask_choice"


class TemplateMatchingToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        # Create a 20x20 test image with a structured 6x6 patch in the center
        self.document.current = Image.new("RGB", (20, 20), color=(10, 20, 30))
        for y in range(7, 13):
            for x in range(7, 13):
                v = (x - 7) * 30 + (y - 7) * 10
                self.document.current.putpixel((x, y), (v + 50, v + 20, v + 10))

        # Distinctive template crop
        self.mock_template = self.document.current.crop((7, 7, 13, 13))
        self.tool = TemplateMatchingTool()
        self.parent = None

    @patch(ASK_OPEN_FILENAME, return_value="")
    def test_returns_none_on_image_selection_cancellation(
        self, mock_askopen: MagicMock
    ) -> None:
        result = self.tool.run(self.parent, self.document)
        self.assertIsNone(result)

    @patch(DIALOG_SHOW_ERROR)
    @patch(IMAGE_OPEN, side_effect=UnidentifiedImageError("Cannot read file"))
    @patch(ASK_OPEN_FILENAME, return_value="broken.jpg")
    def test_returns_none_on_image_open_error(
        self, mock_askopen: MagicMock, mock_open: MagicMock, mock_error: MagicMock
    ) -> None:
        result = self.tool.run(self.parent, self.document)
        self.assertIsNone(result)
        mock_error.assert_called_once()

    @patch(ASK_CHOICE, return_value=None)
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="template.png")
    def test_returns_none_on_choice_cancellation(
        self, mock_askopen: MagicMock, mock_open: MagicMock, mock_choice: MagicMock
    ) -> None:
        mock_context = MagicMock()
        mock_context.copy.return_value = self.mock_template
        mock_open.return_value.__enter__.return_value = mock_context

        result = self.tool.run(self.parent, self.document)
        self.assertIsNone(result)

    @patch(ASK_CHOICE, return_value=1)  # Rescale (%)
    @patch(ASK_YES_NO, return_value=True)
    @patch(ASK_FLOAT)
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="template.png")
    def test_returns_none_on_dialog_cancellations(
        self,
        mock_askopen: MagicMock,
        mock_open: MagicMock,
        mock_askfloat: MagicMock,
        mock_askyesno: MagicMock,
        mock_choice: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.copy.return_value = self.mock_template
        mock_open.return_value.__enter__.return_value = mock_context

        # 4 dialogs in rescale mode: scale_pct, rotation, threshold, alpha
        valid_inputs = [30.0, 0.0, 0.8, 0.6]
        for step in range(4):
            with self.subTest(cancelled_step=step):
                mock_askfloat.side_effect = [
                    valid_inputs[i] if i != step else None for i in range(4)
                ]
                result = self.tool.run(self.parent, self.document)
                self.assertIsNone(result)

    @patch(ASK_CHOICE, return_value=2)  # Resize (H & W)
    @patch(ASK_FLOAT)
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="template.png")
    def test_returns_none_on_resize_hw_cancellations(
        self,
        mock_askopen: MagicMock,
        mock_open: MagicMock,
        mock_askfloat: MagicMock,
        mock_choice: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.copy.return_value = self.mock_template
        mock_open.return_value.__enter__.return_value = mock_context

        # Cancel on Height
        mock_askfloat.side_effect = [None]
        self.assertIsNone(self.tool.run(self.parent, self.document))

        # Cancel on Width
        mock_askfloat.side_effect = [6.0, None]
        self.assertIsNone(self.tool.run(self.parent, self.document))

    @patch(ASK_CHOICE, return_value=0)  # No resizing
    @patch(ASK_YES_NO, return_value=False)
    @patch(ASK_FLOAT, side_effect=[0.0, 0.8])  # rotation, threshold
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="template.png")
    def test_run_with_no_resizing(
        self,
        mock_askopen: MagicMock,
        mock_open: MagicMock,
        mock_askfloat: MagicMock,
        mock_askyesno: MagicMock,
        mock_choice: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.copy.return_value = self.mock_template
        mock_open.return_value.__enter__.return_value = mock_context

        result = self.tool.run(self.parent, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Sizing"], "No resizing (keep template as-is)")
        self.assertEqual(result.details["Size Parameter"], "Original")

    @patch(ASK_CHOICE, return_value=2)  # Resize (H & W)
    @patch(ASK_YES_NO, return_value=False)
    @patch(ASK_FLOAT, side_effect=[6.0, 6.0, 0.0, 0.8])  # height, width, rotation, threshold
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="template.png")
    def test_run_with_fixed_resize(
        self,
        mock_askopen: MagicMock,
        mock_open: MagicMock,
        mock_askfloat: MagicMock,
        mock_askyesno: MagicMock,
        mock_choice: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.copy.return_value = self.mock_template
        mock_open.return_value.__enter__.return_value = mock_context

        result = self.tool.run(self.parent, self.document)
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Sizing"], "Resize (fixed height and width in pixels)")
        self.assertEqual(result.details["Size Parameter"], "6x6")

    @patch(ASK_CHOICE, return_value=1)
    @patch(ASK_YES_NO, return_value=False)
    @patch(ASK_FLOAT, side_effect=[30.0, 0.0, 0.8])
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="template.png")
    def test_template_matching_binary_mask(
        self,
        mock_askopen: MagicMock,
        mock_open: MagicMock,
        mock_askfloat: MagicMock,
        mock_askyesno: MagicMock,
        mock_choice: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.copy.return_value = self.mock_template
        mock_open.return_value.__enter__.return_value = mock_context

        source = self.document.current
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = source.convert(mode)
                mock_askfloat.side_effect = [30.0, 0.0, 0.8]
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, source.size)
                self.assertTrue(set(np.unique(result.image)).issubset({0, 255}))
                self.assertIn("mask generated successfully", result.message)
                self.assertEqual(result.details["Operation"], "Template Matching")
                self.assertEqual(result.details["Template"], "template.png")
                self.assertEqual(result.details["Sizing"], "Rescale (% of base image)")
                self.assertEqual(result.details["Size Parameter"], "30%")
                self.assertEqual(result.details["Rotation (deg)"], 0.0)
                self.assertEqual(result.details["Threshold"], 0.8)
                self.assertFalse(result.details["Superimposed"])
                self.assertEqual(result.details["Alpha"], "N/A")

    @patch(ASK_CHOICE, return_value=1)
    @patch(ASK_YES_NO, return_value=True)
    @patch(ASK_FLOAT, side_effect=[30.0, 0.0, 0.8, 0.5])
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="template.png")
    def test_template_matching_superimposed(
        self,
        mock_askopen: MagicMock,
        mock_open: MagicMock,
        mock_askfloat: MagicMock,
        mock_askyesno: MagicMock,
        mock_choice: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.copy.return_value = self.mock_template
        mock_open.return_value.__enter__.return_value = mock_context

        source = self.document.current
        for mode in ("RGB", "L", "RGBA", "P"):
            with self.subTest(mode=mode):
                self.document.current = source.convert(mode)
                mock_askfloat.side_effect = [30.0, 0.0, 0.8, 0.5]
                result = self.tool.run(self.parent, self.document)

                self.assertIsNotNone(result)
                expected_mode = "RGBA" if mode == "RGBA" else "RGB"
                self.assertEqual(result.image.mode, expected_mode)
                self.assertEqual(result.image.size, source.size)
                self.assertIn("superimposed successfully", result.message)
                self.assertTrue(result.details["Superimposed"])
                self.assertEqual(result.details["Alpha"], 0.5)

    @patch(SHOW_ERROR)
    @patch(ASK_CHOICE, return_value=1)
    @patch(ASK_YES_NO, return_value=False)
    @patch(ASK_FLOAT, side_effect=[-5.0, 0.0, 0.8])
    @patch(IMAGE_OPEN)
    @patch(ASK_OPEN_FILENAME, return_value="template.png")
    def test_processing_error_is_reported(
        self,
        mock_askopen: MagicMock,
        mock_open: MagicMock,
        mock_askfloat: MagicMock,
        mock_askyesno: MagicMock,
        mock_choice: MagicMock,
        mock_error: MagicMock,
    ) -> None:
        mock_context = MagicMock()
        mock_context.copy.return_value = self.mock_template
        mock_open.return_value.__enter__.return_value = mock_context

        result = self.tool.run(self.parent, self.document)
        self.assertIsNone(result)
        mock_error.assert_called_once()
        self.assertIn("Processing Error", mock_error.call_args[0][0])


class TemplateMatchingFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Base image: 30x30 with a textured 6x6 patch at (12..18, 12..18)
        self.base_rgb = Image.new("RGB", (30, 30), color=(10, 20, 30))
        for y in range(12, 18):
            for x in range(12, 18):
                val = (x - 12) * 30 + (y - 12) * 10
                self.base_rgb.putpixel((x, y), (val + 50, val + 20, val + 10))

        self.patch_rgb = self.base_rgb.crop((12, 12, 18, 18))

    def test_exact_match_detected_in_binary_mask(self) -> None:
        mask = apply_template_matching(
            image=self.base_rgb,
            template=self.patch_rgb,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=False,
        )
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, self.base_rgb.size)

        arr = np.array(mask)
        # The center of the 6x6 patch placed at [12..18, 12..18] is (15, 15)
        self.assertEqual(arr[15, 15], 255)
        # Background pixel should be 0
        self.assertEqual(arr[0, 0], 0)

    def test_match_with_rotation(self) -> None:
        # Create a 40x40 image with an L-shaped pattern rotated by 90 degrees
        base = np.zeros((40, 40), dtype=np.float32)
        patch = np.zeros((10, 10), dtype=np.float32)
        patch[2:8, 2:4] = 1.0
        patch[6:8, 2:8] = 1.0

        # Rotate patch by 90 degrees anticlockwise
        patch_rot90 = rotate_template_array(patch, 90.0)
        base[15:25, 15:25] = patch_rot90

        img_pil = Image.fromarray((base * 255).astype(np.uint8), mode="L")
        tmpl_pil = Image.fromarray((patch * 255).astype(np.uint8), mode="L")

        # Matching with 90 degrees should detect the match
        mask_90 = apply_template_matching(
            image=img_pil,
            template=tmpl_pil,
            size=None,
            rotation=90.0,
            threshold=0.9,
            superimpose=False,
        )
        arr_90 = np.array(mask_90)
        self.assertTrue(np.any(arr_90 == 255))

        # Matching with 0 degrees should not detect the match with high threshold
        mask_0 = apply_template_matching(
            image=img_pil,
            template=tmpl_pil,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=False,
        )
        arr_0 = np.array(mask_0)
        self.assertFalse(np.any(arr_0 == 255))

    def test_rectangular_template_match_with_rotation_90(self) -> None:
        # Create a 40x40 base image and an asymmetric rectangular patch (height=12, width=6)
        base = np.zeros((40, 40), dtype=np.float32)
        patch = np.zeros((12, 6), dtype=np.float32)
        for y in range(12):
            for x in range(6):
                patch[y, x] = 0.2 + 0.05 * y + 0.08 * x

        # Rotate patch by 90 degrees anticlockwise -> shape becomes (6, 12)
        patch_rot90 = rotate_template_array(patch, 90.0)
        self.assertEqual(patch_rot90.shape, (6, 12))

        # Place rotated patch into base image at [15..21, 15..27]
        base[15:21, 15:27] = patch_rot90

        img_pil = Image.fromarray((base * 255).astype(np.uint8), mode="L")
        tmpl_pil = Image.fromarray((patch * 255).astype(np.uint8), mode="L")

        # Matching with 90 degrees rotation should match
        mask_90 = apply_template_matching(
            image=img_pil,
            template=tmpl_pil,
            size=None,
            rotation=90.0,
            threshold=0.9,
            superimpose=False,
        )
        arr_90 = np.array(mask_90)
        self.assertTrue(np.any(arr_90 == 255))

    def test_rotated_template_dimensions_cannot_exceed_image(self) -> None:
        # Image is 10 wide by 20 high; template is 8 wide by 15 high
        img = Image.new("L", (10, 20))
        tmpl = Image.new("L", (8, 15))

        # Before rotation (8x15) fits in (10x20).
        # When rotated 90 degrees, dimensions become 15 wide by 8 high.
        # Rotated width (15) exceeds image width (10).
        with self.assertRaises(ValueError) as ctx:
            apply_template_matching(img, tmpl, rotation=90.0)
        self.assertIn("cannot exceed", str(ctx.exception))

    def test_superimposed_overlay_ponderated_colors(self) -> None:
        overlay = apply_template_matching(
            image=self.base_rgb,
            template=self.patch_rgb,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=True,
            alpha=0.5,
        )
        self.assertEqual(overlay.mode, "RGB")
        self.assertEqual(overlay.size, self.base_rgb.size)

        arr = np.array(overlay)
        orig = np.array(self.base_rgb)

        # Background pixel untouched
        np.testing.assert_array_equal(arr[0, 0], orig[0, 0])

        # Matched center pixel blended with red [255, 0, 0] at alpha 0.5
        expected_center = (orig[15, 15] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        np.testing.assert_array_equal(arr[15, 15], expected_center)

    def test_superimposed_alpha_extremes(self) -> None:
        orig = np.array(self.base_rgb)
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha):
                overlay = apply_template_matching(
                    image=self.base_rgb,
                    template=self.patch_rgb,
                    size=None,
                    rotation=0.0,
                    threshold=0.9,
                    superimpose=True,
                    alpha=alpha,
                )
                arr = np.array(overlay)
                if alpha == 0.0:
                    np.testing.assert_array_equal(arr, orig)
                else:
                    np.testing.assert_array_equal(arr[15, 15], [255, 0, 0])

    def test_resize_template_percentage(self) -> None:
        tmpl = Image.new("RGB", (50, 25))  # 2:1 aspect ratio
        base_size = (100, 100)
        # 20% of base size: scale = 0.2 * min(100/50, 100/25) = 0.2 * 2 = 0.4
        # 50 * 0.4 = 20, 25 * 0.4 = 10
        resized = resize_template(tmpl, base_size, 20.0)
        self.assertEqual(resized.size, (20, 10))

    def test_resize_template_fixed_tuple(self) -> None:
        tmpl = Image.new("RGB", (50, 25))
        resized = resize_template(tmpl, (100, 100), (12, 14))
        self.assertEqual(resized.size, (12, 14))

    def test_resize_template_none_keeps_size(self) -> None:
        tmpl = Image.new("RGB", (50, 25))
        resized = resize_template(tmpl, (100, 100), None)
        self.assertEqual(resized.size, (50, 25))

    def test_resize_template_invalid_percentage_raises(self) -> None:
        tmpl = Image.new("RGB", (10, 10))
        for invalid_pct in (-10.0, 0.0, 105.0):
            with self.subTest(pct=invalid_pct):
                with self.assertRaises(ValueError):
                    resize_template(tmpl, (100, 100), invalid_pct)

    def test_resize_template_invalid_dimension_raises(self) -> None:
        tmpl = Image.new("RGB", (10, 10))
        for invalid_size in ((-5, 10), (0, 0), "large"):
            with self.subTest(size=invalid_size):
                with self.assertRaises(ValueError):
                    resize_template(tmpl, (100, 100), invalid_size)  # type: ignore

    def test_template_exceeding_base_image_raises_error(self) -> None:
        small_base = Image.new("RGB", (10, 10))
        large_tmpl = Image.new("RGB", (20, 20))
        with self.assertRaises(ValueError) as ctx:
            apply_template_matching(small_base, large_tmpl, size=None)
        self.assertIn("cannot exceed", str(ctx.exception))

    def test_zero_dimension_image_raises_error(self) -> None:
        zero_img = Image.new("RGB", (0, 0))
        tmpl = Image.new("RGB", (5, 5))
        with self.assertRaises(ValueError):
            apply_template_matching(zero_img, tmpl)

    def test_all_image_modes_matching_logic(self) -> None:
        # 1. Color original + Gray template -> Grayscale matching
        gray_patch = self.patch_rgb.convert("L")
        res1 = apply_template_matching(
            self.base_rgb, gray_patch, size=None, threshold=0.9, superimpose=False
        )
        self.assertEqual(res1.mode, "L")
        self.assertEqual(np.array(res1)[15, 15], 255)

        # 2. Gray original + Color template -> Grayscale matching
        base_gray = self.base_rgb.convert("L")
        res2 = apply_template_matching(
            base_gray, self.patch_rgb, size=None, threshold=0.9, superimpose=False
        )
        self.assertEqual(res2.mode, "L")
        self.assertEqual(np.array(res2)[15, 15], 255)

        # 3. Color original + Color template -> Multichannel matching
        res3 = apply_template_matching(
            self.base_rgb, self.patch_rgb, size=None, threshold=0.9, superimpose=False
        )
        self.assertEqual(res3.mode, "L")
        self.assertEqual(np.array(res3)[15, 15], 255)

        # 4. Gray original + Gray template -> Grayscale matching
        res4 = apply_template_matching(
            base_gray, gray_patch, size=None, threshold=0.9, superimpose=False
        )
        self.assertEqual(res4.mode, "L")
        self.assertEqual(np.array(res4)[15, 15], 255)

        # 5. RGBA and Palette modes
        rgba_base = self.base_rgb.convert("RGBA")
        p_tmpl = self.patch_rgb.convert("P")
        res5 = apply_template_matching(
            rgba_base, p_tmpl, size=None, threshold=0.9, superimpose=False
        )
        self.assertEqual(res5.mode, "L")
        self.assertEqual(np.array(res5)[15, 15], 255)

    def test_high_depth_image_matching_without_clipping(self) -> None:
        # Create an 'I' mode high-depth image with values well above 255 (e.g. 1000..5000)
        high_depth_arr = np.full((30, 30), 1000, dtype=np.int32)
        # Distinct patch with internal variation
        for y in range(12, 18):
            for x in range(12, 18):
                high_depth_arr[y, x] = 2000 + (x - 12) * 500 + (y - 12) * 200
        img_i = Image.fromarray(high_depth_arr, mode="I")

        # Template is high-depth crop
        tmpl_i = img_i.crop((12, 12, 18, 18))

        # Binary mask
        mask = apply_template_matching(
            image=img_i,
            template=tmpl_i,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=False,
        )
        self.assertEqual(mask.mode, "L")
        self.assertEqual(np.array(mask)[15, 15], 255)

        # Superimposed overlay on high-depth image
        overlay = apply_template_matching(
            image=img_i,
            template=tmpl_i,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=True,
            alpha=0.6,
        )
        self.assertEqual(overlay.mode, "RGB")
        self.assertEqual(overlay.size, img_i.size)

    def test_float_mode_image_matching(self) -> None:
        # Create an 'F' mode 32-bit float image
        float_arr = np.full((30, 30), 0.1, dtype=np.float32)
        for y in range(12, 18):
            for x in range(12, 18):
                float_arr[y, x] = 0.3 + (x - 12) * 0.1 + (y - 12) * 0.05
        img_f = Image.fromarray(float_arr, mode="F")
        tmpl_f = img_f.crop((12, 12, 18, 18))

        mask = apply_template_matching(
            image=img_f,
            template=tmpl_f,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=False,
        )
        self.assertEqual(mask.mode, "L")
        self.assertEqual(np.array(mask)[15, 15], 255)

    def test_la_and_premultiplied_la_mode_matching(self) -> None:
        # Create base image in LA mode with a distinct patch at [12..18, 12..18]
        base_la = Image.new("LA", (30, 30), color=(30, 255))
        for y in range(12, 18):
            for x in range(12, 18):
                val = (x - 12) * 20 + (y - 12) * 15 + 60
                base_la.putpixel((x, y), (val, 255))

        patch_la = base_la.crop((12, 12, 18, 18))

        # Test LA mode mask and overlay
        mask_la = apply_template_matching(
            image=base_la,
            template=patch_la,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=False,
        )
        self.assertEqual(mask_la.mode, "L")
        self.assertEqual(np.array(mask_la)[15, 15], 255)

        overlay_la = apply_template_matching(
            image=base_la,
            template=patch_la,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=True,
            alpha=0.6,
        )
        self.assertEqual(overlay_la.mode, "RGBA")
        self.assertEqual(overlay_la.size, base_la.size)

        # Test "La" (premultiplied alpha) mode mask and overlay
        base_la_prem = base_la.convert("La")
        patch_la_prem = patch_la.convert("La")

        mask_prem = apply_template_matching(
            image=base_la_prem,
            template=patch_la_prem,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=False,
        )
        self.assertEqual(mask_prem.mode, "L")
        self.assertEqual(np.array(mask_prem)[15, 15], 255)

        overlay_prem = apply_template_matching(
            image=base_la_prem,
            template=patch_la_prem,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=True,
            alpha=0.6,
        )
        self.assertEqual(overlay_prem.mode, "RGBA")
        self.assertEqual(overlay_prem.size, base_la_prem.size)

    def test_i16_high_depth_image_matching(self) -> None:
        # Create an 'I;16' 16-bit high-depth image with values 5000..40000
        img_i16 = Image.new("I;16", (30, 30))
        for y in range(30):
            for x in range(30):
                img_i16.putpixel((x, y), 5000)
        for y in range(12, 18):
            for x in range(12, 18):
                img_i16.putpixel((x, y), 20000 + (x - 12) * 2000 + (y - 12) * 1000)

        tmpl_i16 = img_i16.crop((12, 12, 18, 18))

        # Binary mask
        mask = apply_template_matching(
            image=img_i16,
            template=tmpl_i16,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=False,
        )
        self.assertEqual(mask.mode, "L")
        self.assertEqual(np.array(mask)[15, 15], 255)

        # Superimposed overlay
        overlay = apply_template_matching(
            image=img_i16,
            template=tmpl_i16,
            size=None,
            rotation=0.0,
            threshold=0.9,
            superimpose=True,
            alpha=0.6,
        )
        self.assertEqual(overlay.mode, "RGB")
        self.assertEqual(overlay.size, img_i16.size)


if __name__ == "__main__":
    unittest.main()
