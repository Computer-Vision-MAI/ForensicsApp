from __future__ import annotations
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, img_as_float

from forensics_app.core.channels import ALPHA, MODE_CHANNELS, NUMERIC_MODES

def extract_target_channel(
    img: Image.Image, as_grayscale: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Safely extracts the target float channel (Grayscale or LAB Luminance) in the [0, 1] range.
    Bypasses Pillow's 8-bit clipping for high-depth images.
    
    Returns:
        - target_channel: The 2D numpy array [0, 1] to process.
        - lab_image: The full LAB image if as_grayscale is False, otherwise None.
        - alpha_channel: The alpha mask if present, otherwise None.
    """
    mode = img.mode
    
    # 1. Enforce strict allowlist covering all recognized Pillow modes
    supported_modes = {
        "1", "L", "I", "F", "P", "PA", "RGB", "CMYK", "RGBA", "RGBX", "RGBa", "LA", "La"
    }

    if mode not in supported_modes and not mode.startswith("I;16"):
        raise ValueError(f"Unsupported image mode: '{mode}'. Cannot safely extract channels.")

    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    
    # 2. Safely detect all alpha-bearing modes
    has_alpha = mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in img.info
    
    alpha_channel = None
    
    # 3. Normalize to [0, 1], using the finite pixel range for F mode
    if is_high_depth:

        float_img = normalize_high_depth_image(img)

        if not as_grayscale:
            float_img = color.gray2rgb(float_img)
            
    else:
        # Pillow safely converts exotic modes (CMYK, YCbCr, HSV, RGBX) to RGB/RGBA here.
        # This keeps the math explicit and guarantees the LAB conversion later won't fail.
        if has_alpha:
            if as_grayscale:
                arr = np.array(img.convert("LA"))
                float_img = img_as_float(arr[..., 0])
                alpha_channel = arr[..., 1]
            else:
                arr = np.array(img.convert("RGBA"))
                float_img = img_as_float(arr[..., :3])
                alpha_channel = arr[..., 3]
        else:
            if as_grayscale:
                float_img = img_as_float(np.array(img.convert("L")))
            else:
                float_img = img_as_float(np.array(img.convert("RGB")))

    # 4. Map to the correct color space channel
    if as_grayscale:
        return float_img, None, alpha_channel
    else:
        lab_image = color.rgb2lab(float_img)
        target_channel = lab_image[..., 0] / 100.0
        return target_channel, lab_image, alpha_channel

def normalize_high_depth_image(img: Image.Image) -> np.ndarray:
    """
    Normalize high-depth images (I, F, I;16) to [0, 1] range.
    Handles NaN and infinite values by replacing them with the min/max of valid pixels.
    """
    orig_array = np.asarray(img)
    
    # Handle high-depth images by converting to float64 for calculations
    raw_array = orig_array.astype(np.float64)
    valid_mask = np.isfinite(raw_array)

    if np.any(valid_mask):
        c_min = np.min(raw_array[valid_mask])
        c_max = np.max(raw_array[valid_mask])
        raw_array = np.nan_to_num(raw_array, nan=c_min, posinf=c_max, neginf=c_min)
        
        if c_max > c_min:
            normalized_array = (raw_array - c_min) / (c_max - c_min)
        else:
            normalized_array = np.zeros_like(raw_array)
    else:
        normalized_array = np.zeros_like(raw_array)

    return normalized_array

def is_grayscale(img: Image.Image) -> bool:
    """Return True if the image has a single intensity channel, with or without alpha."""
    return img.mode in ("1", "L", "LA", "La") or img.mode in NUMERIC_MODES


def image_to_unit_array(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the pixels as floats in [0, 1], plus the alpha channel if there is one.

    Grayscale modes (1, L, LA, La and the numeric ones) give an H x W array; every
    other mode is converted to RGB and gives an H x W x 3 array. Alpha is returned
    apart, as 8-bit, so filters never touch it.
    """
    if img.mode not in MODE_CHANNELS:
        raise ValueError(f"Unsupported image mode: '{img.mode}'.")
    if img.mode in NUMERIC_MODES:
        return normalize_high_depth_image(img), None

    is_gray = is_grayscale(img)
    # Pillow cannot convert La to other modes directly
    if img.mode == "La":
        img = img.convert("LA")
    has_alpha = ALPHA in MODE_CHANNELS[img.mode] or "transparency" in img.info

    if has_alpha:
        arr = np.array(img.convert("LA" if is_gray else "RGBA"))
        colors = arr[..., 0] if is_gray else arr[..., :3]
        return img_as_float(colors), arr[..., -1]
    return img_as_float(np.array(img.convert("L" if is_gray else "RGB"))), None


def unit_array_to_image(pixels: np.ndarray, alpha: np.ndarray | None = None) -> Image.Image:
    """Return [0, 1] floats as an 8-bit L or RGB image (LA or RGBA with ``alpha``).

    Values outside [0, 1] are clipped.
    """
    data = np.round(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
    if alpha is not None:
        data = np.dstack((data, alpha))
    return Image.fromarray(data)


def per_channel(function: Callable[[np.ndarray], np.ndarray], pixels: np.ndarray) -> np.ndarray:
    """Apply a 2-D ``function`` to a gray array, or to each channel of a color one."""
    # A gray image has a single channel: filter it directly
    if pixels.ndim == 2:
        return function(pixels)
    # A color image is filtered one channel at a time (red, then green, then blue)
    # and the three results are stacked back into one color image. This is the same as:
    #   channels = []
    #   for index in range(3):
    #       channels.append(function(pixels[..., index]))
    #   return np.dstack(channels)
    # Callers that need extra arguments pass a lambda, a small function without a name:
    #   lambda channel: filters.gaussian(channel, sigma=2)
    # means "given a channel, blur it with sigma 2"
    return np.dstack([function(pixels[..., index]) for index in range(pixels.shape[-1])])


def to_grayscale_array(img: Image.Image) -> np.ndarray:
    """Safely convert any PIL Image mode to a 2D float64 numpy array in [0, 1].

    Supports standard modes (RGB, RGBA, L, 1, P, CMYK, etc.) and high-depth modes
    (I, F, I;16) as well as tricky formats like 'La'.
    """
    mode = img.mode
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    if is_high_depth:
        return normalize_high_depth_image(img).astype(np.float64)
    if mode == "La":
        conv = img.convert("LA")
    elif mode == "RGBa":
        conv = img.convert("RGBA")
    else:
        conv = img
    return np.array(conv.convert("L"), dtype=np.float64) / 255.0


def render_mask_overlay(
    img: Image.Image,
    mask: np.ndarray,
    superimpose: bool = False,
    alpha: float = 0.6,
) -> Image.Image:
    """Render a 2D boolean mask onto an image as a binary mask or superimposed red overlay.

    Args:
        img: Input PIL Image in any mode.
        mask: Boolean 2D ndarray of shape (H, W).
        superimpose: If True, overlay True pixels in red with alpha blending.
                     If False, return a binary mask (mode 'L') with 255 for True and 0 for False.
        alpha: Blending weight in [0, 1] for superimposition (1.0 = solid red).

    Returns:
        A PIL Image in mode 'L' if superimpose is False, or mode 'RGB' if superimpose is True.
    """
    if alpha < 0.0 or alpha > 1.0 or np.isnan(alpha):
        raise ValueError("Alpha must be between 0.0 and 1.0.")

    if not superimpose:
        return Image.fromarray((mask * 255).astype(np.uint8))

    mode = img.mode
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    conv = img
    # Extract original alpha channel if present
    if mode == "La":
        conv = img.convert("LA")
    elif mode == "RGBa":
        conv = img.convert("RGBA")
    alpha_arr = None
    if "A" in conv.getbands():
        alpha_arr = np.array(conv.getchannel("A"), dtype=np.uint8)
    elif "transparency" in conv.info:
        rgba = conv.convert("RGBA")
        alpha_arr = np.array(rgba.getchannel("A"), dtype=np.uint8)

    if is_high_depth:
        gray_arr = (normalize_high_depth_image(img) * 255.0).astype(np.float32)
        rgb_arr = np.stack((gray_arr,) * 3, axis=-1)
    else:
        rgb_arr = np.array(conv.convert("RGB"), dtype=np.float32)

    edge_color = np.array([255, 0, 0], dtype=np.float32)
    rgb_arr[mask] = rgb_arr[mask] * (1.0 - alpha) + edge_color * alpha
    rgb_u8 = np.clip(rgb_arr, 0, 255).astype(np.uint8)

    if alpha_arr is not None:
        if alpha > 0.0:
            alpha_arr[mask] = np.maximum(alpha_arr[mask], 255)
        rgba_arr = np.dstack((rgb_u8, alpha_arr))
        return Image.fromarray(rgba_arr, mode="RGBA")
    return Image.fromarray(rgb_u8, mode="RGB")


def render_feature_overlay(
    img: Image.Image,
    coords: np.ndarray | None,
    superimpose: bool = False,
    alpha: float = 0.6,
    marker_radius: int = 0,
) -> Image.Image:
    """Render coordinates onto an image as a binary mask or superimposed red overlay.

    Args:
        img: Input PIL Image in any mode.
        coords: (N, 2) array of (row, col) coordinates, or None.
        superimpose: If True, overlay points in red over the original image with alpha blending.
                     If False, return a binary mask (mode 'L') with points as 255 on a black background.
        alpha: Blending weight in [0, 1] for superimposition (1.0 = solid red).
        marker_radius: Radius in pixels around each coordinate. 0 means a single pixel.

    Returns:
        A PIL Image in mode 'L' if superimpose is False, or mode 'RGB' if superimpose is True.
    """
    w, h = img.size
    mask = np.zeros((h, w), dtype=bool)

    if coords is not None and len(coords) > 0:
        r_coords = np.clip(np.round(coords[:, 0]).astype(int), 0, h - 1)
        c_coords = np.clip(np.round(coords[:, 1]).astype(int), 0, w - 1)
        if marker_radius <= 0:
            mask[r_coords, c_coords] = True
        else:
            for r, c in zip(r_coords, c_coords):
                r_min = max(0, r - marker_radius)
                r_max = min(h, r + marker_radius + 1)
                c_min = max(0, c - marker_radius)
                c_max = min(w, c + marker_radius + 1)
                for rr in range(r_min, r_max):
                    for cc in range(c_min, c_max):
                        if (rr - r) ** 2 + (cc - c) ** 2 <= marker_radius ** 2:
                            mask[rr, cc] = True

    return render_mask_overlay(img, mask, superimpose=superimpose, alpha=alpha)


def render_keypoint_overlay(
    img: Image.Image,
    coords: np.ndarray | None,
    scales: np.ndarray | None = None,
    orientations: np.ndarray | None = None,
    superimpose: bool = False,
    alpha: float = 0.6,
    base_radius: float = 3.0,
    draw_orientation: bool = True,
) -> Image.Image:
    """Render keypoints with circumference circles and orientation rays.

    Draws each keypoint with a circular ring (circumference) proportional to its scale,
    a center dot, and an orientation ray pointing in the direction of the dominant gradient.

    Args:
        img: Input PIL Image in any mode.
        coords: (N, 2) array of (row, col) coordinates.
        scales: (N,) array of scale values in pixels, or None.
        orientations: (N,) array of angles in radians, or None.
        superimpose: If True, overlay in red over the original image with alpha blending.
                     If False, return a binary mask (mode 'L') with features as 255 on 0.
        alpha: Blending weight in [0, 1] when superimpose is True.
        base_radius: Baseline / minimum radius in pixels for each circle.
        draw_orientation: Whether to draw the orientation ray from center to circumference.

    Returns:
        PIL Image in mode 'L' if superimpose is False, or mode 'RGB' if superimpose is True.
    """
    w, h = img.size
    mask_im = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_im)

    if coords is not None and len(coords) > 0:
        for idx in range(len(coords)):
            r, c = coords[idx]
            if np.isnan(r) or np.isnan(c):
                continue
            x, y = float(c), float(r)
            if not (0 <= x < w and 0 <= y < h):
                continue

            # Determine circle radius
            if scales is not None and idx < len(scales) and not np.isnan(scales[idx]):
                r_val = float(scales[idx])
                radius = max(base_radius, r_val)
            else:
                radius = base_radius

            # Draw circle (circumference)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=255,
                width=1,
            )
            # Center point
            draw.point((int(round(x)), int(round(y))), fill=255)

            # Draw orientation ray
            if draw_orientation and orientations is not None and idx < len(orientations):
                theta = float(orientations[idx])
                if not np.isnan(theta):
                    end_x = x + radius * np.cos(theta)
                    end_y = y + radius * np.sin(theta)
                    draw.line([(x, y), (end_x, end_y)], fill=255, width=1)

    mask = np.array(mask_im, dtype=bool)
    return render_mask_overlay(img, mask, superimpose=superimpose, alpha=alpha)


