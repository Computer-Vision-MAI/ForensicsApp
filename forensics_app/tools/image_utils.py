from __future__ import annotations
from typing import Callable

import numpy as np
from PIL import Image
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
