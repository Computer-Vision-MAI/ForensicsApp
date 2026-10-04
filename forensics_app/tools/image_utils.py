from __future__ import annotations
import numpy as np
from PIL import Image
from skimage import color, img_as_float

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
    is_high_depth = mode in ("I", "F") or mode.startswith("I;16")
    has_alpha = mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    
    alpha_channel = None
    
    # 1. Get raw normalized float data [0, 1]
    if is_high_depth:
        raw_array = np.asarray(img, dtype=np.float64)
        valid_mask = np.isfinite(raw_array)
        
        if np.any(valid_mask):
            c_min = np.min(raw_array[valid_mask])
            c_max = np.max(raw_array[valid_mask])
            raw_array = np.nan_to_num(raw_array, nan=c_min, posinf=c_max, neginf=c_min)
            
            if c_max > c_min:
                float_img = (raw_array - c_min) / (c_max - c_min)
            else:
                float_img = np.zeros_like(raw_array)
        else:
            float_img = np.zeros_like(raw_array)
            
        if not as_grayscale:
            float_img = color.gray2rgb(float_img)
            
    else:
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

    # 2. Map to the correct color space channel
    if as_grayscale:
        return float_img, None, alpha_channel
    else:
        lab_image = color.rgb2lab(float_img)
        target_channel = lab_image[..., 0] / 100.0
        return target_channel, lab_image, alpha_channel