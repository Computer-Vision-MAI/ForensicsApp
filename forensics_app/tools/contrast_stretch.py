import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image
import warnings

# Importamos las herramientas de scikit-image mostradas en la referencia
from skimage import exposure, color, img_as_float, img_as_ubyte

from .dialogs import ask_choice
from tkinter import simpledialog

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult



def apply_contrast_enhancement(img: Image.Image, method: str, clip_percent: int) -> Image.Image:
    """Mejora el contraste protegiendo los colores en LAB o directamente en Grayscale."""
    mode = img.mode
    
    # 1. Detectar si la imagen es monocromática
    is_grayscale = mode in ("L", "LA", "1", "I", "F")
    
    # 2. Extraer el canal alfa de forma segura
    has_alpha = False
    alpha_channel = None
    
    if mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        has_alpha = True
        if is_grayscale:
            img = img.convert("LA")
            img_array = np.array(img)
            alpha_channel = img_array[..., 1]
            work_array = img_array[..., 0]
        else:
            img = img.convert("RGBA")
            img_array = np.array(img)
            alpha_channel = img_array[..., 3]
            work_array = img_array[..., :3]
    else:
        if is_grayscale:
            img = img.convert("L")
            work_array = np.array(img)
        else:
            img = img.convert("RGB")
            work_array = np.array(img)

    # 3. Convertir a float en el rango [0, 1] de forma nativa
    float_img = img_as_float(work_array)

    # 4. Preparar el canal objetivo y normalizar a [0, 1] para skimage
    if is_grayscale:
        target_channel = float_img
    else:
        lab_image = color.rgb2lab(float_img)
        # Normalizar el canal L (0-100) a [0, 1] para que las matemáticas no colapsen
        target_channel = lab_image[..., 0] / 100.0

    # 5. Aplicar el algoritmo (ahora target_channel SIEMPRE está entre 0.0 y 1.0)
    if method == "percentile":
        p_low, p_high = np.percentile(target_channel, (clip_percent, 100 - clip_percent))
        enhanced = exposure.rescale_intensity(
            target_channel, 
            in_range=(p_low, p_high), 
            out_range=(0, 1)
        )
        
    elif method == "equalize":
        enhanced = exposure.equalize_hist(target_channel)
            
    elif method == "adaptive":
        enhanced = exposure.equalize_adapthist(target_channel, clip_limit=0.03)
            
    else:
        enhanced = target_channel

    # 6. Reconstruir la imagen y empaquetar en 8-bits
    if is_grayscale:
        final_8bit = img_as_ubyte(enhanced)
        output_mode = "LA" if has_alpha else "L"
    else:
        # Restaurar el rango correcto del espacio LAB [0, 100]
        lab_image[..., 0] = enhanced * 100.0
        
        # Ignorar de forma segura el warning de out-of-gamut al transformar a RGB
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enhanced_rgb = color.lab2rgb(lab_image)
            
        final_8bit = img_as_ubyte(enhanced_rgb)
        output_mode = "RGBA" if has_alpha else "RGB"

    # 7. Reensamblar el canal Alfa si existía
    if has_alpha:
        final_array = np.dstack((final_8bit, alpha_channel))
    else:
        final_array = final_8bit

    out_image = Image.fromarray(final_array, mode=output_mode)
    
    # Mantener el modo base para casos exóticos
    if mode in ("1", "I", "F"):
        out_image = out_image.convert("L")

    return out_image

class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast Enhancement"
    category = "Enhancement"
    description = "Normalizes image contrast using multiple algorithms (Percentile, Equalization, CLAHE)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        
        # 1. Seleccionar el modo usando el ChoiceDialog reutilizable
        options = ("Percentile Stretching", "Histogram Equalization", "Adaptive (CLAHE)")
        methods = ("percentile", "equalize", "adaptive")
        
        choice_idx = ask_choice(
            parent, 
            title="Contrast Enhancement", 
            prompt="Select Enhancement Method:", 
            options=options
        )
        
        if choice_idx is None:
            return None  # El usuario canceló el diálogo
            
        method = methods[choice_idx]
        percent = 0
        
        # 2. Si es percentil, encadenar un segundo diálogo para el porcentaje
        if method == "percentile":
            percent = simpledialog.askinteger(
                "Clip Percentage",
                "Enter clip percentage (0-10):",
                initialvalue=2,
                minvalue=0,
                maxvalue=10,
                parent=parent
            )
            if percent is None:
                return None  # El usuario canceló el input del porcentaje
        
        # 3. Aplicar la transformación matemática
        out_image = apply_contrast_enhancement(document.current, method, percent)
        
        return ToolResult(
            image=out_image,
            message=f"Contrast enhanced successfully using {method}.",
            details={
                "Operation": "Contrast Enhancement", 
                "Method": method.capitalize(),
                "Clip Percentage": f"{percent}%" if method == "percentile" else "N/A",
                "Original Mode": document.current.mode,
            }
        )