import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image

# Importamos las herramientas de scikit-image mostradas en la referencia
from skimage import exposure, color, img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ContrastOptionsDialog(tk.Toplevel):
    """Diálogo personalizado para elegir el método de contraste."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Contrast Options")
        self.result = None
        self.geometry("300x200")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # Variables
        self.method_var = tk.StringVar(value="percentile")
        self.percent_var = tk.IntVar(value=2)

        # UI Layout
        ttk.Label(self, text="Select Enhancement Method:").pack(anchor="w", padx=10, pady=(10, 5))
        
        ttk.Radiobutton(self, text="Percentile Stretching", variable=self.method_var, value="percentile").pack(anchor="w", padx=20)
        ttk.Radiobutton(self, text="Histogram Equalization", variable=self.method_var, value="equalize").pack(anchor="w", padx=20)
        ttk.Radiobutton(self, text="Adaptive (CLAHE)", variable=self.method_var, value="adaptive").pack(anchor="w", padx=20)

        frame_pct = ttk.Frame(self)
        frame_pct.pack(fill="x", padx=20, pady=10)
        ttk.Label(frame_pct, text="Clip Percentage (0-10):").pack(side="left")
        ttk.Spinbox(frame_pct, from_=0, to=10, textvariable=self.percent_var, width=5).pack(side="left", padx=5)

        ttk.Button(self, text="Apply", command=self.on_apply).pack(pady=10)
        self.wait_window(self)

    def on_apply(self):
        self.result = {
            "method": self.method_var.get(),
            "percent": self.percent_var.get()
        }
        self.destroy()


def apply_contrast_enhancement(img: Image.Image, method: str, clip_percent: int) -> Image.Image:
    """Mejora el contraste operando de forma segura sobre la luminancia (espacio LAB)."""
    mode = img.mode
    
    # Extraer el canal alfa si existe para no corromper la transparencia
    has_alpha = False
    alpha_channel = None
    if mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
        has_alpha = True
        img_array = np.array(img)
        alpha_channel = img_array[..., 3]
        rgb_array = img_array[..., :3]
    else:
        img = img.convert("RGB")
        rgb_array = np.array(img)

    # 1. Convertir a float [0, 1] y pasar a LAB
    # skimage.color.rgb2lab devuelve L en rango [0, 100]
    float_rgb = img_as_float(rgb_array) #
    lab_image = color.rgb2lab(float_rgb)
    L_channel = lab_image[..., 0]

    # 2. Aplicar el algoritmo seleccionado sobre el canal L
    if method == "percentile":
        # Extraemos los percentiles superior e inferior
        p_low, p_high = np.percentile(L_channel, (clip_percent, 100 - clip_percent)) #[cite: 6]
        # Re-escalamos forzando la salida a mantenerse en [0, 100] para el espacio LAB
        L_enhanced = exposure.rescale_intensity(
            L_channel, 
            in_range=(p_low, p_high), 
            out_range=(0, 100)
        ) #[cite: 6]
        
    elif method == "equalize":
        # equalize_hist devuelve [0, 1], lo multiplicamos por 100 para LAB
        L_enhanced = exposure.equalize_hist(L_channel) * 100 #[cite: 6]
        
    elif method == "adaptive":
        # CLAHE requiere normalizar L a [0, 1] internamente antes de operar
        L_normalized = L_channel / 100.0
        L_enhanced = exposure.equalize_adapthist(L_normalized, clip_limit=0.03) * 100 #[cite: 6]
        
    else:
        L_enhanced = L_channel

    # 3. Reemplazar el canal L, volver a RGB y empaquetar en 8-bits
    lab_image[..., 0] = L_enhanced
    enhanced_rgb = color.lab2rgb(lab_image)
    final_rgb_8bit = img_as_ubyte(enhanced_rgb)

    # 4. Restaurar canales y modos originales
    if has_alpha:
        final_array = np.dstack((final_rgb_8bit, alpha_channel))
        output_mode = "RGBA"
    else:
        final_array = final_rgb_8bit
        output_mode = "RGB"

    # Si la original era monocromática (L o 1), la devolvemos a escala de grises
    out_image = Image.fromarray(final_array, mode=output_mode)
    if mode in ("L", "1", "I", "F"):
        out_image = out_image.convert("L")

    return out_image


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast Enhancement"
    category = "Enhancement"
    description = "Normalizes image contrast using multiple algorithms (Percentile, Equalization, CLAHE)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        
        # Abrir el diálogo para que el usuario elija
        dialog = ContrastOptionsDialog(parent)
        if not dialog.result:
            return None  # El usuario cerró la ventana sin aplicar
            
        method = dialog.result["method"]
        percent = dialog.result["percent"]
        
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