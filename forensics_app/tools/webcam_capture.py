"""Webcam photo capture tool for acquiring clean, unannotated frames.

Allows capturing a raw webcam photo to set as the active working document
(document.current) or saving to disk, with an option to auto-crop the face
to match the LFW dataset aspect ratio (50:37) for Face Recognition.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .face_detection import detect_faces_yunet

# LFW face crops are 50 px high by 37 px wide; the crop is 1.35x the detected face height.
LFW_ASPECT_W_OVER_H = 37.0 / 50.0
LFW_CROP_HEIGHT_FACTOR = 1.35


def compute_lfw_crop_box(
    face_box: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return an LFW-shaped (50:37) crop box ``(x1, y1, x2, y2)`` around a detected face.

    The crop is centred on the face. Near the image borders it is shifted inside the
    frame instead of being cut off, so it keeps the LFW aspect ratio; it is shrunk
    only when it is larger than the frame itself.
    """
    bx, by, bw, bh = face_box
    img_w, img_h = image_size
    cx = bx + bw / 2.0
    cy = by + bh / 2.0

    crop_h = bh * LFW_CROP_HEIGHT_FACTOR
    crop_w = crop_h * LFW_ASPECT_W_OVER_H
    if crop_w > img_w or crop_h > img_h:
        shrink = min(img_w / crop_w, img_h / crop_h)
        crop_w *= shrink
        crop_h *= shrink

    left = min(max(cx - crop_w / 2.0, 0.0), img_w - crop_w)
    top = min(max(cy - crop_h / 2.0, 0.0), img_h - crop_h)

    x1 = min(int(round(left)), img_w - 1)
    y1 = min(int(round(top)), img_h - 1)
    x2 = min(img_w, max(x1 + 1, int(round(left + crop_w))))
    y2 = min(img_h, max(y1 + 1, int(round(top + crop_h))))
    return x1, y1, x2, y2


class WebcamPhotoDialog(tk.Toplevel):
    """Clean live webcam window without overlays, offering capture and save options."""

    def __init__(self, parent: tk.Misc) -> None:
        """Open the webcam and show a clean live preview until the user captures or closes."""
        super().__init__(parent)
        self.title("Webcam Photo Capture")
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)

        self.captured_image: Image.Image | None = None
        self.capture_mode: str = "full"
        self.saved_path: Path | None = None
        self._cap: cv2.VideoCapture | None = None
        self._current_frame_pil: Image.Image | None = None
        self._current_frame_bgr: np.ndarray | None = None
        self._photo_ref: ImageTk.PhotoImage | None = None
        self._running = True

        # 1. Video Display
        self._video_label = ttk.Label(self)
        self._video_label.pack(padx=10, pady=10)

        # 2. Status message
        self._status_var = tk.StringVar(
            self, value="Webcam live. Position your face in view, then select an action below."
        )
        status_label = ttk.Label(
            self, textvariable=self._status_var, font="TkSmallCaptionFont", foreground="#333333"
        )
        status_label.pack(fill="x", padx=12, pady=(0, 6))

        # 3. Action Buttons
        btn_bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        btn_bar.pack(fill="x")

        ttk.Button(
            btn_bar,
            text="Capture Full Frame",
            command=self._on_capture_full,
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            btn_bar,
            text="Capture & Crop Face (LFW)",
            command=self._on_capture_crop,
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            btn_bar,
            text="Save Image As...",
            command=self._on_save_as,
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            btn_bar,
            text="Cancel / Close",
            command=self._on_close,
        ).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Open webcam device
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showwarning(
                "Webcam Error",
                "Could not connect to a webcam device.\n"
                "Please verify camera permissions or ensure a camera is plugged in.",
                parent=self,
            )
            self.destroy()
            return

        self._update_loop()
        try:
            self.grab_set()  # modal: the main window waits until the capture is done
        except tk.TclError:
            pass  # window not viewable yet on some platforms; still usable, just not modal
        self.wait_window(self)

    def _update_loop(self) -> None:
        """Grab a frame and refresh the clean (overlay-free) preview."""
        if not self._running or self._cap is None or not self._cap.isOpened():
            return

        ret, frame_bgr = self._cap.read()
        if ret:
            # Horizontal flip for natural mirror feedback
            frame_bgr = cv2.flip(frame_bgr, 1)
            self._current_frame_bgr = frame_bgr.copy()

            # Clean raw RGB image without any bounding boxes or landmarks
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            self._current_frame_pil = pil_img.copy()

            # Display resized preview if window is large
            preview = pil_img
            if preview.width > 640:
                scale = 640 / preview.width
                preview = preview.resize(
                    (640, int(preview.height * scale)), Image.Resampling.BILINEAR
                )

            self._photo_ref = ImageTk.PhotoImage(preview, master=self)
            self._video_label.configure(image=self._photo_ref)

        self.after(30, self._update_loop)

    def _on_capture_full(self) -> None:
        """Capture the full clean camera frame."""
        if self._current_frame_pil is not None:
            self.captured_image = self._current_frame_pil.copy()
            self.capture_mode = "Full Frame"
        self._on_close()

    def _on_capture_crop(self) -> None:
        """Detect face using YuNet and crop with LFW aspect ratio (50x37)."""
        if self._current_frame_pil is None or self._current_frame_bgr is None:
            self._on_close()
            return

        # Find the face coordinates with YuNet
        try:
            _, faces = detect_faces_yunet(self._current_frame_bgr, score_threshold=0.5)
        except Exception as error:  # e.g. model download failed: report it, keep the full frame
            self.captured_image = self._current_frame_pil.copy()
            self.capture_mode = (
                f"Full Frame (face detection failed: {error}. "
                "Check the YuNet model / network connection and try again)"
            )
            self._status_var.set(self.capture_mode)
            self._on_close()
            return

        if faces:
            # Pick largest detected face
            best_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            crop_box = compute_lfw_crop_box(best_face["box"], self._current_frame_pil.size)
            self.captured_image = self._current_frame_pil.crop(crop_box)
            self.capture_mode = "Cropped Face (LFW 50:37 Format)"
        else:
            # Fallback to full frame if no face was found
            self.captured_image = self._current_frame_pil.copy()
            self.capture_mode = "Full Frame (No face detected)"

        self._on_close()

    def _on_save_as(self) -> None:
        """Save the current frame directly to disk."""
        if self._current_frame_pil is None:
            return

        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Captured Photo",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
        )
        if save_path:
            out_path = Path(save_path)
            try:
                self._current_frame_pil.save(out_path)
            except (OSError, ValueError) as error:
                messagebox.showerror("Could not save photo", str(error), parent=self)
                return
            self.saved_path = out_path
            self._status_var.set(f"Saved photo to: {out_path.name}")

    def _on_close(self) -> None:
        """Stop the preview loop, release the camera, and close the dialog."""
        self._running = False
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        self.destroy()


class WebcamCaptureTool(ForensicsTool):
    """Tool to capture clean photos from the webcam to use as working image or save to disk."""

    tool_id = "webcam_capture"
    title = "Webcam Photo Capture"
    category = "Face Analysis"
    description = "Take clean webcam photos with optional LFW face auto-crop for face recognition."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Open the webcam dialog and return the captured photo, if any."""
        dialog = WebcamPhotoDialog(parent)
        captured = dialog.captured_image
        if captured is None:
            return None

        # Update document.current so the captured image is ready for other tools
        if document.current is None:
            document.original = captured.copy()
            document.current = captured.copy()
            preview_only = True
        else:
            preview_only = False

        details: dict[str, Any] = {
            "Source": "Webcam Live",
            "Capture Type": dialog.capture_mode,
            "Resolution": f"{captured.width} × {captured.height} px",
            "Color Mode": captured.mode,
            "Saved to Disk": str(dialog.saved_path) if dialog.saved_path else "Not saved (Working document only)",
        }

        msg = (
            f"Captured webcam photo ({dialog.capture_mode}, {captured.width}×{captured.height} px). "
            f"Image is now ready in the working document."
        )

        return ToolResult(
            message=msg,
            image=captured,
            details=details,
            preview_only=preview_only,
        )
