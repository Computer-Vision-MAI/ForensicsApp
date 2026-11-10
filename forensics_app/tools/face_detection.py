"""Face detection tool supporting YuNet (deep learning) and Viola-Jones Haar cascade.

Follows 'docs/face_detection_and_recognition.ipynb' and 'docs/Materials-CV-MAI-Set6.pdf'.
Includes:
  - Dynamic sizing for bounding boxes, text font, and landmark circles proportional to face size.
  - Full details for all detected faces and their 5 landmarks.
  - Customizable YuNet parameters (Score threshold, NMS threshold, Top K).
  - Viola-Jones attentional cascade with customizable stages trained on data/faces_non_faces.
  - Safe webcam preview that preserves existing document undo/redo history.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageTk

# Suppress internal OpenCV C++ DNN backend warning:
# '[ WARN:0@...] global net_impl_backend.cpp:345 setPreferableTarget Targets are not supported by the new graph engine for now'
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from forensics_app.core import ImageDocument
from forensics_app.paths import MODELS_DIR
from .base import ForensicsTool, ToolResult
from .dialogs import TaskCancelled, load_image_file, run_with_progress
from .haar_adaboost import (
    CANONICAL_PATCH_SIZE,
    DEFAULT_CASCADE_PATH,
    DEFAULT_TRAIN_NEG,
    DEFAULT_TRAIN_POS,
    ViolaJonesCascade,
    activated_classifiers,
    draw_haar_overlay,
    ensure_default_cascade_model,
    load_cascade_model,
    patch_to_normalized_integral,
    train_viola_jones_cascade,
)

YUNET_MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_MIN_SIZE_BYTES = 100_000
YUNET_DOWNLOAD_TIMEOUT_S = 30.0


def ensure_yunet_model(
    path: Path = YUNET_MODEL_PATH, timeout: float = YUNET_DOWNLOAD_TIMEOUT_S
) -> Path:
    """Ensure the YuNet ONNX weights exist locally; download if needed.

    The download goes to a temporary file next to ``path`` and only replaces it once
    complete, so an interrupted download never leaves a truncated model behind.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size >= YUNET_MIN_SIZE_BYTES:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(YUNET_MODEL_URL, timeout=timeout) as response:
            shutil.copyfileobj(response, out)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def compute_face_drawing_params(box_w: int, box_h: int) -> dict[str, Any]:
    """Calculate line thickness, font scale, and landmark radius proportional to face size.

    Prevents text and markers from obscuring small faces in crowds or appearing
    microscopic in high-resolution webcam frames.
    """
    box_dim = max(box_w, box_h)
    line_thickness = max(1, int(round(box_dim * 0.015)))
    font_scale = max(0.3, min(1.4, box_dim * 0.0035))
    font_thickness = max(1, int(round(font_scale * 1.6)))
    landmark_radius = max(2, int(round(box_dim * 0.022)))
    return {
        "line_thickness": line_thickness,
        "font_scale": font_scale,
        "font_thickness": font_thickness,
        "landmark_radius": landmark_radius,
    }


def create_yunet_detector(
    input_size: tuple[int, int],
    model_path: Path = YUNET_MODEL_PATH,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> Any:
    """Load the YuNet model once and return a reusable ``cv2.FaceDetectorYN``.

    ``input_size`` is (width, height); ``detect_faces_with_detector`` updates it per frame.
    """
    ensure_yunet_model(model_path)
    return cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        tuple(int(v) for v in input_size),
        float(score_threshold),
        float(nms_threshold),
        int(top_k),
    )


def detect_faces_yunet(
    image_bgr: np.ndarray,
    model_path: Path = YUNET_MODEL_PATH,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run OpenCV YuNet face detector on a single image with dynamically scaled annotations."""
    h, w = image_bgr.shape[:2]
    detector = create_yunet_detector((w, h), model_path, score_threshold, nms_threshold, top_k)
    return detect_faces_with_detector(detector, image_bgr)


def detect_faces_with_detector(
    detector: Any, image_bgr: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Detect and annotate faces with an existing YuNet detector (e.g. one per webcam session)."""
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()

    detector.setInputSize((w, h))
    _, faces = detector.detect(image_bgr)
    results: list[dict[str, Any]] = []

    if faces is None or len(faces) == 0:
        return annotated, results

    landmark_names = [
        "Right eye",
        "Left eye",
        "Nose",
        "Right mouth corner",
        "Left mouth corner",
    ]

    for idx, face in enumerate(faces):
        x, y, box_w, box_h = face[:4].astype(int)
        conf = float(face[14])

        # Clamp bounding box inside image
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w - 1, x + box_w)
        y2 = min(h - 1, y + box_h)
        actual_w = max(1, x2 - x1)
        actual_h = max(1, y2 - y1)

        # Proportional styling
        style = compute_face_drawing_params(actual_w, actual_h)
        l_thick = style["line_thickness"]
        f_scale = style["font_scale"]
        f_thick = style["font_thickness"]
        r_radius = style["landmark_radius"]

        # 1. Green bounding box (BGR: 0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), l_thick)

        # 2. Text placement (above box, or inside if too close to image top edge)
        label = f"Face {idx + 1}: {conf:.2f}"
        text_y = y1 - 6 if y1 > 20 else y1 + int(18 * f_scale) + 4
        cv2.putText(
            annotated,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_scale,
            (0, 255, 0),
            f_thick,
        )

        # 3. Five landmarks (strategic points) in Blue (BGR: 255, 0, 0)
        landmarks = face[4:14].reshape(5, 2).astype(int)
        pts_dict: dict[str, tuple[int, int]] = {}
        for (px, py), name in zip(landmarks, landmark_names):
            px_c = int(np.clip(px, 0, w - 1))
            py_c = int(np.clip(py, 0, h - 1))
            cv2.circle(annotated, (px_c, py_c), r_radius, (255, 0, 0), -1)
            pts_dict[name] = (px_c, py_c)

        results.append(
            {
                "box": (int(x1), int(y1), int(actual_w), int(actual_h)),
                "confidence": conf,
                "landmarks": pts_dict,
            }
        )

    return annotated, results


def scale_face_results(faces: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    """Return detections with boxes and landmarks scaled by ``scale`` (e.g. to a resized preview)."""
    if scale == 1.0:
        return faces

    def _s(value: int) -> int:
        """Scale one coordinate and round it to whole pixels."""
        return int(round(value * scale))

    scaled: list[dict[str, Any]] = []
    for face in faces:
        x, y, bw, bh = face["box"]
        scaled.append(
            {
                **face,
                "box": (_s(x), _s(y), max(1, _s(bw)), max(1, _s(bh))),
                "landmarks": {name: (_s(px), _s(py)) for name, (px, py) in face["landmarks"].items()},
            }
        )
    return scaled


class WebcamCaptureDialog(tk.Toplevel):
    """Real-time webcam preview dialog running YuNet face detection."""

    def __init__(
        self,
        parent: tk.Misc,
        model_path: Path = YUNET_MODEL_PATH,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        """Open the webcam and run live YuNet detection until a frame is captured or the dialog closes."""
        super().__init__(parent)
        self.title("Real-Time Webcam Face Detection (YuNet)")
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)

        self._model_path = model_path
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k

        self._cap: cv2.VideoCapture | None = None
        self._detector: Any = None  # created on the first frame, then reused
        self.captured_image: Image.Image | None = None
        self.faces_detected: int = 0
        self.detected_face_list: list[dict[str, Any]] = []

        # UI
        self._video_label = ttk.Label(self)
        self._video_label.pack(padx=10, pady=10)

        self._status_var = tk.StringVar(self, value="Starting webcam feed...")
        status_label = ttk.Label(self, textvariable=self._status_var, font="TkFixedFont")
        status_label.pack(fill="x", padx=10, pady=(0, 6))

        btn_bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        btn_bar.pack(fill="x")
        ttk.Button(btn_bar, text="Capture Frame", command=self._on_capture).pack(side="left")
        ttk.Button(btn_bar, text="Cancel / Close", command=self._on_close).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showwarning(
                "Webcam Error",
                "Could not connect to a webcam device.\n"
                "Please check permissions (macOS: Privacy -> Camera) or ensure camera is plugged in.",
                parent=self,
            )
            self.destroy()
            return

        self._photo_ref: ImageTk.PhotoImage | None = None
        self._running = True
        self._update_loop()
        self.wait_window(self)

    def _update_loop(self) -> None:
        """Grab a frame, detect faces with the reused detector, and refresh the preview."""
        if not self._running or self._cap is None or not self._cap.isOpened():
            return

        ret, frame_bgr = self._cap.read()
        if ret:
            frame_bgr = cv2.flip(frame_bgr, 1)

            if self._detector is None:
                h, w = frame_bgr.shape[:2]
                self._detector = create_yunet_detector(
                    (w, h),
                    self._model_path,
                    self._score_threshold,
                    self._nms_threshold,
                    self._top_k,
                )
            annotated_bgr, faces = detect_faces_with_detector(self._detector, frame_bgr)
            self.faces_detected = len(faces)

            self._status_var.set(
                f"Webcam Active — Detected faces: {len(faces)}. Click 'Capture Frame' to import preview."
            )

            frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            scale = 1.0
            if pil_img.width > 640:
                scale = 640 / pil_img.width
                pil_img = pil_img.resize(
                    (640, int(pil_img.height * scale)), Image.Resampling.BILINEAR
                )
            # The captured image is the (possibly downscaled) preview, so report the
            # face coordinates in that same scale.
            self.detected_face_list = scale_face_results(faces, scale)

            self._photo_ref = ImageTk.PhotoImage(pil_img, master=self)
            self._video_label.configure(image=self._photo_ref)
            self._current_frame = pil_img.copy()

        self.after(30, self._update_loop)

    def _on_capture(self) -> None:
        """Keep the current annotated frame and close the dialog."""
        if hasattr(self, "_current_frame"):
            self.captured_image = self._current_frame
        self._on_close()

    def _on_close(self) -> None:
        """Stop the preview loop, release the camera, and close the dialog."""
        self._running = False
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        self.destroy()


class FaceDetectionDialog(simpledialog.Dialog):
    """Dialog allowing user to choose method, customize YuNet/Viola-Jones params, and select input source."""

    def __init__(self, parent: tk.Misc, has_current_image: bool) -> None:
        """Show the face detection options for a document with or without an open image."""
        self._has_current_image = has_current_image
        self.selected_method: str = "yunet"
        self.selected_source: str = "current" if has_current_image else "file"
        self.preserve_history: bool = has_current_image

        # YuNet customizable parameters
        self.score_threshold: float = 0.60
        self.nms_threshold: float = 0.30
        self.top_k: int = 5000

        # Viola-Jones / Cascade parameters
        self.cascade_mode: str = "pretrained"  # 'pretrained' or 'train_new'
        self.cascade_stages_str: str = "2, 5, 10"
        self.train_samples: int = 250
        self.max_haar_features: int = 10

        self.user_confirmed = False
        super().__init__(parent, "Face Detection Options")

    def body(self, master: tk.Frame) -> tk.Widget:
        """Build the method, parameter, and input-source widgets."""
        # 1. Method Frame
        method_frame = ttk.LabelFrame(master, text="Detection Method", padding=8)
        method_frame.pack(fill="x", padx=6, pady=4)

        self._method_var = tk.StringVar(master, value="yunet")
        ttk.Radiobutton(
            method_frame,
            text="YuNet (Deep learning: proportional bounding box + 5 landmarks)",
            value="yunet",
            variable=self._method_var,
            command=self._on_method_changed,
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            method_frame,
            text="Haar / Viola-Jones (Attentional cascade on data/faces_non_faces)",
            value="haar",
            variable=self._method_var,
            command=self._on_method_changed,
        ).pack(anchor="w", pady=2)

        # 2. YuNet Parameters Frame
        self._yunet_frame = ttk.LabelFrame(master, text="YuNet Parameters", padding=8)
        self._yunet_frame.pack(fill="x", padx=6, pady=4)

        ttk.Label(self._yunet_frame, text="Score threshold (confidence):").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self._score_var = tk.StringVar(master, value="0.60")
        ttk.Entry(self._yunet_frame, textvariable=self._score_var, width=8).grid(
            row=0, column=1, sticky="w", padx=6, pady=2
        )

        ttk.Label(self._yunet_frame, text="NMS threshold (non-maxima):").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self._nms_var = tk.StringVar(master, value="0.30")
        ttk.Entry(self._yunet_frame, textvariable=self._nms_var, width=8).grid(
            row=1, column=1, sticky="w", padx=6, pady=2
        )

        ttk.Label(self._yunet_frame, text="Max candidates (top K):").grid(
            row=2, column=0, sticky="w", pady=2
        )
        self._topk_var = tk.StringVar(master, value="5000")
        ttk.Entry(self._yunet_frame, textvariable=self._topk_var, width=8).grid(
            row=2, column=1, sticky="w", padx=6, pady=2
        )

        # 3. Viola-Jones Cascade Parameters Frame
        self._vj_frame = ttk.LabelFrame(master, text="Viola-Jones Cascade Parameters", padding=8)
        self._vj_frame.pack(fill="x", padx=6, pady=4)

        self._cascade_mode_var = tk.StringVar(master, value="pretrained")
        ttk.Radiobutton(
            self._vj_frame,
            text="Use pre-trained cascade (models/viola_jones_cascade.json)",
            value="pretrained",
            variable=self._cascade_mode_var,
            command=self._on_cascade_mode_changed,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Radiobutton(
            self._vj_frame,
            text="Train custom cascade on data/faces_non_faces",
            value="train_new",
            variable=self._cascade_mode_var,
            command=self._on_cascade_mode_changed,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(self._vj_frame, text="Classifiers per stage (e.g. 2, 5, 10):").grid(
            row=2, column=0, sticky="w", padx=18, pady=2
        )
        self._stages_var = tk.StringVar(master, value="2, 5, 10")
        self._stages_entry = ttk.Entry(self._vj_frame, textvariable=self._stages_var, width=12)
        self._stages_entry.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(self._vj_frame, text="Train samples per class:").grid(
            row=3, column=0, sticky="w", padx=18, pady=2
        )
        self._samples_var = tk.StringVar(master, value="250")
        self._samples_entry = ttk.Entry(self._vj_frame, textvariable=self._samples_var, width=12)
        self._samples_entry.grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(self._vj_frame, text="Haar features to display:").grid(
            row=4, column=0, sticky="w", padx=18, pady=2
        )
        self._max_features_var = tk.StringVar(master, value="10")
        self._max_features_entry = ttk.Entry(
            self._vj_frame, textvariable=self._max_features_var, width=12
        )
        self._max_features_entry.grid(row=4, column=1, sticky="w", pady=2)

        # 4. Source Selection Frame
        source_frame = ttk.LabelFrame(master, text="Input Source", padding=8)
        source_frame.pack(fill="x", padx=6, pady=4)

        self._source_var = tk.StringVar(master, value=self.selected_source)

        if self._has_current_image:
            ttk.Radiobutton(
                source_frame,
                text="Current working image",
                value="current",
                variable=self._source_var,
            ).pack(anchor="w", pady=2)

        ttk.Radiobutton(
            source_frame,
            text="Open image from folder...",
            value="file",
            variable=self._source_var,
        ).pack(anchor="w", pady=2)

        self._rb_webcam = ttk.Radiobutton(
            source_frame,
            text="Real-time webcam stream (YuNet)",
            value="webcam",
            variable=self._source_var,
        )
        self._rb_webcam.pack(anchor="w", pady=2)

        self._webcam_note = ttk.Label(
            source_frame, text="", foreground="#666666", font="TkSmallCaptionFont"
        )
        self._webcam_note.pack(anchor="w", pady=(1, 2))

        # Checkbox to protect undo/redo history
        self._preserve_var = tk.BooleanVar(master, value=self.preserve_history)
        if self._has_current_image:
            ttk.Checkbutton(
                source_frame,
                text="Preview result without modifying working image (preserves undo/redo)",
                variable=self._preserve_var,
            ).pack(anchor="w", pady=(4, 0))

        self._on_method_changed()
        self._on_cascade_mode_changed()
        return method_frame

    def _on_method_changed(self) -> None:
        """Enable the widgets of the selected method and disable the others."""
        is_haar = self._method_var.get() == "haar"
        if is_haar:
            self._rb_webcam.configure(state="disabled")
            self._webcam_note.configure(
                text="(Webcam discarded for Haar: pure-Python cascade is too slow for 30fps real-time)."
            )
            if self._source_var.get() == "webcam":
                self._source_var.set("current" if self._has_current_image else "file")
            # Dim YuNet frame, highlight Viola-Jones frame
            for child in self._yunet_frame.winfo_children():
                try:
                    child.configure(state="disabled")
                except Exception:
                    pass
            for child in self._vj_frame.winfo_children():
                try:
                    child.configure(state="normal")
                except Exception:
                    pass
            self._on_cascade_mode_changed()
        else:
            self._rb_webcam.configure(state="normal")
            self._webcam_note.configure(text="")
            for child in self._yunet_frame.winfo_children():
                try:
                    child.configure(state="normal")
                except Exception:
                    pass
            for child in self._vj_frame.winfo_children():
                try:
                    child.configure(state="disabled")
                except Exception:
                    pass

    def _on_cascade_mode_changed(self) -> None:
        """Enable the training fields only when training a custom cascade."""
        if self._method_var.get() == "haar":
            is_new = self._cascade_mode_var.get() == "train_new"
            self._stages_entry.configure(state="normal" if is_new else "disabled")
            self._samples_entry.configure(state="normal" if is_new else "disabled")

    def validate(self) -> bool:
        """Check the entered parameters; show an error and return False if any is invalid."""
        try:
            st = float(self._score_var.get().strip())
            nms = float(self._nms_var.get().strip())
            tk_val = int(self._topk_var.get().strip())
            if not (0.01 <= st <= 1.0):
                raise ValueError("Score threshold must be between 0.01 and 1.0.")
            if not (0.01 <= nms <= 1.0):
                raise ValueError("NMS threshold must be between 0.01 and 1.0.")
            if tk_val < 1:
                raise ValueError("Top K must be at least 1.")

            self.score_threshold = st
            self.nms_threshold = nms
            self.top_k = tk_val

            if self._method_var.get() == "haar":
                max_f = int(self._max_features_var.get().strip())
                if max_f < 1:
                    raise ValueError("Haar features to display must be at least 1.")
                self.max_haar_features = max_f

                if self._cascade_mode_var.get() == "train_new":
                    stages_text = self._stages_var.get().strip()
                    stages = [int(s.strip()) for s in stages_text.split(",") if s.strip()]
                    if not stages:
                        raise ValueError("Specify at least one stage (e.g. 2, 5, 10).")
                    if any(count < 1 for count in stages):
                        raise ValueError("Each stage needs at least one classifier (e.g. 2, 5, 10).")
                    self.cascade_stages_str = stages_text
                    self.train_samples = max(20, int(self._samples_var.get().strip()))

            return True
        except ValueError as err:
            messagebox.showerror("Invalid Parameter", str(err), parent=self)
            return False

    def apply(self) -> None:
        """Store the confirmed choices on the dialog."""
        self.selected_method = self._method_var.get()
        self.selected_source = self._source_var.get()
        self.cascade_mode = self._cascade_mode_var.get()
        self.preserve_history = bool(self._preserve_var.get())
        if hasattr(self, "_max_features_var"):
            try:
                self.max_haar_features = int(self._max_features_var.get().strip())
            except ValueError:
                self.max_haar_features = 10
        self.user_confirmed = True


class FaceDetectionTool(ForensicsTool):
    """Face detection tool supporting both YuNet and Viola-Jones Cascade."""

    tool_id = "face_detection"
    title = "Face Detection"
    category = "Face Analysis"
    description = "Detect faces with YuNet (landmarks) or Viola-Jones Haar Cascade."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Ask for options, run the chosen detector, and return the annotated result."""
        dialog = FaceDetectionDialog(parent, has_current_image=document.is_loaded)
        if not dialog.user_confirmed:
            return None

        method = dialog.selected_method
        source = dialog.selected_source
        preserve_history = dialog.preserve_history

        # ---------------------------------------------------------
        # Case A: Webcam stream (YuNet)
        # ---------------------------------------------------------
        if source == "webcam":
            webcam_dlg = WebcamCaptureDialog(
                parent,
                model_path=YUNET_MODEL_PATH,
                score_threshold=dialog.score_threshold,
                nms_threshold=dialog.nms_threshold,
                top_k=dialog.top_k,
            )
            captured = webcam_dlg.captured_image
            if captured is None:
                return None

            # If document was empty, initialize it with the captured frame
            if document.current is None:
                document.original = captured.copy()
                document.current = captured.copy()
                preview_only = False
            else:
                # If document is loaded, respect user's choice to preserve undo/redo history
                preview_only = preserve_history

            num_faces = webcam_dlg.faces_detected
            details: dict[str, Any] = {
                "Method": "YuNet (Webcam Live)",
                "Faces Detected": num_faces,
                "Resolution": f"{captured.width} × {captured.height}",
                "Score Threshold": f"{dialog.score_threshold:.2f}",
                "NMS Threshold": f"{dialog.nms_threshold:.2f}",
            }
            for i, f in enumerate(webcam_dlg.detected_face_list):
                face_num = i + 1
                details[f"Face #{face_num}"] = f"Box: {f['box']} | Conf: {f['confidence']:.2%}"
                for lm_name, pt in f["landmarks"].items():
                    details[f"  Face #{face_num} - {lm_name}"] = f"({pt[0]}, {pt[1]})"

            return ToolResult(
                message=f"Captured webcam frame with {num_faces} detected face(s).",
                image=captured,
                details=details,
                preview_only=preview_only,
            )

        # ---------------------------------------------------------
        # Case B: Load image from disk or use current working document
        # ---------------------------------------------------------
        if source == "file":
            loaded = load_image_file(parent, title="Open Image for Face Detection")
            if loaded is None:
                return None
            source_path, input_image = loaded
            if document.current is None:
                document.path = source_path
                document.original = input_image.copy()
                document.current = input_image.copy()
                preview_only = False
            else:
                preview_only = preserve_history
        else:
            if not document.is_loaded or document.current is None:
                messagebox.showinfo("No image", "Open an image first.", parent=parent)
                return None
            input_image = document.current
            preview_only = preserve_history

        # ---------------------------------------------------------
        # Method 1: YuNet Detection
        # ---------------------------------------------------------
        if method == "yunet":
            rgb_np = np.array(input_image.convert("RGB"))
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

            annotated_bgr, faces = detect_faces_yunet(
                bgr_np,
                score_threshold=dialog.score_threshold,
                nms_threshold=dialog.nms_threshold,
                top_k=dialog.top_k,
            )
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(annotated_rgb)

            num_faces = len(faces)
            details = {
                "Method": "YuNet (Deep Learning ONNX)",
                "Faces Detected": num_faces,
                "Model": YUNET_MODEL_PATH.name,
                "Score Threshold": f"{dialog.score_threshold:.2f}",
                "NMS Threshold": f"{dialog.nms_threshold:.2f}",
                "Top K Candidates": dialog.top_k,
            }
            # List every face and all 5 landmarks with unique keys
            for i, f in enumerate(faces):
                conf = f["confidence"]
                box = f["box"]
                face_num = i + 1
                details[f"Face #{face_num}"] = f"BBox (x, y, w, h): {box} | Conf: {conf:.2%}"
                for lm_name, pt in f["landmarks"].items():
                    details[f"  Face #{face_num} - {lm_name}"] = f"({pt[0]}, {pt[1]})"

            msg = (
                f"YuNet detected {num_faces} face(s) with proportional landmarks."
                if num_faces > 0
                else "YuNet: No faces detected with current thresholds."
            )
            return ToolResult(
                message=msg,
                image=result_img,
                details=details,
                preview_only=preview_only,
            )

        # ---------------------------------------------------------
        # Method 2: Viola-Jones Attentional Cascade
        # ---------------------------------------------------------
        # Training is slow, so it runs on a worker thread behind a progress window.
        try:
            if dialog.cascade_mode == "train_new":
                stage_list = [int(s.strip()) for s in dialog.cascade_stages_str.split(",") if s.strip()]
                cascade = run_with_progress(
                    parent,
                    "Training Viola-Jones cascade",
                    lambda report, cancelled: train_viola_jones_cascade(
                        stage_configs=stage_list,
                        max_pos=dialog.train_samples,
                        max_neg=dialog.train_samples * 2,
                        progress=report,
                        should_cancel=cancelled,
                    ),
                )
            else:
                cascade = load_cascade_model(DEFAULT_CASCADE_PATH)
                if cascade is None or not cascade.stages:
                    cascade = run_with_progress(
                        parent,
                        "Training default Viola-Jones cascade",
                        lambda report, cancelled: ensure_default_cascade_model(
                            DEFAULT_CASCADE_PATH, progress=report, should_cancel=cancelled
                        ),
                    )
        except TaskCancelled:
            return None

        all_classifiers = cascade.get_all_classifiers()
        total_stages = len(cascade.stages)

        # Propose candidate face regions via YuNet
        rgb_arr = np.array(input_image.convert("RGB"))
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        _, candidate_faces = detect_faces_yunet(bgr_arr, score_threshold=0.4)

        annotated_image = input_image.copy()
        details: dict[str, Any] = {
            "Method": "Viola-Jones Attentional Cascade",
            "Dataset Used": "data/faces_non_faces",
            "Canonical Window Size": f"{CANONICAL_PATCH_SIZE[1]} × {CANONICAL_PATCH_SIZE[0]} px",
            "Total Cascade Stages": total_stages,
            "Total Classifiers": len(all_classifiers),
            "Haar Features Visualized": f"{dialog.max_haar_features} unique (max)",
        }

        if candidate_faces:
            confirmed_count = 0
            for idx, cand in enumerate(candidate_faces):
                face_num = idx + 1
                fx, fy, fw, fh = cand["box"]
                face_crop = input_image.crop((fx, fy, fx + fw, fy + fh))

                # Canonical 19x19 evaluation
                int_norm = patch_to_normalized_integral(face_crop)

                is_face, stages_passed, stage_scores = cascade.predict(int_norm)
                scores_str = ", ".join([f"S{s+1}: {sc:+.2f}" for s, sc in enumerate(stage_scores)])

                if is_face:
                    confirmed_count += 1
                    active = activated_classifiers(int_norm, all_classifiers)
                    status_lbl = (
                        f"CONFIRMED (Passed {stages_passed}/{total_stages} stages, "
                        f"{len(active)}/{len(all_classifiers)} features active)"
                    )
                    # ONLY superimpose the Haar features that fired on this face, on CONFIRMED faces
                    annotated_image = draw_haar_overlay(
                        image=annotated_image,
                        classifiers=active,
                        alpha=0.5,
                        patch_size=CANONICAL_PATCH_SIZE,
                        target_box=(fx, fy, fw, fh),
                        max_features=dialog.max_haar_features,
                        draw_box=True,
                        box_label=f"Face {face_num}: VJ Confirmed",
                    )
                else:
                    status_lbl = f"REJECTED at Stage {stages_passed + 1} ({scores_str})"

                details[f"Candidate #{face_num}"] = f"Box: {(fx, fy, fw, fh)} | {status_lbl}"

            details["Viola-Jones Confirmed"] = f"{confirmed_count} of {len(candidate_faces)} candidate(s)"

            if confirmed_count > 0:
                verdict = f"Confirmed {confirmed_count} face(s)"
                msg_text = (
                    f"Viola-Jones Cascade Result: {verdict}!\n\n"
                    f"{confirmed_count} of {len(candidate_faces)} candidate(s) passed all cascade stages.\n"
                    f"Haar features ({dialog.max_haar_features} unique max) and bounding boxes have been rendered on confirmed faces."
                )
                status_msg = f"Viola-Jones: Confirmed {confirmed_count}/{len(candidate_faces)} face(s) with Haar features superimposed."
            else:
                verdict = "All Candidates Rejected"
                msg_text = (
                    f"Viola-Jones Cascade Result: {verdict}.\n\n"
                    f"None of the {len(candidate_faces)} candidate regions passed all cascade stages.\n"
                    f"No Haar features or boxes were superimposed."
                )
                status_msg = f"Viola-Jones: 0 of {len(candidate_faces)} candidate(s) confirmed. No features superimposed."

        else:
            # Standalone image (e.g. 19x19 crop from data/faces_non_faces or user image without detected bbox)
            int_norm = patch_to_normalized_integral(input_image)

            is_face, stages_passed, stage_scores = cascade.predict(int_norm)
            scores_str = ", ".join([f"S{s+1}: {sc:+.2f}" for s, sc in enumerate(stage_scores)])

            if is_face:
                # ONLY superimpose the Haar features that fired, and only if confirmed as a face
                active = activated_classifiers(int_norm, all_classifiers)
                details["Active Features"] = f"{len(active)} / {len(all_classifiers)}"
                annotated_image = draw_haar_overlay(
                    image=input_image,
                    classifiers=active,
                    alpha=0.5,
                    patch_size=CANONICAL_PATCH_SIZE,
                    target_box=None,
                    max_features=dialog.max_haar_features,
                    draw_box=True,
                    box_label="Face 1: VJ Confirmed",
                )
                verdict = "Face Detected"
                details["Verdict"] = f"Face Detected (Passed all {total_stages} stages)"
                msg_text = (
                    f"Viola-Jones Cascade Result: Face Detected!\n\n"
                    f"Passed all {total_stages} stages ({scores_str}).\n"
                    f"Haar features have been superimposed with alpha = 0.5."
                )
                status_msg = f"Viola-Jones: Face Detected! Passed all {total_stages} stages ({scores_str})."
            else:
                # DO NOT superimpose features on rejected non-face images
                verdict = "Rejected as Non-Face"
                details["Verdict"] = f"Rejected at Stage {stages_passed + 1} ({scores_str})"
                msg_text = (
                    f"Viola-Jones Cascade Result: Rejected as Non-Face.\n\n"
                    f"Rejected at Stage {stages_passed + 1} of {total_stages} ({scores_str}).\n"
                    f"No Haar features were superimposed."
                )
                status_msg = f"Viola-Jones: Rejected as Non-Face at Stage {stages_passed + 1}. No features superimposed."

            details["Stages Passed"] = f"{stages_passed} / {total_stages}"
            details["Stage Scores"] = scores_str

        messagebox.showinfo("Viola-Jones Cascade Result", msg_text, parent=parent)

        return ToolResult(
            message=status_msg,
            image=annotated_image,
            details=details,
            preview_only=preview_only,
        )
