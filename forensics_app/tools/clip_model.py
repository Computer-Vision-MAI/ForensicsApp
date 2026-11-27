"""Set 10: CLIP zero-shot classification and text → image search.

CLIP (OpenAI, 2021) has two encoders trained together on 400 M image-caption pairs:

* an **image encoder** (a ViT: the 224 × 224 image is cut into 32 × 32 or 14 × 14 patches,
  and the output of the class token is the image's representation);
* a **text encoder** (a 12-layer transformer over at most 77 tokens; the output at the
  end-of-text token is the sentence's representation).

A linear **projection** after each encoder maps both into the same 512-D space (768-D for
ViT-L/14). Training (contrastive, InfoNCE) pulls each image towards its own caption and
pushes it away from the other captions of the batch, so after normalising the vectors,
the **cosine similarity** says how well a text describes an image.

Zero-shot classification needs no training at all:

1. write each class as a sentence with a template: ``"a photo of a {}."`` → ``"a photo of a cat."``;
2. encode the image and every sentence, and normalise them (length 1);
3. logit_c = s · cos(image, text_c), with the learnt temperature s = exp(logit_scale) = 100;
4. softmax over the classes → probabilities that always sum to 1.

**Prompt ensembling** (CLIP paper): encode the class with several templates, average the
normalised text vectors and normalise again; that single vector represents the class.

The model is loaded with Hugging Face ``transformers`` (optional, ``requirements-clip.txt``)
and downloaded once to ``models/clip`` (git-ignored). Only ``load_encoder`` needs it: the
maths below works on NumPy vectors, so it is tested with a fake encoder.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib.util
from pathlib import Path
import threading
import time
from typing import Any, Protocol

import numpy as np
from PIL import Image

from forensics_app.paths import MODELS_DIR

CLIP_DIR = MODELS_DIR / "clip"
CLIP_MODELS = {
    "openai/clip-vit-base-patch32": "ViT-B/32 (notebook, 151 M parameters, ~600 MB)",
    "openai/clip-vit-base-patch16": "ViT-B/16 (finer 16 px patches, ~600 MB)",
    "openai/clip-vit-large-patch14": "ViT-L/14 (Exercise 2: bigger, slower, better, ~1.7 GB)",
}
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TEMPLATE = "a photo of a {}."
# The five templates of the notebook's prompt ensemble
ENSEMBLE_TEMPLATES = ("a photo of a {}.", "a close-up photo of a {}.", "a good photo of a {}.",
                      "a cropped photo of a {}.", "a picture of a {}.")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
INSTALL_HINT = (
    "CLIP needs PyTorch and Hugging Face transformers, which are not installed. Install them with:  "
    ".venv/bin/pip install -r requirements-clip.txt"
)
BATCH = 32
MAX_GALLERY = 2000

Report = Callable[[int, int, str], None]


def clip_available() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


def require_clip() -> None:
    if not clip_available():
        raise ValueError(INSTALL_HINT)


def is_downloaded(name: str) -> bool:
    """True when the weights are already in models/clip (no download needed)."""
    folder = CLIP_DIR / f"models--{name.replace('/', '--')}" / "snapshots"
    return folder.is_dir() and any(folder.iterdir())


class Encoder(Protocol):
    name: str
    logit_scale: float

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


# --------------------------------------------------------------------------- the real model


def clip_view(image: Image.Image, size: int = 224) -> Image.Image:
    """What CLIP's preprocessing feeds the network: shorter side to ``size`` (bicubic), centre crop."""
    image = image.convert("RGB")
    scale = size / min(image.size)
    resized = image.resize((max(size, round(image.width * scale)), max(size, round(image.height * scale))),
                           Image.Resampling.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


class ClipEncoder:
    """CLIPModel + CLIPProcessor; the two encoders and projections are called explicitly."""

    def __init__(self, name: str, device: str) -> None:
        require_clip()
        import torch
        from transformers import CLIPModel, CLIPProcessor

        CLIP_DIR.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.device = device
        self.model = CLIPModel.from_pretrained(name, cache_dir=CLIP_DIR).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(name, cache_dir=CLIP_DIR)
        self._torch = torch
        self.logit_scale = float(self.model.logit_scale.exp().item())

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for start in range(0, len(images), BATCH):
                batch = [im.convert("RGB") for im in images[start:start + BATCH]]
                pixels = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device)
                pooled = self.model.vision_model(pixel_values=pixels).pooler_output  # class token, after LayerNorm
                emb = self.model.visual_projection(pooled)  # linear map into the shared space
                out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), BATCH):
                tokens = self.processor.tokenizer(list(texts[start:start + BATCH]), padding=True, truncation=True,
                                                  return_tensors="pt").to(self.device)
                pooled = self.model.text_model(input_ids=tokens["input_ids"],
                                               attention_mask=tokens["attention_mask"]).pooler_output  # [EOS] token
                emb = self.model.text_projection(pooled)
                out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        """Tokens per sentence, including the start and end tokens (CLIP's limit is 77)."""
        return [len(ids) for ids in self.processor.tokenizer(list(texts))["input_ids"]]

    def info(self) -> dict[str, Any]:
        config = self.model.config
        vision, text = config.vision_config, config.text_config
        params = sum(p.numel() for p in self.model.parameters())
        vision_params = sum(p.numel() for p in self.model.vision_model.parameters())
        text_params = sum(p.numel() for p in self.model.text_model.parameters())
        patches = (vision.image_size // vision.patch_size) ** 2
        return {
            "Model": self.name,
            "Parameters": f"{params / 1e6:.0f} M (image encoder {vision_params / 1e6:.0f} M, text encoder "
                          f"{text_params / 1e6:.0f} M)",
            "Image encoder": f"ViT, {vision.num_hidden_layers} layers, width {vision.hidden_size}, "
                             f"{vision.image_size}×{vision.image_size} input cut into {patches} patches of "
                             f"{vision.patch_size}×{vision.patch_size} px + 1 class token",
            "Text encoder": f"transformer, {text.num_hidden_layers} layers, width {text.hidden_size}, "
                            f"at most {text.max_position_embeddings} tokens (BPE)",
            "Shared embedding": f"{config.projection_dim} dimensions (linear projection of each encoder)",
            "Temperature": f"exp(logit_scale) = {self.logit_scale:.1f}",
            "Preprocessing": f"shorter side → {vision.image_size} px (bicubic), centre crop "
                             f"{vision.image_size}×{vision.image_size}, normalise with CLIP mean/std",
        }


_CACHE: dict[tuple[str, str], ClipEncoder] = {}
_LOCK = threading.Lock()


def load_encoder(name: str = DEFAULT_MODEL, device: str | None = None) -> ClipEncoder:
    """The CLIP model (downloaded to models/clip the first time), cached in memory."""
    require_clip()
    from .unet_model import default_device

    device = device or default_device()
    key = (name, device)
    with _LOCK:
        if key not in _CACHE:
            if len(_CACHE) >= 2:  # at most two models in memory (ViT-L/14 alone is 1.7 GB)
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = ClipEncoder(name, device)
        return _CACHE[key]


# --------------------------------------------------------------------------- zero-shot (NumPy)


def parse_lines(text: str) -> list[str]:
    """One entry per non-empty line, without duplicates (the order is kept)."""
    seen: dict[str, None] = {}
    for line in text.splitlines():
        line = line.strip()
        if line:
            seen.setdefault(line, None)
    return list(seen)


def check_templates(templates: Sequence[str]) -> str | None:
    """A problem with the templates, or None. Each one needs exactly one '{}' for the class."""
    if not templates:
        return "Write at least one template, e.g. 'a photo of a {}.'"
    for t in templates:
        if t.count("{}") != 1:
            return f"The template '{t}' must contain '{{}}' exactly once (where the class text goes)."
    return None


def build_prompts(labels: Sequence[str], template: str) -> list[str]:
    return [template.replace("{}", label) for label in labels]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def class_embeddings(encoder: Encoder, labels: Sequence[str], templates: Sequence[str]) -> tuple[np.ndarray, list[list[str]]]:
    """One normalised text vector per class: the mean over the templates, normalised again."""
    prompts = [build_prompts(labels, t) for t in templates]  # [template][class]
    flat = [p for per_template in prompts for p in per_template]
    emb = normalize(encoder.embed_texts(flat)).reshape(len(templates), len(labels), -1)
    return normalize(emb.mean(axis=0)), [[prompts[t][c] for t in range(len(templates))] for c in range(len(labels))]


@dataclass
class ZeroShotResult:
    model: str
    labels: list[str]
    templates: list[str]  # one entry, or several averaged (prompt ensembling)
    cosine: np.ndarray  # (C,) cosine similarity between the image and each class
    logit_scale: float

    @property
    def logits(self) -> np.ndarray:
        return self.logit_scale * self.cosine

    @property
    def probs(self) -> np.ndarray:
        return softmax(self.logits)

    @property
    def best(self) -> int:
        return int(np.argmax(self.cosine))

    @property
    def title(self) -> str:
        if len(self.templates) == 1:
            return f'template: "{self.templates[0]}"'
        return f"ensemble of {len(self.templates)} templates"


def zero_shot(encoder: Encoder, image: Image.Image | np.ndarray, labels: Sequence[str],
              templates: Sequence[str]) -> ZeroShotResult:
    """Probabilities of each label for one image (``image`` may be an embedding already)."""
    if isinstance(image, Image.Image):
        image_emb = normalize(encoder.embed_images([image])[0])
    else:
        image_emb = normalize(np.asarray(image))
    text_emb, _ = class_embeddings(encoder, labels, templates)
    return ZeroShotResult(encoder.name, list(labels), list(templates), text_emb @ image_emb, encoder.logit_scale)


@dataclass
class ZeroShotRun:
    results: list[ZeroShotResult]
    image_embedding: dict[str, np.ndarray] = field(default_factory=dict)
    seconds: float = 0.0
    info: dict[str, dict[str, Any]] = field(default_factory=dict)
    token_counts: dict[str, list[int]] = field(default_factory=dict)


def run_zero_shot(encoders: Sequence[Encoder], image: Image.Image, labels: Sequence[str], templates: Sequence[str],
                  compare_templates: bool, report: Report | None = None) -> ZeroShotRun:
    """Ensemble (one result per model) or comparison (one per template, plus the ensemble)."""
    started = time.perf_counter()
    run = ZeroShotRun([])
    for k, encoder in enumerate(encoders):
        if report is not None:
            report(k, len(encoders), f"Encoding the image and {len(labels)} classes with {encoder.name}…")
        emb = normalize(encoder.embed_images([image])[0])
        run.image_embedding[encoder.name] = emb
        if compare_templates and len(templates) > 1:
            for t in templates:
                run.results.append(zero_shot(encoder, emb, labels, [t]))
        run.results.append(zero_shot(encoder, emb, labels, templates))
        if hasattr(encoder, "info"):
            run.info[encoder.name] = encoder.info()
        if hasattr(encoder, "token_counts"):
            run.token_counts[encoder.name] = encoder.token_counts(
                [p for t in templates for p in build_prompts(labels, t)])
    run.seconds = time.perf_counter() - started
    return run


# --------------------------------------------------------------------------- text → image search


SAMPLE_IMAGES = ("astronaut", "chelsea", "coffee", "rocket", "immunohistochemistry", "hubble_deep_field",
                 "retina", "coins", "camera", "moon", "page", "brick", "grass", "gravel", "colorwheel",
                 "cell", "clock", "stereo_motorcycle", "text")


def sample_image(name: str) -> Image.Image:
    """A scikit-image sample (bundled, works offline) as an RGB PIL image."""
    from skimage import data

    array = getattr(data, name)()
    if isinstance(array, tuple):  # stereo_motorcycle returns (left, right, disparity)
        array = array[0]
    return Image.fromarray(np.asarray(array)).convert("RGB")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gallery_files(folder: Path, limit: int = MAX_GALLERY) -> list[Path]:
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    return files[:limit]


@dataclass
class Gallery:
    names: list[str]
    embeddings: np.ndarray  # (N, d), normalised
    thumbnails: list[Image.Image]
    paths: list[Path | None]


_GALLERY_CACHE: dict[str, Gallery] = {}


def _gallery_key(model: str, sources: Sequence[Path | str]) -> str:
    digest = hashlib.sha1(model.encode())
    for s in sources:
        digest.update(str(s).encode())
        if isinstance(s, Path) and s.exists():
            digest.update(str(s.stat().st_mtime_ns).encode())
    return digest.hexdigest()


def build_gallery(encoder: Encoder, folder: Path | None, report: Report | None = None,
                  cancelled: Callable[[], bool] | None = None, limit: int = MAX_GALLERY) -> Gallery:
    """Embed every image of ``folder`` (or the scikit-image samples when None); cached per model."""
    sources: list[Path | str] = list(gallery_files(folder, limit)) if folder is not None else list(SAMPLE_IMAGES)
    if not sources:
        raise FileNotFoundError(f"There are no images in {folder}.")
    key = _gallery_key(encoder.name, sources)
    if key in _GALLERY_CACHE:
        return _GALLERY_CACHE[key]
    names, embeddings, thumbs, paths = [], [], [], []
    for start in range(0, len(sources), BATCH):
        if cancelled is not None and cancelled():
            from .dialogs import TaskCancelled

            raise TaskCancelled()
        if report is not None:
            report(start, len(sources), f"Encoding the gallery: {start}/{len(sources)} images…")
        batch_images = []
        for source in sources[start:start + BATCH]:
            if isinstance(source, Path):
                try:
                    with Image.open(source) as handle:
                        image = handle.convert("RGB")
                except OSError:
                    continue
                names.append(source.name)
                paths.append(source)
            else:
                image = sample_image(source)
                names.append(source)
                paths.append(None)
            batch_images.append(image)
            thumb = image.copy()
            thumb.thumbnail((256, 256))
            thumbs.append(thumb)
        if batch_images:
            embeddings.append(encoder.embed_images(batch_images))
    gallery = Gallery(names, normalize(np.concatenate(embeddings)), thumbs, paths)
    if len(_GALLERY_CACHE) > 4:
        _GALLERY_CACHE.pop(next(iter(_GALLERY_CACHE)))
    _GALLERY_CACHE[key] = gallery
    return gallery


def search(encoder: Encoder, gallery: Gallery, queries: Sequence[str]) -> np.ndarray:
    """Cosine similarity of every query (rows) with every gallery image (columns)."""
    return normalize(encoder.embed_texts(list(queries))) @ gallery.embeddings.T
