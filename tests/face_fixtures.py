"""Synthetic fixtures for the face analysis tests.

The real datasets (``data/``) and trained models (``models/``) are git-ignored, so
they are not available in CI. These helpers build small, deterministic stand-ins in
a temporary directory so the tests never download LFW or depend on local files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from forensics_app.tools.haar_adaboost import (
    CascadeStage,
    FeatureType,
    HaarLikeFeature,
    ViolaJonesCascade,
)

# Same layout as the LFW export produced by ``prepare_lfw_dataset`` (resize=0.4).
FACE_SHAPE = (50, 37)  # (H, W)
PEOPLE = ("Ariel Sharon", "Colin Powell", "George W Bush")
TRAIN_PER_PERSON = 40
TEST_PER_PERSON = 20
N_COMPONENTS = 30


def _person_prototype(index: int, rng: np.random.Generator) -> np.ndarray:
    """Return a distinct, smooth base pattern for one synthetic identity."""
    h, w = FACE_SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pattern = 128.0 + 60.0 * np.sin((index + 1) * xx / w * np.pi) * np.cos((index + 2) * yy / h * np.pi)
    return pattern + rng.normal(0.0, 5.0, size=FACE_SHAPE)


def build_lfw_like_dataset(data_dir: Path, seed: int = 0) -> None:
    """Write ``train/`` and ``test/`` PNGs named ``{split}_{idx:04d}_{Person_Name}.png``."""
    rng = np.random.default_rng(seed)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    counters = {"train": 0, "test": 0}
    for person_idx, name in enumerate(PEOPLE):
        prototype = _person_prototype(person_idx, rng)
        safe_name = name.replace(" ", "_")
        for split, folder, count in (
            ("train", train_dir, TRAIN_PER_PERSON),
            ("test", test_dir, TEST_PER_PERSON),
        ):
            for _ in range(count):
                sample = np.clip(prototype + rng.normal(0.0, 12.0, size=FACE_SHAPE), 0, 255)
                filename = f"{split}_{counters[split]:04d}_{safe_name}.png"
                Image.fromarray(sample.astype(np.uint8), mode="L").save(folder / filename)
                counters[split] += 1


def build_simple_cascade() -> ViolaJonesCascade:
    """Return a tiny two-stage cascade that fits the 19x19 canonical window."""
    feat1 = HaarLikeFeature(FeatureType.TWO_VERTICAL, (0, 0), 6, 6, threshold=0.0, weight=1.0)
    feat2 = HaarLikeFeature(FeatureType.TWO_HORIZONTAL, (0, 0), 6, 6, threshold=0.0, weight=1.0)
    return ViolaJonesCascade(
        stages=[
            CascadeStage(classifiers=[feat1], threshold=0.0),
            CascadeStage(classifiers=[feat2], threshold=0.0),
        ]
    )
