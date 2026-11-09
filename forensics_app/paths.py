"""Filesystem locations shared by the tools.

Datasets and trained models are resolved from the project root rather than the
process working directory, so the app finds (and writes) the same files no matter
where it is started from.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def display_path(path: Path) -> str:
    """Return ``path`` relative to the project root when possible, for short UI labels."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
