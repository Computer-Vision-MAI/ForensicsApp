"""Set 5.4: find the image in a folder that is closest to the current one.

The course asks at least for "return the image with the maximum number of matches".
That is what happens with RANSAC off. With RANSAC on (the default) the score is the
number of RANSAC inliers instead: matches that all agree with one geometric
transform. Random matches between unrelated images rarely agree with each other,
so this ranks the true picture first more often (19/19 vs 17/19 on Set5_images
with SIFT).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import TaskCancelled, run_with_progress
from .matching import (
    ImageFeatures,
    MatchResult,
    MatchSettings,
    compare_features,
    detect_features,
    render_matches,
    summarize_match,
)
from .matching_dialog import ask_match_settings

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"})
RANKING_ROWS = 10  # how many candidates are listed in the Results panel


@dataclass(frozen=True)
class Candidate:
    """One image of the folder and how well it matched the reference."""

    path: Path
    result: MatchResult | None = None
    keypoints: int = 0
    error: str | None = None

    @property
    def score(self) -> int:
        return self.result.score if self.result is not None else -1

    @property
    def sort_key(self) -> tuple[int, int]:
        # Score first; raw matches break ties (and order the unverified images)
        return (self.score, self.result.n_matches if self.result is not None else -1)


@dataclass(frozen=True)
class LookupResult:
    ranking: list[Candidate]  # best first; unreadable files at the end
    reference: ImageFeatures
    best_features: ImageFeatures | None

    @property
    def best(self) -> Candidate | None:
        if self.ranking and self.ranking[0].error is None:
            return self.ranking[0]
        return None

    @property
    def skipped(self) -> list[Candidate]:
        return [candidate for candidate in self.ranking if candidate.error is not None]


def list_image_files(
    folder: str | Path, recursive: bool = False, exclude: Iterable[str | Path] = ()
) -> list[Path]:
    """Return the image files of ``folder`` sorted by path.

    Hidden files (``.DS_Store``, macOS ``._*`` resource forks) are ignored, and so
    are the paths in ``exclude`` (e.g. the reference image itself).
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"'{root}' is not a folder.")
    excluded = {Path(path).resolve() for path in exclude}
    pattern = root.rglob("*") if recursive else root.glob("*")
    files = [
        path
        for path in pattern
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and path.resolve() not in excluded
    ]
    return sorted(files)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.copy()


def rank_candidates(
    reference: ImageFeatures,
    paths: Iterable[Path],
    settings: MatchSettings,
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[list[Candidate], ImageFeatures | None]:
    """Match ``reference`` against every image in ``paths``.

    Returns the candidates best first (unreadable files, with their error, at the
    end) and the keypoints of the best one, needed to draw its matches. Only the
    leader's keypoints are kept, so a big folder does not fill the memory.
    Raises TaskCancelled as soon as ``cancelled()`` returns True.
    """
    paths = list(paths)
    candidates: list[Candidate] = []
    best: tuple[Candidate, ImageFeatures] | None = None
    for index, path in enumerate(paths):
        if cancelled is not None and cancelled():
            raise TaskCancelled()
        if progress is not None:
            progress(index, len(paths), f"Matching {path.name} ({index + 1}/{len(paths)})…")
        try:
            features = detect_features(load_image(path), settings)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            candidates.append(Candidate(path, error=str(error) or type(error).__name__))
            continue
        candidate = Candidate(path, compare_features(reference, features, settings), len(features))
        candidates.append(candidate)
        if best is None or candidate.sort_key > best[0].sort_key:
            best = (candidate, features)
    if progress is not None:
        progress(len(paths), len(paths), "Done.")
    # sorted() is stable, so equal scores keep the folder order, as the leader did
    ranking = sorted(candidates, key=lambda candidate: candidate.sort_key, reverse=True)
    return ranking, best[1] if best is not None else None


def find_closest_image(
    reference_image: Image.Image,
    paths: Iterable[Path],
    settings: MatchSettings,
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> LookupResult:
    """Rank ``paths`` by similarity to ``reference_image``."""
    if progress is not None:
        progress(0, 1, "Detecting keypoints on the current image…")
    reference = detect_features(reference_image, settings)
    if len(reference) == 0:
        raise ValueError("No keypoints were found in the current image, so nothing can be matched.")
    ranking, best_features = rank_candidates(reference, paths, settings, progress, cancelled)
    return LookupResult(ranking, reference, best_features)


def describe_candidate(candidate: Candidate) -> str:
    if candidate.error is not None:
        return f"skipped: {candidate.error}"
    return f"{summarize_match(candidate.result)} ({candidate.keypoints} keypoints)"


class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Find Closest Image in Folder"
    category = "Feature Detection"
    description = (
        "Matches the current image against every image in a folder and shows the "
        "closest one (most RANSAC inliers, or most matches without RANSAC)."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        folder = filedialog.askdirectory(title="Choose the folder to search", parent=parent, mustexist=True)
        if not folder:
            return None

        answer = ask_match_settings(parent, self.title, MatchSettings(), ask_recursive=True)
        if answer is None:
            return None
        settings, options = answer

        exclude = [document.path] if document.path is not None else []
        paths = list_image_files(folder, recursive=bool(options.get("recursive")), exclude=exclude)
        if not paths:
            raise ValueError(f"No images were found in '{folder}'.")

        reference_image = document.current

        def task(report, cancelled):
            return find_closest_image(reference_image, paths, settings, report, cancelled)

        try:
            lookup = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None
        return self._build_result(document, Path(folder), lookup, settings)

    def _build_result(
        self, document: ImageDocument, folder: Path, lookup: LookupResult, settings: MatchSettings
    ) -> ToolResult:
        reference_name = document.path.name if document.path is not None else "current image"
        details: dict[str, object] = {
            "Operation": "Closest Image Lookup",
            "Folder": str(folder),
            "Images compared": len(lookup.ranking) - len(lookup.skipped),
            "Reference keypoints": len(lookup.reference),
            "Score": "RANSAC inliers" if settings.use_ransac else "number of matches",
        }
        best = lookup.best
        if best is None:
            details["Skipped files"] = len(lookup.skipped)
            details.update(settings.describe())
            return ToolResult(message="None of the images in the folder could be read.", details=details)

        details["Best match"] = best.path.name
        details["Best score"] = best.score
        for position, candidate in enumerate(lookup.ranking[:RANKING_ROWS], start=1):
            details[f"#{position}"] = f"{candidate.path.name}: {describe_candidate(candidate)}"
        if lookup.skipped:
            details["Skipped files"] = ", ".join(candidate.path.name for candidate in lookup.skipped)
        details.update(settings.describe())

        title = f"Reference: {reference_name}  <--->  Best: {best.path.name}  ({summarize_match(best.result)})"
        image = render_matches(lookup.reference, lookup.best_features, best.result, title=title)
        if best.score == 0:
            message = f"No image matched reliably; the closest guess is {best.path.name}."
        else:
            message = f"Closest image: {best.path.name} ({summarize_match(best.result)})."
        return ToolResult(message=message, image=image, details=details)
