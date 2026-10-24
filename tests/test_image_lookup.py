"""Tests for Set 5.4: looking up the closest image in a folder."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import data, transform

from forensics_app.core import ImageDocument
from forensics_app.tools import ImageLookupTool
from forensics_app.tools.dialogs import TaskCancelled
from forensics_app.tools.image_lookup import (
    find_closest_image,
    list_image_files,
    rank_candidates,
)
from forensics_app.tools.matching import MatchSettings, detect_features

FAST_ORB = MatchSettings(method="orb", n_keypoints=300, max_side=256)


def astronaut() -> Image.Image:
    return Image.fromarray(data.astronaut())


def rotated_astronaut() -> Image.Image:
    return astronaut().rotate(15, resample=Image.Resampling.BILINEAR)


class FolderTestCase(unittest.TestCase):
    """Creates a temporary folder with one related image and several unrelated ones."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        rotated_astronaut().save(self.folder / "astronaut_rotated.jpg")
        Image.fromarray(data.camera()).save(self.folder / "camera.png")
        Image.fromarray(data.coins()).save(self.folder / "coins.png")
        (self.folder / "broken.png").write_bytes(b"this is not a png")
        (self.folder / "notes.txt").write_text("not an image")
        (self.folder / ".hidden.png").write_bytes(b"")
        (self.folder / "._camera.png").write_bytes(b"macOS resource fork")
        sub = self.folder / "sub"
        sub.mkdir()
        Image.fromarray(data.chelsea()).save(sub / "cat.png")

    def tearDown(self) -> None:
        self._tmp.cleanup()


class ListImageFilesTests(FolderTestCase):
    def test_lists_only_visible_images(self) -> None:
        names = [path.name for path in list_image_files(self.folder)]
        self.assertEqual(names, ["astronaut_rotated.jpg", "broken.png", "camera.png", "coins.png"])

    def test_recursive_includes_subfolders(self) -> None:
        names = [path.name for path in list_image_files(self.folder, recursive=True)]
        self.assertIn("cat.png", names)
        self.assertEqual(len(names), 5)

    def test_skips_hidden_subfolders(self) -> None:
        hidden = self.folder / ".cache"
        hidden.mkdir()
        Image.new("L", (10, 10)).save(hidden / "thumb.png")
        names = [path.name for path in list_image_files(self.folder, recursive=True)]
        self.assertNotIn("thumb.png", names)

    def test_excludes_given_paths(self) -> None:
        names = [path.name for path in list_image_files(self.folder, exclude=[self.folder / "camera.png"])]
        self.assertNotIn("camera.png", names)

    def test_missing_folder_raises(self) -> None:
        with self.assertRaises(ValueError):
            list_image_files(self.folder / "nope")


class RankingTests(FolderTestCase):
    def test_the_related_image_ranks_first(self) -> None:
        paths = list_image_files(self.folder, recursive=True)
        lookup = find_closest_image(astronaut(), paths, FAST_ORB)
        self.assertEqual(lookup.best.path.name, "astronaut_rotated.jpg")
        self.assertGreater(lookup.best.score, lookup.ranking[1].score)
        self.assertEqual(len(lookup.best_features), lookup.best.keypoints)

    def test_unreadable_files_are_skipped_and_listed_last(self) -> None:
        lookup = find_closest_image(astronaut(), list_image_files(self.folder), FAST_ORB)
        self.assertEqual([c.path.name for c in lookup.skipped], ["broken.png"])
        self.assertEqual(lookup.ranking[-1].path.name, "broken.png")
        self.assertEqual(lookup.ranking[-1].score, -1)

    def test_baseline_without_ransac_ranks_by_matches(self) -> None:
        settings = MatchSettings(method="orb", n_keypoints=300, max_side=256, use_ransac=False)
        lookup = find_closest_image(astronaut(), list_image_files(self.folder), settings)
        self.assertEqual(lookup.best.path.name, "astronaut_rotated.jpg")
        self.assertEqual(lookup.best.score, lookup.best.result.n_matches)

    def test_ranking_is_sorted_by_score(self) -> None:
        reference = detect_features(astronaut(), FAST_ORB)
        ranking, _ = rank_candidates(reference, list_image_files(self.folder), FAST_ORB)
        keys = [candidate.sort_key for candidate in ranking]
        self.assertEqual(keys, sorted(keys, reverse=True))

    def test_reports_progress(self) -> None:
        reference = detect_features(astronaut(), FAST_ORB)
        paths = list_image_files(self.folder)
        reports = []
        rank_candidates(reference, paths, FAST_ORB, progress=lambda *args: reports.append(args))
        self.assertEqual(reports[0][:2], (0, len(paths)))
        self.assertEqual(reports[-1][:2], (len(paths), len(paths)))

    def test_cancelling_stops_the_search(self) -> None:
        reference = detect_features(astronaut(), FAST_ORB)
        with self.assertRaises(TaskCancelled):
            rank_candidates(reference, list_image_files(self.folder), FAST_ORB, cancelled=lambda: True)

    def test_reference_without_keypoints_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No keypoints"):
            find_closest_image(Image.new("L", (100, 100)), list_image_files(self.folder), FAST_ORB)

    def test_folder_with_only_broken_files_has_no_best(self) -> None:
        lookup = find_closest_image(astronaut(), [self.folder / "broken.png"], FAST_ORB)
        self.assertIsNone(lookup.best)
        self.assertIsNone(lookup.best_features)


class ImageLookupToolTests(FolderTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.document = ImageDocument()
        # The reference lives in the searched folder: it must not find itself
        reference_path = self.folder / "astronaut.png"
        astronaut().save(reference_path)
        self.document.load(reference_path)
        self.tool = ImageLookupTool()

    @patch("forensics_app.tools.image_lookup.filedialog.askdirectory", return_value="")
    def test_cancelling_the_folder_dialog_returns_none(self, _ask) -> None:
        self.assertIsNone(self.tool.run(None, self.document))

    @patch("forensics_app.tools.image_lookup.ask_match_settings", return_value=None)
    @patch("forensics_app.tools.image_lookup.filedialog.askdirectory")
    def test_cancelling_the_settings_returns_none(self, ask_dir, _ask) -> None:
        ask_dir.return_value = str(self.folder)
        self.assertIsNone(self.tool.run(None, self.document))

    @patch("forensics_app.tools.image_lookup.ask_match_settings")
    @patch("forensics_app.tools.image_lookup.filedialog.askdirectory")
    def test_finds_the_related_image_and_not_itself(self, ask_dir, ask) -> None:
        ask_dir.return_value = str(self.folder)
        ask.return_value = (FAST_ORB, {"recursive": True})
        result = self.tool.run(None, self.document)
        self.assertEqual(result.details["Best match"], "astronaut_rotated.jpg")
        self.assertIn("astronaut_rotated.jpg", result.message)
        self.assertTrue(result.details["#1"].startswith("astronaut_rotated.jpg: "))
        ranked = [value.split(":")[0] for key, value in result.details.items() if key.startswith("#")]
        self.assertNotIn("astronaut.png", ranked)
        self.assertEqual(result.details["Skipped files"], "broken.png")
        self.assertEqual(result.details["Images compared"], 4)
        self.assertEqual(result.image.mode, "RGB")
        self.assertFalse(result.preview_only)

    @patch("forensics_app.tools.image_lookup.ask_match_settings")
    @patch("forensics_app.tools.image_lookup.filedialog.askdirectory")
    def test_empty_folder_raises(self, ask_dir, ask) -> None:
        empty = self.folder / "empty"
        empty.mkdir()
        ask_dir.return_value = str(empty)
        ask.return_value = (FAST_ORB, {"recursive": False})
        with self.assertRaisesRegex(ValueError, "No images"):
            self.tool.run(None, self.document)

    @patch("forensics_app.tools.image_lookup.ask_match_settings")
    @patch("forensics_app.tools.image_lookup.filedialog.askdirectory")
    def test_only_unreadable_files_gives_a_message_without_image(self, ask_dir, ask) -> None:
        broken = self.folder / "broken_only"
        broken.mkdir()
        (broken / "a.jpg").write_bytes(b"nope")
        ask_dir.return_value = str(broken)
        ask.return_value = (FAST_ORB, {})
        result = self.tool.run(None, self.document)
        self.assertIsNone(result.image)
        self.assertIn("could be read", result.message)


if __name__ == "__main__":
    unittest.main()
