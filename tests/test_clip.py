"""Set 10: CLIP zero-shot maths, search and figures with a fake encoder (no PyTorch / transformers)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from forensics_app.tools import clip_model as cm
from forensics_app.tools.clip_render import render_search, render_zero_shot
from forensics_app.tools.clip_tools import PRESETS, ZeroShotSettings, parse_zero_shot, search_result, \
    SearchSettings, zero_shot_result


class FakeEncoder:
    """Images are encoded by their mean colour; a text by the colour words it contains."""

    name = "fake/clip"
    logit_scale = 100.0
    WORDS = {"red": 0, "green": 1, "blue": 2}

    def embed_images(self, images):
        out = [np.asarray(im.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0) + 1.0 for im in images]
        return cm.normalize(np.array(out))

    def embed_texts(self, texts):
        out = []
        for text in texts:
            v = np.full(3, 0.1)
            for word, k in self.WORDS.items():
                if word in text:
                    v[k] += 1.0
            if "photo" in text:  # templates change the vector a little
                v += 0.05
            out.append(v)
        return cm.normalize(np.array(out))


class ClipMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = FakeEncoder()
        self.red = Image.new("RGB", (40, 30), (230, 10, 10))

    def test_parsing_and_templates(self) -> None:
        self.assertEqual(cm.parse_lines(" a \n\n b\n a\n"), ["a", "b"])
        self.assertIsNone(cm.check_templates(["a photo of a {}.", "{}"]))
        self.assertIsNotNone(cm.check_templates(["no placeholder"]))
        self.assertIsNotNone(cm.check_templates(["{} and {}"]))
        self.assertIsNotNone(cm.check_templates([]))
        self.assertEqual(cm.build_prompts(["cat", "dog"], "a photo of a {}."), ["a photo of a cat.", "a photo of a dog."])
        labels, templates, problem = parse_zero_shot("cat", "a {}")
        self.assertIn("two classes", problem)
        for preset in PRESETS:
            self.assertIsNone(cm.check_templates(preset.templates), preset.label)
            self.assertGreaterEqual(len(preset.labels), 2)

    def test_softmax_and_zero_shot(self) -> None:
        probs = cm.softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(probs.sum(), 1.0)
        self.assertTrue(np.all(np.diff(probs) > 0))
        result = cm.zero_shot(self.encoder, self.red, ["red", "green", "blue"], ["a photo of a {} thing."])
        self.assertEqual(result.labels[result.best], "red")
        self.assertAlmostEqual(result.probs.sum(), 1.0)
        np.testing.assert_allclose(result.logits, 100 * result.cosine)
        self.assertIn("template", result.title)

    def test_ensemble_is_the_normalised_mean(self) -> None:
        templates = ["{}", "a photo of {}"]
        emb, prompts = cm.class_embeddings(self.encoder, ["red", "blue"], templates)
        self.assertEqual(prompts[0], ["red", "a photo of red"])
        manual = cm.normalize(self.encoder.embed_texts(["red", "a photo of red"]).mean(axis=0))
        np.testing.assert_allclose(emb[0], manual)
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0)

    def test_run_modes_and_result(self) -> None:
        run = cm.run_zero_shot([self.encoder], self.red, ["red", "blue"], ["{}", "a photo of {}"], True)
        self.assertEqual(len(run.results), 3)  # one per template + the ensemble
        run = cm.run_zero_shot([self.encoder, self.encoder], self.red, ["red", "blue"], ["{}"], False)
        self.assertEqual(len(run.results), 2)  # one per model
        settings = ZeroShotSettings(["fake/clip"], "sample", labels=["red", "green", "blue"],
                                    templates=["a {} thing"], show_cosine=True)
        result = zero_shot_result(self.red, "red square", settings, [self.encoder])
        self.assertIn("red", result.message)
        self.assertTrue(any("softmax sums to 1" in k for k in result.details))
        self.assertTrue(result.preview_only)

    def test_clip_view_and_samples(self) -> None:
        view = cm.clip_view(Image.new("RGB", (600, 400)))
        self.assertEqual(view.size, (224, 224))
        coffee = cm.sample_image("coffee")
        self.assertEqual(coffee.mode, "RGB")
        self.assertEqual(cm.sample_image("camera").mode, "RGB")  # grey sample becomes RGB

    def test_gallery_and_search(self) -> None:
        with TemporaryDirectory() as tmp:
            for name, colour in (("r.png", (250, 0, 0)), ("g.png", (0, 250, 0)), ("b.png", (0, 0, 250))):
                Image.new("RGB", (8, 8), colour).save(Path(tmp) / name)
            (Path(tmp) / "notes.txt").write_text("not an image")
            gallery = cm.build_gallery(self.encoder, Path(tmp))
            self.assertEqual(sorted(gallery.names), ["b.png", "g.png", "r.png"])
            sims = cm.search(self.encoder, gallery, ["something blue", "something green"])
            self.assertEqual(gallery.names[int(np.argmax(sims[0]))], "b.png")
            self.assertEqual(gallery.names[int(np.argmax(sims[1]))], "g.png")
            self.assertIs(cm.build_gallery(self.encoder, Path(tmp)), gallery)  # cached
            result = search_result(self.encoder, gallery, SearchSettings("fake/clip", Path(tmp), ["blue"], 2, True))
            self.assertIn("b.png", result.details['"blue"'].split(",")[0])
            self.assertGreater(render_search(gallery, ["blue"], sims[:1], 3, False).width, 500)

    def test_figure(self) -> None:
        results = cm.run_zero_shot([self.encoder], self.red, ["red", "green"], ["{}", "a photo of {}"], True).results
        self.assertGreater(render_zero_shot(self.red, results, True).width, 500)
        self.assertGreater(render_zero_shot(self.red, results[:1], False).width, 500)


if __name__ == "__main__":
    unittest.main()
