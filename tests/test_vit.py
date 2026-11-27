"""Set 10 extra: ViT data, patches, SPT, reports and figures (torch parts skipped without PyTorch)."""

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from forensics_app.tools import vit_data as vd
from forensics_app.tools import vit_model as vm
from forensics_app.tools import vit_tools as vt
from forensics_app.tools.unet_model import torch_available


def write_fake_cifar10(data_dir: Path, n_train_per_batch: int = 20, n_test: int = 30) -> None:
    folder = data_dir / vd.DATASETS["cifar10"].folder
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)

    def batch(n):
        labels = rng.integers(0, 10, n)
        # class k images are brighter for larger k, so a classifier can learn something
        data = np.clip(labels[:, None] * 25 + rng.integers(0, 20, (n, 3072)), 0, 255).astype(np.uint8)
        return {b"data": data, b"labels": labels.tolist()}

    for i in range(1, 6):
        with open(folder / f"data_batch_{i}", "wb") as handle:
            pickle.dump(batch(n_train_per_batch), handle, protocol=2)
    with open(folder / "test_batch", "wb") as handle:
        pickle.dump(batch(n_test), handle, protocol=2)
    with open(folder / "batches.meta", "wb") as handle:
        pickle.dump({b"label_names": [f"class_{k}".encode() for k in range(10)]}, handle, protocol=2)


class Evil:
    def __reduce__(self):
        import os

        return os.system, ("echo should-not-run",)


class VitDataTests(unittest.TestCase):
    def test_safe_unpickler_refuses_code(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil"
            path.write_bytes(pickle.dumps({b"data": Evil()}))
            with self.assertRaises(pickle.UnpicklingError):
                vd.safe_load(path)

    def test_load_fake_cifar(self) -> None:
        with TemporaryDirectory() as tmp:
            write_fake_cifar10(Path(tmp))
            with patch.object(vd, "DATA_DIR", Path(tmp)), patch.dict(vd._CACHE, clear=True):
                self.assertTrue(vd.is_downloaded("cifar10"))
                self.assertFalse(vd.is_downloaded("cifar100"))
                data = vd.load_cifar("cifar10")
        self.assertEqual(data.x_train.shape, (100, 32, 32, 3))
        self.assertEqual(data.x_test.shape, (30, 32, 32, 3))
        self.assertEqual(data.class_names[3], "class 3")

    def test_validation_split_is_the_last_part(self) -> None:
        train, val = vd.keras_validation_split(50000, 0.1)
        self.assertEqual((len(train), len(val)), (45000, 5000))
        self.assertEqual(val[0], 45000)

    def test_patches_and_shifts(self) -> None:
        image = np.arange(12 * 12 * 3, dtype=np.int64).reshape(12, 12, 3)
        patches = vd.extract_patches(image, 6)
        self.assertEqual(patches.shape, (2, 2, 6, 6, 3))
        np.testing.assert_array_equal(patches[1, 0], image[6:12, 0:6])
        shifted = vd.shift_image(image, -3, -3)  # left-up: content moves up-left
        np.testing.assert_array_equal(shifted[0, 0], image[3, 3])
        self.assertTrue(np.all(shifted[-3:] == 0))
        names = [name for name, _ in vd.shifted_images(image, 6)]
        self.assertEqual(names, ["left-up", "left-down", "right-up", "right-down"])
        self.assertEqual(vd.tokens_per_image(72, 6, False), (144, 108))
        self.assertEqual(vd.tokens_per_image(72, 6, True), (144, 540))


class VitReportTests(unittest.TestCase):
    def test_classification_report_matches_sklearn(self) -> None:
        from sklearn.metrics import f1_score, precision_score, recall_score

        rng = np.random.default_rng(3)
        y, p = rng.integers(0, 5, 200), rng.integers(0, 5, 200)
        report = vm.classification_report(y, p, 5)
        np.testing.assert_allclose(report["f1"], f1_score(y, p, average=None), atol=1e-4)
        np.testing.assert_allclose(report["precision"], precision_score(y, p, average=None), atol=1e-4)
        np.testing.assert_allclose(report["recall"], recall_score(y, p, average=None), atol=1e-4)
        self.assertAlmostEqual(report["macro_f1"], f1_score(y, p, average="macro"), places=3)
        self.assertEqual(np.array(report["confusion"]).sum(), 200)

    def test_top_k_and_rollout(self) -> None:
        logits = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.05]])
        self.assertEqual(vm.top_k_accuracy(logits, np.array([2, 0]), 1), 0.5)
        self.assertEqual(vm.top_k_accuracy(logits, np.array([2, 2]), 2), 0.0)
        self.assertEqual(vm.top_k_accuracy(logits, np.array([2, 2]), 3), 1.0)
        uniform = [np.full((4, 16, 16), 1 / 16) for _ in range(3)]
        rollout = vm.attention_rollout(uniform)
        self.assertEqual(rollout.shape, (4, 4))
        np.testing.assert_allclose(rollout, 1 / 16)

    def test_patch_figure_and_settings(self) -> None:
        image = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        result = vt.patches_result(image, "random", 72, 6, True)
        self.assertIn("540", result.details["SPT values per token"])
        self.assertGreater(result.image.width, 1000)
        self.assertEqual(vm.VitSettings(spt=True).default_name(), "vit_cifar100_spt_10e")
        self.assertIn("shifted patch", vm.architecture_summary(vm.VitSettings(spt=True)))
        image = vm.to_cifar_input(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (100, 60)))
        self.assertEqual(image.shape, (32, 32, 3))


@unittest.skipUnless(torch_available(), "PyTorch is not installed")
class VitTorchTests(unittest.TestCase):
    def test_patchify_and_shift_match_numpy(self) -> None:
        import torch

        from forensics_app.tools.vit_net import patchify, shift

        image = np.random.default_rng(1).random((12, 12, 3)).astype(np.float32)
        x = torch.from_numpy(image).permute(2, 0, 1)[None]
        tokens = patchify(x, 6)[0].numpy()
        expected = vd.extract_patches(image, 6).reshape(4, -1)
        np.testing.assert_allclose(tokens, expected)
        shifted = shift(x, 3, -3)[0].permute(1, 2, 0).numpy()
        np.testing.assert_allclose(shifted, vd.shift_image(image, 3, -3))

    def test_network_shapes(self) -> None:
        import torch

        from forensics_app.tools.vit_net import ViT, count_parameters

        mean, std = torch.zeros(3), torch.ones(3)
        for spt, lsa in ((False, False), (True, True)):
            net = ViT(10, mean, std, image_size=24, patch=6, dim=16, heads=2, layers=2, head_units=(32,),
                      spt=spt, lsa=lsa)
            self.assertEqual(tuple(net(torch.rand(3, 3, 32, 32) * 255).shape), (3, 10))
            net.eval()
            net.keep_attention(True)
            net(torch.rand(1, 3, 32, 32))
            maps = net.attention_maps()
            self.assertEqual(tuple(maps[0].shape), (1, 2, 16, 16))
            if lsa:  # a token never attends to itself
                self.assertTrue(torch.allclose(torch.diagonal(maps[0][0, 0]), torch.zeros(16)))
            self.assertGreater(count_parameters(net), 0)

    def test_train_and_classify(self) -> None:
        with TemporaryDirectory() as tmp:
            write_fake_cifar10(Path(tmp))
            with patch.object(vd, "DATA_DIR", Path(tmp)), patch.dict(vd._CACHE, clear=True):
                data = vd.load_cifar("cifar10")
            settings = vm.VitSettings(dataset="cifar10", epochs=2, batch=16, micro_batch=8, image_size=24, patch=6,
                                      dim=16, heads=2, layers=1, device="cpu", name="tiny")
            outcome = vm.train(settings, lambda *a: None, lambda: False, target_dir=Path(tmp) / "vit", data=data)
            passport = outcome.passport
            self.assertEqual(passport.data["dataset"]["validation"], 10)
            self.assertEqual(len(passport.test["predictions"]), 30)
            self.assertEqual(len(passport.history["val_accuracy"]), 2)
            self.assertEqual(vm.list_models(Path(tmp) / "vit")[0].name, "tiny")
            prediction = vm.classify(passport, data.x_test[0], device="cpu")
            self.assertAlmostEqual(float(prediction.probs.sum()), 1.0, places=4)
            self.assertEqual(prediction.rollout.shape, (4, 4))
            result = vt.classification_result(passport, data.x_test[0], "t", int(data.y_test[0]), prediction)
            self.assertIn("True class", result.details)
            self.assertGreater(vt.passport_result(passport).image.width, 1000)
            self.assertIn("Image by image", vt.comparison_result([passport, passport]).details)


if __name__ == "__main__":
    unittest.main()
