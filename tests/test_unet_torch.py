"""Set 11: the U-Net network, training and inference (skipped without PyTorch)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from forensics_app.tools import tray_dataset as td
from forensics_app.tools import unet_model as um

from test_segmentation import make_dataset


@unittest.skipUnless(um.torch_available(), "PyTorch is not installed")
class UnetTorchTests(unittest.TestCase):
    def test_network_matches_the_notebook(self) -> None:
        import torch

        from forensics_app.tools.unet_net import UNet, count_parameters, segmentation_loss

        net = UNet()
        self.assertEqual(count_parameters(net), 482_737)
        self.assertEqual(tuple(net(torch.zeros(1, 3, 32, 48)).shape), (1, 1, 32, 48))
        no_skip = UNet(skip=False)
        self.assertLess(count_parameters(no_skip), count_parameters(net))
        self.assertEqual(tuple(no_skip(torch.zeros(1, 3, 32, 48)).shape), (1, 1, 32, 48))
        logits = torch.zeros(2, 1, 8, 8)
        target = torch.ones(2, 1, 8, 8)
        bce = segmentation_loss(logits, target, "bce")
        self.assertAlmostEqual(bce.item(), float(np.log(2)), places=5)
        self.assertAlmostEqual(segmentation_loss(logits, target, "bce_dice").item(),
                               bce.item() + segmentation_loss(logits, target, "dice").item(), places=5)
        multi = segmentation_loss(torch.zeros(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long), "bce_dice")
        self.assertGreater(multi.item(), 0)
        with self.assertRaises(ValueError):
            segmentation_loss(logits, target, "focal")

    def test_train_save_load_and_segment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = make_dataset(Path(tmp) / "TrayDataset", n_train=14, n_test=2)
            for task in ("binary", "three"):
                settings = um.TrainSettings(root=root, task=task, size=(32, 48), epochs=2, batch=4, base=4,
                                            device="cpu", name=f"tiny_{task}")
                steps = []
                outcome = um.train(settings, lambda d, t, label: steps.append((d, t)), lambda: False,
                                   target_dir=Path(tmp) / "models")
                self.assertEqual(outcome.epochs_done, 2)
                self.assertEqual(steps[-1][0], steps[-1][1])
                passport = outcome.passport
                self.assertEqual(passport.name, f"tiny_{task}")
                self.assertEqual(len(passport.history["val_dice"]), 2)
                self.assertIn("test", passport.data["evaluation"])
                if task == "binary":
                    self.assertEqual(len(outcome.val.sweep_dice), len(um.SWEEP_THRESHOLDS))
                image = td.read_image(td.split_pairs(root, "test")[0][0])
                mask, confidence, small = um.segment_image(passport, image, device="cpu")
                self.assertEqual(mask.shape, (image.height, image.width))
                self.assertEqual(small.shape, (32, 48, 3))
                self.assertTrue(np.all((confidence >= 0) & (confidence <= 1)))
                evaluation, split, prediction = um.evaluate_passport(passport, "val", device="cpu")
                self.assertEqual(len(split), 2)
                self.assertEqual(prediction.masks.shape, (2, 32, 48))
            self.assertEqual(len(um.list_models(Path(tmp) / "models")), 2)

    def test_cancel_before_the_first_epoch(self) -> None:
        from forensics_app.tools.dialogs import TaskCancelled

        with TemporaryDirectory() as tmp:
            root = make_dataset(Path(tmp) / "TrayDataset", n_train=8, n_test=1)
            settings = um.TrainSettings(root=root, size=(32, 48), epochs=1, batch=4, base=4, device="cpu")
            with self.assertRaises(TaskCancelled):
                um.train(settings, lambda *a: None, lambda: True, target_dir=Path(tmp) / "models")
            self.assertFalse((Path(tmp) / "models" / settings.default_name()).exists())


if __name__ == "__main__":
    unittest.main()
