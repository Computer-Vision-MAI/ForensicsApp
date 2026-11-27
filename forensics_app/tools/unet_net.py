"""Set 11: the U-Net of the notebook and its losses (needs PyTorch).

This module imports torch at the top, so it is only imported by ``unet_model`` inside
the functions that need it; the rest of the app and the tests work without PyTorch.

The network is the notebook's ``UNet`` (482,737 parameters with ``base=16``):

    input (3×H×W)
      enc1 ─────────────────────────────► dec1 ─► 1×1 conv ─► output (C×H×W)
        │ pool                              ▲ up
        enc2 ───────────────────────► dec2
          │ pool                        ▲ up
          enc3 ─────────────────► dec3
            │ pool                ▲ up
            └──── bottleneck ─────┘

Each block is two 3×3 convolutions with BatchNorm and ReLU. ``skip=False`` removes the
skip connections (Exercise 2): the decoder then only gets the upsampled features, so
the first convolution of each decoder block has half the input channels.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

LOSSES = {
    "bce_dice": "BCE + Dice (notebook)",
    "bce": "BCE only (cross-entropy for several classes)",
    "dice": "Dice only",
}


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3×3 convolutions, each followed by BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 16, skip: bool = True) -> None:
        super().__init__()
        self.skip = skip
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        joined = 2 if skip else 1  # the skip connection doubles the decoder's input channels
        # Encoder: what is in the image (fewer pixels, more channels)
        self.enc1, self.enc2, self.enc3 = conv_block(in_ch, c1), conv_block(c1, c2), conv_block(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(c3, c4)
        # Decoder: upsample, concatenate the skip connection, then convolve
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), conv_block(c3 * joined, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), conv_block(c2 * joined, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), conv_block(c1 * joined, c1)
        self.head = nn.Conv2d(c1, out_ch, 1)  # one score (logit) per pixel and class

    def _join(self, up: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
        return torch.cat([up, encoder], dim=1) if self.skip else up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._join(self.up3(b), e3))
        d2 = self.dec2(self._join(self.up2(d3), e2))
        d1 = self.dec1(self._join(self.up1(d2), e1))
        return self.head(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def soft_dice_binary(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The notebook's ``dice_coef``: one Dice per image, (B, 1, H, W) tensors in [0, 1]."""
    inter = (probs * target).sum(dim=(1, 2, 3))
    total = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * inter + eps) / (total + eps)


def soft_dice_multiclass(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft Dice over the classes present in the batch. probs (B, C, H, W), target (B, H, W)."""
    one_hot = F.one_hot(target, probs.shape[1]).permute(0, 3, 1, 2).to(probs.dtype)
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (total + eps)
    present = one_hot.sum(dim=(0, 2, 3)) > 0  # classes absent from the batch would only add noise
    return dice[present].mean() if present.any() else dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    """``bce_dice`` (notebook), ``bce`` or ``dice``; target (B, 1, H, W) float for 1 output, (B, H, W) long otherwise."""
    if kind not in LOSSES:
        raise ValueError(f"Unknown loss {kind!r}.")
    if logits.shape[1] == 1:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = soft_dice_binary(torch.sigmoid(logits), target).mean()
    else:
        bce = F.cross_entropy(logits, target)
        dice = soft_dice_multiclass(torch.softmax(logits, dim=1), target)
    if kind == "bce":
        return bce
    if kind == "dice":
        return 1 - dice
    return bce + (1 - dice)
