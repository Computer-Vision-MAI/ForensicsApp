"""Set 10 extra: the Vision Transformer of the extra notebook, in PyTorch (needs torch).

The notebook builds the model in Keras. TensorFlow has no GPU support on macOS, so here
the same layers are written in PyTorch, which trains on the Apple GPU (MPS) or CUDA.
Layer by layer, with the notebook's hyper-parameters:

1. **Augmentation** (training only): normalise with the training mean/variance, resize
   to 72 × 72, random horizontal flip, rotation of ±0.02 turns (±7.2°) and zoom of ±20 %
   (reflected borders, like Keras). At test time only normalise + resize.
2. **Patches**: 144 patches of 6 × 6 × 3 = 108 values (or SPT: 6 × 6 × 15 = 540, with a
   LayerNorm before the projection, as in the SPT paper).
3. **PatchEncoder**: a Dense projection to 64 dimensions + a learnt position embedding
   (one 64-D vector per patch position): without it the transformer would not know
   where each patch was.
4. **8 transformer blocks** (pre-norm): LayerNorm → multi-head self-attention (4 heads,
   key_dim 64, dropout 0.1) → + skip; LayerNorm → MLP (128, 64, GELU, dropout 0.1) → + skip.
5. **Head**: LayerNorm → flatten the 144 × 64 tokens → dropout 0.5 → MLP 2048, 1024 (GELU,
   dropout 0.5) → Dense(classes) logits.

Keras' ``MultiHeadAttention(num_heads=4, key_dim=64)`` gives each head its own 64-D
queries, keys and values (4 × 64 = 256 inside the layer) and projects back to 64; the
``Attention`` module below does exactly that. ``lsa=True`` adds Locality Self-Attention
from the SPT paper: a learnt temperature instead of the fixed sqrt(d), and each token
may not attend to itself (diagonal masked), so the attention looks at the others.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Augment(nn.Module):
    """Normalisation + resizing, and random flip / rotation / zoom while training (on the GPU)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, size: int, rotation: float = 0.02, zoom: float = 0.2):
        super().__init__()
        self.register_buffer("mean", mean.view(1, 3, 1, 1).float())
        self.register_buffer("std", std.view(1, 3, 1, 1).float())
        self.size = size
        self.rotation = rotation
        self.zoom = zoom

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 3, 32, 32) in [0, 255]
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        if not self.training:
            return x
        b = x.shape[0]
        device = x.device
        flip = torch.where(torch.rand(b, device=device) < 0.5, -1.0, 1.0)
        angle = (torch.rand(b, device=device) * 2 - 1) * self.rotation * 2 * math.pi
        # Keras' zoom factor z in [-0.2, 0.2] scales the sampled area by (1 + z), separately in x and y
        zoom_x = 1 + (torch.rand(b, device=device) * 2 - 1) * self.zoom
        zoom_y = 1 + (torch.rand(b, device=device) * 2 - 1) * self.zoom
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros(b, 2, 3, device=device)
        theta[:, 0, 0] = cos * zoom_x * flip
        theta[:, 0, 1] = -sin * zoom_y
        theta[:, 1, 0] = sin * zoom_x * flip
        theta[:, 1, 1] = cos * zoom_y
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, patch·patch·C), patches row by row, values in (row, col, channel) order like TF."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // patch, patch, w // patch, patch)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // patch) * (w // patch), patch * patch * c)


def shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Move the content by (dy, dx) pixels, zeros where nothing comes in (crop + pad, as in the Keras SPT)."""
    out = torch.zeros_like(x)
    h, w = x.shape[-2:]
    out[..., max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] = \
        x[..., max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


class PatchTokens(nn.Module):
    """Patches → Dense projection (+ the position embedding added in ``ViT``)."""

    def __init__(self, patch: int, dim: int):
        super().__init__()
        self.patch = patch
        self.projection = nn.Linear(patch * patch * 3, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(patchify(x, self.patch))


class ShiftedPatchTokens(nn.Module):
    """SPT: the image and its 4 diagonal half-patch shifts → patches → LayerNorm → Dense projection."""

    def __init__(self, patch: int, dim: int):
        super().__init__()
        self.patch = patch
        self.norm = nn.LayerNorm(patch * patch * 15, eps=1e-6)
        self.projection = nn.Linear(patch * patch * 15, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.patch // 2
        stacked = torch.cat([x] + [shift(x, sy * half, sx * half) for sy, sx in ((-1, -1), (1, -1), (-1, 1), (1, 1))],
                            dim=1)
        return self.projection(self.norm(patchify(stacked, self.patch)))


class Attention(nn.Module):
    """Keras-style multi-head self-attention: every head has key_dim-sized queries, keys and values."""

    def __init__(self, dim: int, heads: int, key_dim: int, dropout: float, lsa: bool = False):
        super().__init__()
        self.heads, self.key_dim = heads, key_dim
        self.query = nn.Linear(dim, heads * key_dim)
        self.key = nn.Linear(dim, heads * key_dim)
        self.value = nn.Linear(dim, heads * key_dim)
        self.out = nn.Linear(heads * key_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.lsa = lsa
        # LSA: learnable temperature, initialised to the usual sqrt(d_k)
        self.tau = nn.Parameter(torch.tensor(math.sqrt(key_dim))) if lsa else None
        self.last_attention: torch.Tensor | None = None  # kept only when keep_attention is set
        self.keep_attention = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, self.heads, self.key_dim).transpose(1, 2)  # (B, heads, N, key_dim)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        scale = self.tau if self.lsa else math.sqrt(self.key_dim)
        scores = q @ k.transpose(-2, -1) / scale  # (B, heads, N, N): how much token i looks at token j
        if self.lsa:
            eye = torch.eye(n, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(eye, float("-inf"))
        weights = scores.softmax(dim=-1)
        if self.keep_attention:
            self.last_attention = weights.detach()
        out = self.dropout(weights) @ v
        return self.out(out.transpose(1, 2).reshape(b, n, self.heads * self.key_dim))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: tuple[int, ...], lsa: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = Attention(dim, heads, dim, 0.1, lsa)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        layers: list[nn.Module] = []
        width = dim
        for units in mlp:
            layers += [nn.Linear(width, units), nn.GELU(), nn.Dropout(0.1)]
            width = units
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))  # skip connection 1
        return x + self.mlp(self.norm2(x))  # skip connection 2


class ViT(nn.Module):
    def __init__(self, num_classes: int, mean: torch.Tensor, std: torch.Tensor, image_size: int = 72,
                 patch: int = 6, dim: int = 64, heads: int = 4, layers: int = 8,
                 head_units: tuple[int, ...] = (2048, 1024), spt: bool = False, lsa: bool = False):
        super().__init__()
        self.num_patches = (image_size // patch) ** 2
        self.augment = Augment(mean, std, image_size)
        self.tokens = ShiftedPatchTokens(patch, dim) if spt else PatchTokens(patch, dim)
        self.position = nn.Embedding(self.num_patches, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, (dim * 2, dim), lsa) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        head: list[nn.Module] = [nn.Flatten(), nn.Dropout(0.5)]
        width = self.num_patches * dim
        for units in head_units:
            head += [nn.Linear(width, units), nn.GELU(), nn.Dropout(0.5)]
            width = units
        head.append(nn.Linear(width, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 3, 32, 32) in [0, 255]
        x = self.augment(x)
        positions = torch.arange(self.num_patches, device=x.device)
        z = self.tokens(x) + self.position(positions)
        for block in self.blocks:
            z = block(z)
        return self.head(self.norm(z))

    def keep_attention(self, keep: bool) -> None:
        for block in self.blocks:
            block.attention.keep_attention = keep
            block.attention.last_attention = None

    def attention_maps(self) -> list[torch.Tensor]:
        return [block.attention.last_attention for block in self.blocks]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
