import math

import torch
from torch import nn
import torch.nn.functional as F


class LearnablePositionEmbedding(nn.Module):
    """Learnable 1D positional embeddings for Vision Transformers.

    Stores a learnable ``nn.Parameter`` of shape ``(1, max_len, embed_dim)``
    and adds it to the input sequence.  Supports sequences shorter than
    ``max_len`` (truncates the embeddings automatically).

    As shown in the original ViT paper (Sec 3.1), 1D learned position
    embeddings perform on par with 2D-aware variants because the model
    learns spatial topology implicitly from the patch order.

    Parameters
    ----------
    max_len : int
        Maximum sequence length (number of patches + 1 for CLS token).
    embed_dim : int
        Embedding dimension per token.

    References
    ----------
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at
    Scale" (Dosovitskiy et al., 2020, arXiv:2010.11929), Sec 3.1.
    """

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Token sequence, ``(B, N, D)``. ``N`` must be ≤ ``max_len``.

        Returns
        -------
        torch.Tensor
            ``(B, N, D)`` with positional embeddings added.
        """
        n = x.size(1)
        max_len = self.pos_embed.size(1)
        if n > max_len:
            raise IndexError(
                f"sequence length {n} exceeds max_len {max_len}"
            )
        return x + self.pos_embed[:, :n]

    def interpolate_grid(self, h: int, w: int) -> torch.Tensor:
        """Bicubic-interpolated positional embeddings for an ``h`` × ``w`` grid.

        Mirrors ``interpolate_pos_encoding`` in ``facebookresearch/dino``:
        the ``[CLS]`` embedding is kept as-is, the patch embeddings are
        reshaped to their stored square grid and bicubic-interpolated to the
        target grid (e.g. an 8×8 table learned on 32×32 images compressed to
        4×4 for 16×16 local crops). Requesting the stored grid returns the
        underlying parameter slice untouched (bitwise identity, no resampling).

        Parameters
        ----------
        h : int
            Target grid height in patches.
        w : int
            Target grid width in patches.

        Returns
        -------
        torch.Tensor
            ``(1, 1 + h * w, embed_dim)`` positional table.

        References
        ----------
        "Emerging Properties in Self-Supervised Vision Transformers" (Caron et
        al., 2021, arXiv:2104.14294).
        """
        n_patches = self.pos_embed.size(1) - 1
        n0 = math.isqrt(n_patches)
        if n0 * n0 != n_patches:
            raise ValueError(
                f"stored patch table {n_patches} is not a square grid, "
                "cannot interpolate"
            )
        if h <= 0 or w <= 0:
            raise ValueError(f"target grid must be positive, got {(h, w)}")
        if h == n0 and w == n0:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:].reshape(1, n0, n0, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_grid(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Add grid-interpolated positional embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Token sequence, ``(B, N, D)`` with ``N ≤ 1 + h * w``.
        h : int
            Patch-grid height of the input image.
        w : int
            Patch-grid width of the input image.

        Returns
        -------
        torch.Tensor
            ``(B, N, D)`` with positional embeddings added.
        """
        table = self.interpolate_grid(h, w)
        n = x.size(1)
        if n > table.size(1):
            raise IndexError(
                f"sequence length {n} exceeds interpolated table {table.size(1)}"
            )
        return x + table[:, :n]
