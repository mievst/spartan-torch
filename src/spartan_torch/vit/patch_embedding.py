import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Linear patch embedding via convolution for Vision Transformers.

    Projects image patches into an embedding space using a single
    ``nn.Conv2d`` with ``kernel_size=patch_size`` and
    ``stride=patch_size``, then flattens the spatial dimensions into a
    token sequence.

    Given input ``(B, C, H, W)`` the output is
    ``(B, num_patches, embed_dim)`` where
    ``num_patches = (H // patch_size) * (W // patch_size)``.

    Supports non-square images and non-square patch sizes. The input
    dimensions must be divisible by ``patch_size``.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 for grayscale, 3 for RGB).
    embed_dim : int
        Output embedding dimension per patch token.
    patch_size : int | tuple[int, int]
        Size of each patch. Single int → square patches; tuple →
        ``(patch_h, patch_w)``.

    References
    ----------
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at
    Scale" (Dosovitskiy et al., 2020, arXiv:2010.11929).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | tuple[int, int],
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Patch tokens, ``(B, num_patches, embed_dim)``.
        """
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
