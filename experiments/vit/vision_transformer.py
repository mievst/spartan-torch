import torch
from torch import nn

from spartan_torch import (
    ClassToken,
    FeedForward,
    LearnablePositionEmbedding,
    MultiHeadAttention,
    PatchEmbedding,
    TransformerBlock,
)


class VisionTransformer(nn.Module):
    """Vision Transformer (Dosovitskiy et al., 2020) built from spartan-torch primitives.

    Composes :class:`PatchEmbedding`, :class:`ClassToken`,
    :class:`LearnablePositionEmbedding`, and :class:`TransformerBlock`
    into a complete ViT encoder with a classification head.

    Follows the original ViT architecture (Sec 3.1):
    ``patch_embed → cls_token → + pos_embed → N × TransformerBlock → norm → head``.

    The ``state_dict`` key layout is intentionally aligned with ``timm`` so
    that pretrained weights can be loaded with a lightweight key remapping
    (see ``experiments/vit/timm_compat/``).

    Parameters
    ----------
    img_size : int
        Input image size (assumed square).
    patch_size : int
        Patch size (square).
    in_channels : int
        Number of image channels.
    num_classes : int
        Number of output classes.
    embed_dim : int
        Embedding dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    ff_hidden_size : int
        Hidden size of the feed-forward network.
    dropout_p : float
        Dropout probability in transformer blocks.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ff_hidden_size: int = 3072,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = ClassToken(embed_dim)
        self.pos_embed = LearnablePositionEmbedding(num_patches + 1, embed_dim)

        head_size = embed_dim // num_heads
        self.encoder = nn.ModuleList([
            TransformerBlock(
                in_size=embed_dim,
                head_size=head_size,
                num_heads=num_heads,
                out_size=embed_dim,
                ff_hidden_size=ff_hidden_size,
                dropout_p=dropout_p,
                use_sdpa=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits, ``(B, num_classes)``.
        """
        x = self.patch_embed(x)
        x = self.cls_token(x)
        x = self.pos_embed(x)

        for block in self.encoder:
            x, _ = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])
