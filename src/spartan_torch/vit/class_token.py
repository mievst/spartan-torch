import torch
from torch import nn


class ClassToken(nn.Module):
    """Prepend a learnable classification token to a sequence.

    Follows the BERT/ViT convention: a single learnable embedding is
    prepended to the patch token sequence. The output sequence is one
    token longer than the input.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension of the input tokens.

    References
    ----------
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at
    Scale" (Dosovitskiy et al., 2020, arXiv:2010.11929).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend the CLS token.

        Parameters
        ----------
        x : torch.Tensor
            Patch token sequence, ``(B, N, D)``.

        Returns
        -------
        torch.Tensor
            ``(B, N + 1, D)`` with the CLS token at position 0.
        """
        return torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
