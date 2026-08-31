import torch
from torch import nn


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
