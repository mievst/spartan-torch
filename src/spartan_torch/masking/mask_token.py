import torch
from torch import nn


class MaskedToken(nn.Module):
    """Learnable token placed on masked positions of a sequence.

    A single learnable embedding shared across all masked positions — the
    BERT/[MASK] convention adapted to sequences of patch tokens. Unlike
    :class:`~spartan_torch.ClassToken` (which is prepended at position 0),
    :class:`MaskedToken` is substituted *in place* at arbitrary positions,
    which is what a masked-autoencoder decoder needs when reassembling the
    full sequence from the kept and masked subsets.

    Given a ``number`` of positions to fill, returns ``(batch, number, dim)``
    copies of the token. The `number`-many positions are expected to be the
    masked (reconstruction) tokens; kept tokens are inserted around them by
    the caller via :data:`ids_restore` (see
    :class:`~spartan_torch.RandomPatchMasking`).

    Parameters
    ----------
    dim : int
        Embedding dimension of the token.

    References
    ----------
    "BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding" (Devlin et al., 2018, arXiv:1810.04805) — the [MASK]
    convention this adapts to patch tokens.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(dim))

    def forward(self, number: int) -> torch.Tensor:
        """Broadcast the mask token.

        Parameters
        ----------
        number : int
            Number of masked positions to produce.

        Returns
        -------
        torch.Tensor
            ``(batch_agnostic=1, number, dim)`` repeated mask tokens. The
            leading dim is ``1`` so the caller can ``expand`` to the batch
            size when concatenating with kept tokens.
        """
        return self.mask_token.view(1, 1, -1).expand(1, number, -1)
