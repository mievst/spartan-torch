import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encodings from "Attention Is All You Need"
    (Vaswani et al., 2017).

    Computes a fixed ``(max_seq_len, emb_size)`` table once in ``__init__`` and
    stores it as a non-trainable buffer. ``forward`` slices the table to the
    actual sequence length and adds it to the input.

    Parameters
    ----------
    emb_size : int
        Embedding dimension.
    max_seq_len : int, default=5000
        Maximum supported sequence length.
    """

    def __init__(self, emb_size: int, max_seq_len: int = 5000):
        super().__init__()
        if emb_size % 2 != 0:
            raise ValueError(f"emb_size ({emb_size}) must be even, got odd")

        self.emb_size = emb_size
        self.max_seq_len = max_seq_len

        pos = torch.arange(max_seq_len, dtype=torch.float32)[:, None]
        div_term = torch.exp(
            torch.arange(0, emb_size, 2, dtype=torch.float32)
            * (-(math.log(10000.0)) / emb_size)
        )

        pe = torch.zeros(max_seq_len, emb_size)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, seq_len, emb_size)``. ``seq_len`` must be
            ``<= max_seq_len``.

        Returns
        -------
        torch.Tensor
            ``x + pe[:, :seq_len]``, same shape.
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )
        return x + self.pe[:, :seq_len]
