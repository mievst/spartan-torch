import math

import torch
from torch import nn


def alibi_slopes(num_heads: int) -> list[float]:
    """Per-head slopes from the ALiBi paper (geometric sequence).

    For a power-of-two head count ``n = 2^m`` the slopes are
    ``2^(-8/n), 2^(-16/n), ..., 2^(-8)``; other counts take the slopes of
    the closest smaller power of two plus every second slope of the next
    doubling (same construction as the reference implementation).
    """
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    def _pow2_slopes(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return _pow2_slopes(num_heads)
    closest = 2 ** math.floor(math.log2(num_heads))
    slopes = _pow2_slopes(closest)
    extra = _pow2_slopes(2 * closest)[0::2][: num_heads - closest]
    return slopes + extra


class ALiBiBias(nn.Module):
    """Attention with Linear Biases (ALiBi) additive score bias.

    Instead of adding position vectors to the embeddings, ALiBi biases each
    head's attention scores by ``-slope_h * distance`` (Press et al., 2021).
    No learned tables, no ``max_seq_len`` — sequences longer than anything
    seen in training just extend the bias, which is why ALiBi extrapolates.

    The natural injection point is the ``mask`` argument of
    :class:`~spartan_torch.MultiHeadAttention` (additive float bias,
    broadcastable to ``(batch, heads, query_len, key_len)``)::

        alibi = ALiBiBias(num_heads=8)
        out, _ = attn(q, k, v, mask=alibi(q_len, k_len))

    Combine with ``is_causal=True`` for the causal variant from the paper;
    without it the bias is symmetric in ``|i - j|``.

    Parameters
    ----------
    num_heads : int
        Number of attention heads (sets the slope schedule).

    References
    ----------
    "Train Short, Test Long: Attention with Linear Biases Enables Input
    Length Extrapolation" (Press, Smith & Lewis, 2021, arXiv:2108.12409).
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor(alibi_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(
        self,
        query_len: int,
        key_len: int,
        k_offset: int = 0,
        causal: bool = True,
    ) -> torch.Tensor:
        """Build the ``(1, heads, query_len, key_len)`` additive bias.

        Parameters
        ----------
        query_len : int
            Number of query positions (current chunk during decoding).
        key_len : int
            Number of key positions in the current chunk.
        k_offset : int, default=0
            Global offset of the chunk (KV-cache decoding: keys before the
            chunk shift the distance matrix so positions stay absolute).
        causal : bool, default=True
            Penalize ``key_pos > query_pos`` with ``-inf`` (in addition to
            the linear distance penalty), i.e. the causal ALiBi from the
            paper. ``False`` gives the symmetric ``|i - j|`` penalty.
        """
        device = self.slopes.device
        q_pos = torch.arange(k_offset, k_offset + query_len, dtype=torch.float32, device=device)
        k_pos = torch.arange(k_offset, k_offset + key_len, dtype=torch.float32, device=device)
        distance = (k_pos[None, :] - q_pos[:, None]).abs()  # (Q, K)
        bias = -distance[None, None, :, :] * self.slopes.view(1, -1, 1, 1)
        if causal:
            future = (k_pos[None, :] - q_pos[:, None]) > 0
            bias = bias.masked_fill(future[None, None, :, :], float("-inf"))
        return bias
