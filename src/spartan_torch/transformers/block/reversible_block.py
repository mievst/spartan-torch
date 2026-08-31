from collections.abc import Callable

import torch
from torch import nn


class ChunkedFeedForward(nn.Module):
    """Feed-forward wrapper that processes the sequence in chunks.

    The inner module (typically a :class:`FeedForward`) is applied to
    ``chunk_size`` tokens at a time along the sequence dimension. Since FF
    layers are pointwise (no cross-token interaction), the result is identical
    to a single call — the wrapper only caps peak activation memory, which is
    what Reformer's chunked FF layer does for long sequences.

    Parameters
    ----------
    ff : nn.Module
        Pointwise module with the ``(batch, seq, embed)`` in/out contract.
    chunk_size : int, default=1024
        Sequence chunk length. ``ff`` is called once when ``seq_len`` does not
        exceed ``chunk_size``.
    """

    def __init__(self, ff: nn.Module, chunk_size: int = 1024):
        super().__init__()
        self.ff = ff
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        if n <= self.chunk_size:
            return self.ff(x)
        num_chunks = (n + self.chunk_size - 1) // self.chunk_size
        return torch.cat([self.ff(part) for part in x.chunk(num_chunks, dim=1)], dim=1)


class SelfAttention(nn.Module):
    """Adapter turning a ``(query, key, value)`` attention into self-attention.

    Attention bricks in this library take separate query/key/value tensors.
    Reversible blocks and other single-input wrappers need a module that
    consumes one tensor and feeds it to all three roles. The adapter is a
    regular child module, so its weights (and the inner attention's) live
    normally in ``state_dict``.

    Parameters
    ----------
    attention : nn.Module
        Module with the ``(query, key, value)`` in/out contract, e.g.
        :class:`~spartan_torch.transformers.attention.ReformerAttention`.
    """

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x, x, x)


class ReversibleBlock(nn.Module):
    """Reversible residual block — the memory-saving core of Reformer.

    Splits the input in half along the embedding dim and computes
    ``y1 = x1 + F(LN(x2))``, ``y2 = x2 + G(LN(y1))``, then concatenates. The
    block stores no hidden activations between ``y1`` and ``y2``; with
    ``use_checkpoint=True`` the whole forward runs under
    :func:`torch.utils.checkpoint`, so at training time intermediate
    activations are discarded and recomputed in backward — memory scales as
    :math:`O(\\text{seq} \\cdot d)`, not :math:`O(\\text{seq} \\cdot \\text{layers} \\cdot d)`.

    Because the split is reversible, :meth:`reverse` recovers the input from
    the output exactly (evaluation only), so the backward pass could in
    principle be run from the output without the checkpoint machinery — the
    wrapper below already achieves the same memory profile.

    ``f`` and ``g`` are injected as modules (e.g. a
    :class:`SelfAttention` wrapping :class:`ReformerAttention` and a
    :class:`ChunkedFeedForward`) — the two LayerNorms are part of the block
    itself, matching the paper's pre-norm reversible structure.

    Parameters
    ----------
    f : nn.Module
        First sublayer (attention in the paper), ``(batch, seq, hidden)`` in/out.
    g : nn.Module
        Second sublayer (feed-forward in the paper), ``(batch, seq, hidden)`` in/out.
    hidden_size : int
        Per-half embedding size. The block input/output is
        ``(batch, seq, 2 * hidden_size)``.
    use_checkpoint : bool, default=False
        Wrap the forward in :func:`torch.utils.checkpoint.checkpoint` during
        training (ignored in evaluation).
    """

    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        hidden_size: int,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.f = f
        self.g = g
        self.norm_f = nn.LayerNorm(hidden_size)
        self.norm_g = nn.LayerNorm(hidden_size)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 + self.f(self.norm_f(x2))
        y2 = x2 + self.g(self.norm_g(y1))
        return torch.cat([y1, y2], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Reverse the block: ``reverse(forward(x)) == x``.

        Uses the inverse residual arithmetic ``x2 = y2 - G(LN(y1))``,
        ``x1 = y1 - F(LN(x2))``. Evaluation only — any stochastic sublayer
        (dropout) would break the reconstruction, so it raises in training
        mode and runs under ``no_grad``.
        """
        if self.training:
            raise RuntimeError("ReversibleBlock.reverse() requires eval mode (no dropout)")
        with torch.no_grad():
            y1, y2 = y.chunk(2, dim=-1)
            x2 = y2 - self.g(self.norm_g(y1))
            x1 = y1 - self.f(self.norm_f(x2))
            return torch.cat([x1, x2], dim=-1)
