import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Normalizes by the RMS of the input instead of the mean/variance used by
    ``nn.LayerNorm``, so there is no mean subtraction and no re-centering
    bias by default. Cheaper than ``nn.LayerNorm`` and used by most modern
    LLMs (LLaMA, Gemma, Qwen). Drop-in for ``nn.LayerNorm`` in the
    ``norm_layer`` factories: exposes ``normalized_shape`` and takes the same
    constructor arguments.

    ``x_normed = x / sqrt(mean(x^2) + eps) * weight (+ bias)``.

    Parameters
    ----------
    normalized_shape : int | tuple[int, ...]
        Shape of the normalized trailing dimensions. An int is treated as a
        one-element tuple, matching the ``nn.LayerNorm`` convention.
    eps : float, default=1e-5
        Stability term added under the square root.
    bias : bool, default=False
        Add a learnable bias. ``nn.LayerNorm`` always has one; RMSNorm
        commonly omits it, but ``bias=True`` reproduces the affine set
        (weight + bias) so checkpoints can be swapped 1-to-1.

    References
    ----------
    "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019,
    arXiv:1910.07467).
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        )
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(*self.normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))
        rms = torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + self.eps)
        out = x * rms * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
