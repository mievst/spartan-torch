import torch
from torch import nn


class Centering(nn.Module):
    """Teacher-output centering (DINO), a collapse-avoidance bias.

    Maintains an exponential-moving-average estimate of the teacher's output
    mean and subtracts it from each new output before sharpening. Combined
    with :class:`Sharpening`, centering prevents the uniformly-flat (entropy-
    maximized) collapse mode by removing the DC component of the distribution.

    ``center = (1 - decay) * center + decay * batch_mean`` implemented as a
    bias tracked under ``torch.no_grad()``-safe momentum (the buffer is updated
    in-place on the forward pass, no gradient).

    Parameters
    ----------
    dim : int
        Output dimension of the logits.
    momentum : float, default=0.9
        EMA decay for the running center. Higher = slower-moving center.

    References
    ----------
    "Emerging Properties in Self-Supervised Vision Transformers" (Caron et
    al., 2021, arXiv:2104.14294) — centering + sharpening against collapse.
    """

    def __init__(self, dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract the running center and update it.

        Parameters
        ----------
        x : torch.Tensor
            Teacher logits ``(B, dim)``.

        Returns
        -------
        torch.Tensor
            ``(B, dim)`` with the running mean removed.
        """
        with torch.no_grad():
            self.center = self.momentum * self.center + (1 - self.momentum) * x.mean(0)
        return x - self.center


class Sharpening(nn.Module):
    """Temperature-based sharpening (DINO) of the teacher distribution.

    Applies low-temperature softmax to the teacher's centered logits so the
    target distribution is peaked. Combined with :class:`Centering`, this is
    what prevents collapse when training without labels.

    Parameters
    ----------
    temperature : float, default=0.04        Teacher softmax temperature. Lower = sharper/peaked distribution.

    References
    ----------
    "Emerging Properties in Self-Supervised Vision Transformers" (Caron et
    al., 2021, arXiv:2104.14294) — centering + sharpening against collapse.
    """

    def __init__(self, temperature: float = 0.04):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sharpen via low-temperature softmax.

        Parameters
        ----------
        x : torch.Tensor
            Centered teacher logits ``(B, dim)``.

        Returns
        -------
        torch.Tensor
            ``(B, dim)`` probability distribution.
        """
        return (x / self.temperature).softmax(dim=-1)
