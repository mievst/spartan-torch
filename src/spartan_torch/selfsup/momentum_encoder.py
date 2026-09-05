import copy
from collections.abc import Iterable

import torch
from torch import nn


class MomentumEncoder(nn.Module):
    """Exponential-moving-average (EMA) copy of a source module (momentum teacher).

    Holds a frozen replica of ``source`` whose weights are pulled toward it via
    ``theta_t <- m * theta_t + (1 - m) * theta_s`` after every call to
    :meth:`update`. This is the momentum-encoder mechanism of MoCo/BYOL/DINO:
    the teacher is a slowly-following, more stable copy that provides better
    targets than the rapidly-updated student.

    The replica runs with gradients disabled — ``forward`` delegates to
    ``source``'s own forward. You build the student from your architecture and
    create a same-configured ``MomentumEncoder`` around it to act as the
    teacher. The teacher is batch-norm-free in DINO, so no running-stat
    synchronization is required.

    Parameters
    ----------
    source : nn.Module
        Module to mirror; its ``parameters`` are deep-copied to form the
        teacher. ``update`` pulls the live (possibly separate) student weights
        from this same module by default — pass a distinct live student via
        :attr:`source` reassignment (or use :meth:`from_student`) when the
        teacher target differs from the initial copy.
    momentum : float, default=0.996
        EMA decay ``m`` in ``[0, 1)``. Higher = slower teacher update. DINO
        raises it from 0.996 to 1.0 over training with a cosine schedule.

    References
    ----------
    "Momentum Contrast for Unsupervised Visual Representation Learning"
    (He et al., 2019, arXiv:1911.05722) — the momentum-encoder mechanism;
    teacher schedule follows DINO (Caron et al., 2021, arXiv:2104.14294).
    """

    def __init__(self, source: nn.Module, momentum: float = 0.996):
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.momentum = momentum
        self.source = source
        self.teacher = copy.deepcopy(source).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the teacher (frozen, no gradient).

        Parameters
        ----------
        x : torch.Tensor
            Input to the teacher network.

        Returns
        -------
        torch.Tensor
            Teacher output; no gradient flows through it.
        """
        return self.teacher(x)

    def update(self, momentum: float | None = None) -> None:
        """EMA-update the teacher weights from the student.

        Parameters
        ----------
        momentum : float | None
            Override decay for this update. ``None`` uses ``self.momentum`` —
            pass a cosine-scheduled value to anneal momentum toward 1.0.
        """
        m = self.momentum if momentum is None else momentum
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), self._student_params(), strict=True
            ):
                t_param.mul_(m).add_(s_param, alpha=1 - m)

    def _student_params(self) -> Iterable[torch.Tensor]:
        """Parameters of the module the teacher mirrors (default: ``self.source``)."""
        return self.source.parameters()

    @classmethod
    def from_student(cls, student: nn.Module, momentum: float = 0.996) -> "MomentumEncoder":
        """Build a momentum copy of a *live* student module.

        Parameters
        ----------
        student : nn.Module
            The live student module the teacher should mirror.

        Returns
        -------
        MomentumEncoder
            Encoder whose ``teacher`` is an EMA copy of ``student`` and whose
            ``update`` pulls from ``student`` directly.
        """
        enc = cls(student, momentum=momentum)
        enc.source = student
        return enc
