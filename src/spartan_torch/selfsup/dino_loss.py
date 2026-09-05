import torch
from torch import nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Cross-entropy distillation loss between student and teacher distributions.

    For each global (teacher) view ``t`` and every other view ``s`` (global or
    local), the loss is the cross-entropy of the teacher's (softmaxed,
    sharpened, center-corrected) distribution ``P_t`` against the student's
    logits ``P_s``, with gradients stopped on the teacher side:

        loss = - sum_t sum_{s != t} P_t(x_t).log P_s(x_s)

    A per-example entropy bonus (with weight ``ent_weight``) keeps features
    spread when the target distribution is degenerate; ``freeze_last_epoch``
    allows freezing centering for the final epochs (a training-loop concern,
    left as a flag here for completeness). The loss is computed purely with
    :mod:`torch` — no external framework — so it plugs into any training loop.

    Parameters
    ----------
    out_dim : int
        Number of logits per view.
    teacher_temp : float, default=0.04
        Teacher softmax temperature (sharpening).
    student_temp : float, default=0.1
        Student softmax temperature.
    center_momentum : float, default=0.9
        EMA decay for the running teacher center.
    ent_weight : float, default=0.0
        Weight of the mean-entropy regularizer on the student distribution.
    freeze_last_epoch : int, default=0
        Number of final epochs during which centering/temp sharpening are
        frozen (pass-through). Kept as a no-op parameter here, honoured by
        the training loop.

    References
    ----------
    "Emerging Properties in Self-Supervised Vision Transformers" (Caron et
    al., 2021, arXiv:2104.14294).
    """

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        ent_weight: float = 0.0,
        freeze_last_epoch: int = 0,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.ent_weight = ent_weight
        self.center_momentum = center_momentum
        self.freeze_last_epoch = freeze_last_epoch
        self.register_buffer("center", torch.zeros(out_dim))

    def update_center(self, teacher_out: torch.Tensor) -> None:
        """Update the running teacher center (call before the loss each step).

        Parameters
        ----------
        teacher_out : torch.Tensor
            Raw teacher logits ``(sum of views, out_dim)``.
        """
        with torch.no_grad():
            self.center = self.center * self.center_momentum + teacher_out.mean(0) * (1 - self.center_momentum)

    def _teacher_distribution(self, teacher_out: torch.Tensor) -> torch.Tensor:
        centered = teacher_out - self.center
        return (centered / self.teacher_temp).softmax(dim=-1)

    @torch.no_grad()
    def _teacher_loss(self, teacher_out: torch.Tensor) -> torch.Tensor:
        p = self._teacher_distribution(teacher_out)
        return -(p * p.log()).sum(dim=-1, keepdim=True).mean(dim=0, keepdim=True)

    def forward(
        self,
        student_out: list[torch.Tensor],
        teacher_out: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the DINO loss.

        Parameters
        ----------
        student_out : list[torch.Tensor]
            Student logits for each view, each ``(B, out_dim)``. Number of
            views = number of global teacher views (typically 2).
        teacher_out : list[torch.Tensor]
            Teacher logits for each *global* view, each ``(B, out_dim)``.
            Gradients are stopped on these.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        student_out = [s / self.student_temp for s in student_out]
        student_out = [s.log_softmax(dim=-1) for s in student_out]

        teacher_out = [t.detach() for t in teacher_out]
        teacher_probs = [self._teacher_distribution(t) for t in teacher_out]

        total_loss = torch.zeros((), device=student_out[0].device)
        n_loss_terms = 0
        for t_idx, t_probs in enumerate(teacher_probs):
            for s_idx, s_log in enumerate(student_out):
                if s_idx == t_idx:
                    continue
                loss = -(t_probs * s_log).sum(dim=-1).mean()
                if self.ent_weight:
                    loss -= self.ent_weight * self._teacher_loss(teacher_out[t_idx]).sum()
                total_loss += loss
                n_loss_terms += 1
        return total_loss / max(n_loss_terms, 1)
