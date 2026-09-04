import numpy as np
import torch


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine warmup wrapper around another LR scheduler.

    For the first ``warmup`` steps the LR ramps from ``min_lr`` to the base
    LR on a cosine curve, then delegates to ``scheduler``. Warmup stabilizes
    early training of deep transformers (post-norm especially).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose LR is scheduled.
    warmup : int
        Number of warmup steps.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Scheduler to delegate to after warmup.
    min_lr : float, default=1e-9
        LR floor at the start of warmup.

    References
    ----------
    "Attention Is All You Need" (Vaswani et al., 2017, arXiv:1706.03762) —
    the warmup schedule (Sec 5.3).
    """

    def __init__(self, optimizer, warmup, scheduler, min_lr=1e-9):
        self.warmup = warmup
        self.scheduler = scheduler
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.warmup <= 0:
            return list(self.base_lrs)
        progress = min(self.last_epoch, self.warmup) / self.warmup
        cosine = 0.5 * (1 - np.cos(np.pi * progress))
        return [self.min_lr + (base - self.min_lr) * cosine for base in self.base_lrs]

    def step(self, *args, **kwargs):
        self.last_epoch += 1
        if self.last_epoch <= self.warmup:
            lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
                param_group["lr"] = lr
        else:
            self.scheduler.step(*args, **kwargs)

    def get_last_lr(self):
        if self.last_epoch <= self.warmup:
            return self.get_lr()
        return list(self.scheduler.get_last_lr())
