import numpy as np
import torch


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
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
