"""DINO (self-distillation with no labels) model built from spartan-torch primitives.

Implements "Emerging Properties in Self-Supervised Vision Transformers"
(Caron et al., 2021, arXiv:2104.14294): a student backbone optimized by
matching the output distribution of a *momentum teacher* on different views
(global + local) of the same image, with the teacher's output centered and
sharpened to prevent collapse.

This is a *model assembly* living in `experiments/vit/dino/` (complete models
stay in experiments; the library holds only reusable primitives —
`DINOProjectionHead`, `DINOLoss`, `Centering`, `Sharpening`, `MomentumEncoder`).

Pipeline per step:
1. feed every view (all global + local) through the **student** backbone+head;
2. feed only the *global* views through the **teacher** backbone+head (EMA copy);
3. minimise cross-entropy between teacher (softmax, center-corrected, sharpened,
   grad-stopped) and student logits;
4. update the teacher weights via EMA toward the student.

The lightning wrapper drives the loop; the datamodule supplies multi-crop views.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
import lightning as L
from torchmetrics import MeanMetric, Accuracy

from spartan_torch import DINOLoss, DINOProjectionHead, MomentumEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision_transformer import VisionTransformer  # noqa: E402


class ViTBackbone(VisionTransformer):
    """Shared classification ViT re-scoped to return the `[CLS]` feature.

    Reuses ``experiments/vit/vision_transformer.py`` unchanged; ``forward``
    keeps the classification contract (linear head) while ``forward_features``
    returns the ``[CLS]`` embedding DINO's projection head consumes.
    """

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Patch -> tokens -> blocks -> norm, return the ``[CLS]`` embedding.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            ``(B, embed_dim)``.
        """
        x = self.patch_embed(x)
        x = self.cls_token(x)
        x = self.pos_embed(x)
        for block in self.encoder:
            x, _ = block(x)
        x = self.norm(x)
        return x[:, 0]


class DINONet(nn.Module):
    """Backbone + projection head composing a single DINO student or teacher.

    Parameters
    ----------
    backbone : dict
        Keyword args for :class:`ViTBackbone` (``img_size``, ``patch_size``,
        ``in_channels``, ``embed_dim``, ``depth``, ``num_heads``).
    head : dict
        Keyword args for :class:`DINOProjectionHead` (``hidden_dim``,
        ``out_dim``, ``norm_last_layer``).
    """

    def __init__(self, backbone: dict, head: dict):
        super().__init__()
        self.backbone = ViTBackbone(**backbone)
        self.head = DINOProjectionHead(
            in_dim=backbone["embed_dim"], **head
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone feature -> project -> L2-normalized logits.

        Parameters
        ----------
        images : torch.Tensor
            ``(B, C, H, W)`` (a single view).

        Returns
        -------
        torch.Tensor
            ``(B, out_dim)``.
        """
        return self.head(self.backbone.forward_features(images))


class DINOLightning(L.LightningModule):
    """Lightning wrapper driving DINO self-distillation.

    Parameters
    ----------
    backbone : dict
        Args for the shared ViT backbone.
    head_hidden_dim : int, default=2048
    head_out_dim : int, default=65536
        Projection head geometry (bottleneck hidden, output prototypes).
    teacher_temp : float, default=0.04
    student_temp : float, default=0.1
    center_momentum : float, default=0.9
    momentum : float, default=0.996
        Base EMA decay for the teacher; annealed toward 1.0 with a cosine
        schedule over training (0.996 -> 1.0), per the paper.
    lr : float, default=5e-4
    weight_decay : float, default=0.04
    warmup_epochs : int, default=10
    max_epochs : int, default=100
    """

    def __init__(
        self,
        backbone: dict,
        head_hidden_dim: int = 2048,
        head_out_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        momentum: float = 0.996,
        lr: float = 5e-4,
        weight_decay: float = 0.04,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone_cfg = backbone

        self.student = DINONet(
            backbone=backbone,
            head={"hidden_dim": head_hidden_dim, "out_dim": head_out_dim, "norm_last_layer": True},
        )
        self.teacher_encoder = MomentumEncoder(self.student, momentum=momentum)
        self.loss = DINOLoss(
            out_dim=head_out_dim,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.train_loss = MeanMetric()

    # --- training helpers --------------------------------------------------
    def _teacher_momentum(self) -> float:
        """Cosine-annealed momentum from base toward 1.0 over `max_epochs`."""
        if self.trainer is None:
            return self.teacher_encoder.momentum
        progress = self.trainer.current_epoch / max(self.max_epochs - 1, 1)
        m = 1.0 - (1.0 - self.teacher_encoder.momentum) * (math.cos(math.pi * progress) + 1) / 2
        return min(m, 0.9999)

    def training_step(self, batch, batch_idx):
        # batch: (student_views [list of tensors], teacher_views [global views])
        student_views, teacher_views = batch
        student_out = [self.student(v) for v in student_views]
        with torch.no_grad():
            teacher_out = [self.teacher_encoder(v) for v in teacher_views]
        self.loss.update_center(torch.cat(teacher_out, dim=0))
        loss = self.loss(student_out, teacher_out)
        self.teacher_encoder.update(momentum=self._teacher_momentum())
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train/momentum", self._teacher_momentum())
        return loss

    def validation_step(self, batch, batch_idx):
        student_views, teacher_views = batch
        with torch.no_grad():
            teacher_out = [self.teacher_encoder(v) for v in teacher_views]
        acc = self._proto_accuracy(student_views, teacher_out)
        self.log("val/proto_acc", acc, on_step=False, on_epoch=True)

    @torch.no_grad()
    def _proto_accuracy(
        self,
        student_views: list[torch.Tensor],
        teacher_out: list[torch.Tensor],
    ) -> torch.Tensor:
        """Fraction of student predictions matching the teacher's argmax.

        Only the views that have a teacher target (the global ones) are
        compared; local student views are skipped.
        """
        student_global = [self.student(v) for v in student_views[: len(teacher_out)]]
        predictions = torch.cat(student_global, dim=0)
        teacher_probs = torch.cat(
            [self.loss._teacher_distribution(t) for t in teacher_out], dim=0
        )
        return (predictions.argmax(-1) == teacher_probs.argmax(-1)).float().mean()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        steps_per_epoch = self._steps_per_epoch()
        total_steps = max(self.max_epochs, 1) * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: _cosine_warmup(step, warmup_steps, total_steps),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def _steps_per_epoch(self) -> int:
        if self.trainer is None:
            return 1
        try:
            dl = self.trainer.train_dataloader
            if dl is not None:
                return max(len(dl), 1)
        except Exception:
            pass
        try:
            dls = self.trainer.datamodule
            if dls is not None:
                return max(len(dls.train_dataloader()), 1)
        except Exception:
            pass
        return 1

    # --- post-training eval -------------------------------------------------
    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """L2-normalized teacher backbone features for k-NN / linear probe.

        Moves the network to the input device — lightning returns the module
        to CPU after ``fit`` (2.6 behaviour), so an explicit re-anchor here
        keeps post-training ``embed`` calls device-consistent.
        """
        self.eval()
        self.to(images.device)
        feats = self.teacher_encoder.teacher.backbone.forward_features(images)
        return torch.nn.functional.normalize(feats, dim=-1)


def _cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
