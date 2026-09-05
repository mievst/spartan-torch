"""pytorch-lightning wrappers for MAE pretrain and MAE fine-tuning.

The raw :class:`MAEModel` is pure ``torch.nn`` (see ``mae_model.py``); the
lightning modules below are the training-loop glue — pretrain reconstruction
and the standard MAE fine-tune recipe (a L2-normalized feature + linear head
learnt from scratch on top of a *frozen* encoder, or full fine-tune).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
import lightning as L
from torchmetrics import MeanMetric, Accuracy

from mae_model import MAEModel


class MAEPretrainLightning(L.LightningModule):
    """Lightning wrapper driving MAE self-supervised pretraining.

    Logs mean reconstruction loss (train + val) and the mean fraction of
    masked tokens so the ratio check is visible.

    Parameters
    ----------
    model : MAEModel
        The (framework-agnostic) MAE model to pretrain.
    lr : float, default=1.5e-4
        Peak learning rate (paper uses 1.5e-4 for ViT-Base, 800 epochs).
    weight_decay : float, default=0.05
    warmup_epochs : int, default=5
    max_epochs : int, default=800
        Used to build the cosine schedule warmup length.
    """

    def __init__(
        self,
        model: MAEModel,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 800,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, _ = batch
        out = self.model(image)
        self.train_loss(out["loss"])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/mask_ratio", out["mask"].float().mean())
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        image, _ = batch
        out = self.model(image)
        self.val_loss(out["loss"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        steps_per_epoch = self._steps_per_epoch()
        total_steps = max(self.max_epochs, 1) * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda step: _cosine_warmup(step, warmup_steps, total_steps),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

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

    @staticmethod
    def load_shared_state(ckpt_path: str, model: nn.Module) -> nn.Module:
        """Load the MAE encoder into a classification ViT (``patch_embed`` + ``blocks`` + ``norm``).

        The MAE encoder state (``encoder.patch_embed.*``, ``encoder.blocks.*``,
        ``encoder.norm.*``) is remapped to the classifier keys
        (``patch_embed.*``, ``encoder.*``, ``norm.*``); ``cls_token``,
        ``pos_embed`` and ``head`` are left to be learnt from scratch.
        """
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        mapped = {}
        for k, v in sd.items():
            if not k.startswith("model.encoder."):
                continue
            rel = k[len("model.encoder."):]
            if rel.startswith("patch_embed.") or rel.startswith("norm."):
                mapped[rel] = v
            elif rel.startswith("blocks."):
                mapped["encoder." + rel[len("blocks."):]] = v
        filtered = {k: v for k, v in mapped.items() if k in model.state_dict()}
        model.load_state_dict(filtered, strict=False)
        print(f"[MAE] loaded {len(filtered)} encoder tensors -> classifier")
        return model


def _cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """Warmup linear 0->1 for `warmup_steps`, then cosine decay to ~0."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


class MAEFinetuneLightning(L.LightningModule):
    """Linear (or full) fine-tune of a MAE-pretrained ViT on a classification task.

    Builds the standard classification ViT (``patch_embed -> cls -> +pos ->
    blocks -> norm -> head``) via the shared
    ``experiments/vit/vision_transformer.py`` and load encoder weights from a
    MAE pretrain checkpoint (``patch_embed`` + ``blocks`` + ``norm`` transfer;
    ``cls_token``/``pos_embed``/``head`` are learnt from scratch).

    Parameters
    ----------
    img_size : int
    patch_size : int
    in_channels : int
    num_classes : int
    embed_dim : int
    depth : int
    num_heads : int
        Classification ViT geometry — must match the pretrained encoder for the
        shared weights to load.
    pretrain_ckpt : str | None, default=None
        Path to a ``MAEPretrainLightning`` checkpoint. If provided, loads the
        encoder weights (``strict=False``).
    lr : float, default=3e-4
    weight_decay : float, default=0.05
    warmup_epochs : int, default=5
    max_epochs : int, default=100
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        pretrain_ckpt: str | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from vision_transformer import VisionTransformer

        self.model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ff_hidden_size=embed_dim * 4,
        )
        if pretrain_ckpt:
            self.model = MAEPretrainLightning.load_shared_state(pretrain_ckpt, self.model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train/cls_loss", loss)
        self.acc(logits, y)
        self.log("train/cls_acc", self.acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log("val/cls_loss", loss, on_step=False, on_epoch=True)
        self.log("val/cls_acc", self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
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
