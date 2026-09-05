# ViT experiments

Витрина Vision-Transformer экспериментов на pytorch-lightning + `spartan_torch`.
Модели собираются из примитивов библиотеки; тяжёлые трейн-фреймворки здесь лежат
в `experiments/` (ядро `spartan_torch` от них свободна).

## Состав

| Каталог | Что делает | Бейзлайн |
|---|---|---|
| `vit_cifar10/` | Классификационный ViT с нуля (CIFAR-10, ViT-Tiny) | — |
| `mae/` | MAE pretrain + fine-tune (CIFAR-10, ViT-Tiny) | `vit_cifar10` |
| `dino/` | DINO self-distillation + k-NN / linear probe (CIFAR-10, ViT-Tiny) | `vit_cifar10` |

## Как запускать

```bash
uv sync --extra dev --extra experiments
uv run jupyter lab experiments/vit
```

Ноутбуки самодостаточны (локально и в Colab) — качают данные в `data/`, чекпоинты
пишут в `checkpoints/` (оба пути в .gitignore). MLflow-логирование опционально:
если `localhost:5000` недоступен, логгер отключается автоматически.

## MAE (`mae/`)

Masked Autoencoder (arXiv:2111.06377) на примитивах `spartan_torch`
(`spartan_torch.masking.RandomPatchMasking`, `spartan_torch.vit.mae`,
`MaskedToken`).

- `mae_model.py` — чистый `torch.nn` ассемблер: асимметричный
  encoder-ViT **без** `[CLS]` и без mask-token'ов (работает только на видимых
  патчах) + лёгкий decoder + пер-патч реконструкция нормализованных пикселей.
- `mae_lightning.py` — трейн-луп: cosine + warmup, `MAEPretrainLightning` и
  `MAEFinetuneLightning` (перенос `patch_embed` + `blocks` + `norm` из MAE-encoder
  в классификационный ViT; `cls_token`/`pos_embed`/head обучаются с нуля).

```python
mae = MAEModel(img_size=32, patch_size=4, mask_ratio=0.75, ...)
pre = MAEPretrainLightning(mae, warmup_epochs=5, max_epochs=60)
ft = MAEFinetuneLightning(img_size=32, patch_size=4, num_classes=10,
                          embed_dim=192, depth=6, num_heads=3,
                          pretrain_ckpt=best_pretrain_ckpt, ...)
```

## DINO (`dino/`)

Self-Distillation with No Labels (arXiv:2104.14294) на примитивах `spartan_torch`
(`DINOProjectionHead`, `DINOLoss` + `Centering`/`Sharpening`, `MomentumEncoder`).

- `dino_model.py` — `ViTBackbone` (переиспользует `vision_transformer.py`, отдаёт
  `[CLS]`-фичу), `DINONet` (backbone + голова), `DINOLightning` (student/teacher,
  EMA 0.996 → 1.0 по косинусу, multi-crop, `embed()` для k-NN / linear probe).

```python
dino = DINOLightning(backbone=backbone, head_hidden_dim=512, head_out_dim=4096,
                     student_temp=0.1, teacher_temp=0.04, ...)
trainer.fit(dino, train_loader, val_dino_loader)   # multi-crop batches
feats = dino.embed(images)                          # teacher-фичи для eval
```

## Общая база

`vision_transformer.py` — классификационный ViT (timm-совместимый `state_dict`):
цель переноса весов для MAE fine-tune и backbone для DINO. `timm_compat/` — утилиты
сопоставления ключей времён экспериментов ImageNet.