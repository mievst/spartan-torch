# RESULTS

Parity и бенчмарки `spartan-torch`. Parity-таблица генерируется скриптом
(`uv run python scripts/verify_pretrained.py --write-results`), руками не править.
Бенч-таблица — скриптом `scripts/bench_attention.py --write-results`.

## Weight parity

Условия: CPU, fp32, eval, `torch.manual_seed(0)`, синтетический вход;
`pred agreement` — доля совпадающих argmax на том же входе (тест, что веса
легли 1-в-1). Gate в `tests/test_weight_parity.py`: `max abs diff < 1e-5`,
`cosine > 0.99999`.

Реестр воспроизведённых архитектур (сборки живут только в
`tests/test_weight_parity.py`, в `src` полных сетей нет по AGENTS.md):

- ViT-Base/16 ← `timm vit_base_patch16_224` — `CompatViT` (примитивы +
  `TransformerBlock(qkv_bias=True)`, `LayerNorm(eps=1e-6)` как в timm)
- ResNet-18 ← `torchvision ResNet18_Weights.IMAGENET1K_V1` — `CompatResNet18`
  (stem + `ResidualBlock`-стадии `[2,2,2,2]`)

<!-- parity:begin -->
| arch | source | max abs diff | cosine | pred agreement | published top-1 |
| --- | --- | --- | --- | --- | --- |
| ViT-Base/16 | timm vit_base_patch16_224 (pretrained) | 5.72e-06 | 0.99999994 | 1.0000 | 81.80 |
| ResNet-18 | torchvision ResNet18_Weights.IMAGENET1K_V1 | 0.00e+00 | 1.00000000 | 1.0000 | 69.76 |
<!-- parity:end -->

`published top-1` — числа с карточек моделей (ImageNet-val), для контекста, не gate.

## Attention benchmarks

Методология: CUDA, eval + `no_grad`, warmup + median, `synchronize`,
`reset_peak_memory_stats`/`max_memory_allocated`, фикс сид/batch/dtype.
Шапка окружения пишется в CSV рядом (`bench/results_attention.csv`).

<!-- bench:begin -->
env: torch 2.13.0+cu132 | NVIDIA GeForce RTX 3050 Ti Laptop GPU | cuda 13.2 | batch=8

| variant | seq 256 | seq 512 | seq 1024 | seq 2048 | seq 4096 | seq 8192 |
| --- | --- | --- | --- | --- | --- | --- |
| mha_manual | 2.74ms / 72.1MB | 6.99ms / 196.1MB | 20.40ms / 636.1MB | 68.89ms / 2284.1MB | SPILL | OOM |
| mha_sdpa | 2.13ms / 40.1MB | 5.58ms / 68.1MB | 15.65ms / 124.1MB | 51.88ms / 236.1MB | 192.88ms / 460.1MB | 760.48ms / 908.1MB |
| linformer | 3.90ms / 92.1MB | 6.17ms / 156.1MB | 11.91ms / 284.1MB | 23.37ms / 540.1MB | 47.03ms / 1052.1MB | 95.35ms / 2076.1MB |
| performer | 4.85ms / 82.5MB | 9.38ms / 152.9MB | 18.20ms / 292.2MB | 36.22ms / 575.2MB | 72.25ms / 1132.2MB | 144.80ms / 2252.2MB |
| linear | 3.31ms / 49.2MB | 4.77ms / 85.3MB | 9.03ms / 157.4MB | 17.70ms / 301.6MB | 35.40ms / 590.1MB | 71.33ms / 1167.1MB |
| reformer | 13.28ms / 357.1MB | 25.77ms / 702.1MB | 50.70ms / 1392.0MB | 102.41ms / 2771.8MB | SPILL | SPILL |
<!-- bench:end -->
