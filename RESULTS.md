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
- LLaMA MLP ← `transformers LlamaMLP` — `SwiGLUFeedForward` (block-level,
  ключи 1-в-1, strict-load)
- LLaMA rotary ← `transformers LlamaRotaryEmbedding + apply_rotary_pos_emb` —
  `RotaryPositionalEmbedding` (math-level, весoв нет; включая offset-позиции
  для KV-cache decoding)
- MobileNetV2 `InvertedResidual(expand_ratio=6)` ← `torchvision` —
  block-level, включая slice `features[3]` из `IMAGENET1K_V1`. Границы:
  `expansion=1` у нас Identity (в tv реальный conv — задокументировано),
  stride-2 собирается с `use_skip=False` (в tv shortcut дропается, не
  проецируется)
- DINO `DINOHead` ← `facebookresearch/dino` — `DINOProjectionHead`
  (block-level, random-weights; `weight_g/v` → `parametrizations.original0/1`).
  Граница: паблишнутых весов головы нет (hub-чеки — только backbone), поэтому
  только офлайн-parity, без pretrained-ноги
- DINO ViT-S/16 backbone ← `dino_deitsmall16_pretrain.pth` (torch.hub) —
  `CompatViTSmall` (backbone-фичи, таблица ниже; k-NN 74.5 — число из hubconf)
- Mamba-1 `MambaMixer` ← `state-spaces/mamba-130m-hf` — `mixer[0]`
  (mixer-блок после norm; `published top-1` нет — у блока нет головы, `nan`)
- Mamba-1 `MambaMixer` ← `mamba-ssm` `Mamba` / HF `MambaMixer` —
  block-level, random-weights (офлайн; ssm-нога — skip без пакета).
  Граница: префилл побитово с референсом; чанкед/декод несёт ~1e-4 дрейф
  (порядок суммирования conv × усиление рекуррентностью — у самого HF
  префилл-vs-декод ~9e-4, см. docstring `MambaMixer`)
- Mamba-2 `Mamba2Mixer` ← `AntonV/mamba2-130m-hf` — `mixer[0]`
  (нативный HF-чек; для official `.bin` те же веса, группы `G=1, N=128` —
  подтверждено конфигом конверсии). Офлайн-ноги: HF random-weights,
  `mamba-ssm` `Mamba2` (skip без пакета). `published top-1` нет — `nan`
- Mamba-3 `Mamba3Mixer` ← `ib-ssm/mamba3-370M-10BT` — `mixer[0]`
  (настоящие веса, strict-load; таблицы нет — runnable-референса нет:
  официальные ядра только Linux/CUDA, в `transformers` Mamba-3 нет).
  Покрытие: key-map на реальных весах + step-vs-prefill self-согласие
  (`3.34e-06` на этом чеке). Офлайн-нога `mamba-ssm` `Mamba3` — skip без
  пакета. Недоказуемые допущения (pairing SISO-RoPE, bias-до-ротации,
  unfused==fused) — в docstring `Mamba3Mixer`

<!-- parity:begin -->
| arch | source | max abs diff | cosine | pred agreement | published top-1 |
| --- | --- | --- | --- | --- | --- |
| ViT-Base/16 | timm vit_base_patch16_224 (pretrained) | 1.00e-05 | 0.99999988 | 1.0000 | 81.80 |
| ResNet-18 | torchvision ResNet18_Weights.IMAGENET1K_V1 | 0.00e+00 | 1.00000000 | 1.0000 | 69.76 |
| DINO ViT-S/16 (backbone) | dino_deitsmall16_pretrain.pth (torch.hub) | 0.00e+00 | 0.99999982 | 1.0000 | 74.50 |
| Mamba-130m (mixer[0]) | state-spaces/mamba-130m-hf | 0.00e+00 | 1.00000000 | 1.0000 | nan |
| Mamba2-130m (mixer[0]) | AntonV/mamba2-130m-hf | 0.00e+00 | 0.99999988 | 1.0000 | nan |
<!-- parity:end -->

`published top-1` — числа с карточек моделей (ImageNet-val), для контекста, не gate.

## Attention benchmarks

Методология: CUDA, eval + `no_grad`, warmup + median, `synchronize`,
`reset_peak_memory_stats`/`max_memory_allocated`, фикс сид/batch/dtype.
Шапка окружения пишется в CSV рядом (`bench/results_attention.csv`).

<!-- bench:begin -->
env: torch 2.13.0+cu132 | NVIDIA GeForce RTX 3050 Ti Laptop GPU | cuda 13.2 | batch=8

**prefill** (latency ms / peak mem)
| variant | seq 256 | seq 512 | seq 1024 | seq 2048 | seq 4096 | seq 8192 | seq 16384 | seq 32768 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mha_manual | 2.50ms / 68.1MB | 6.84ms / 188.1MB | 20.25ms / 620.1MB | 68.38ms / 2252.1MB | SPILL | OOM | OOM | OOM |
| mha_sdpa | 1.99ms / 36.1MB | 5.49ms / 60.1MB | 15.71ms / 108.1MB | 52.01ms / 204.1MB | 195.30ms / 396.1MB | 767.02ms / 780.1MB | 3078.40ms / 1548.1MB | 9854.88ms / 3084.1MB |
| linformer | 3.02ms / 112.1MB | 6.04ms / 172.1MB | 11.50ms / 292.1MB | 23.01ms / 532.1MB | 46.73ms / 1012.1MB | 95.21ms / 1972.1MB | 1175.66ms / 3892.1MB | SPILL |
| performer | 4.76ms / 78.2MB | 9.44ms / 144.2MB | 18.62ms / 276.2MB | 36.87ms / 540.2MB | 74.01ms / 1068.2MB | 147.45ms / 2124.2MB | SPILL | SPILL |
| linear | 1.90ms / 45.2MB | 4.80ms / 77.3MB | 8.81ms / 141.4MB | 17.92ms / 269.6MB | 34.64ms / 526.1MB | 70.17ms / 1039.1MB | 155.25ms / 2065.1MB | SPILL |
| reformer | 11.78ms / 353.1MB | 22.92ms / 694.1MB | 45.86ms / 1376.0MB | 90.45ms / 2739.8MB | SPILL | SPILL | OOM | OOM |
| mamba1 | 79.40ms / 83.7MB | 135.88ms / 152.2MB | 283.29ms / 289.2MB | 538.50ms / 563.2MB | 1150.53ms / 1111.2MB | 2361.43ms / 2207.2MB | SPILL | SPILL |
| mamba2 | 34.76ms / 669.4MB | 70.30ms / 1318.2MB | 509.57ms / 2615.8MB | SPILL | OOM | OOM | OOM | OOM |
| mamba3_siso | 113.94ms / 85.0MB | 224.36ms / 151.5MB | 448.55ms / 282.5MB | 840.44ms / 546.5MB | 1721.56ms / 1074.5MB | 3983.82ms / 2130.5MB | SPILL | SPILL |
| mamba3_mimo | 197.11ms / 255.1MB | 394.51ms / 485.1MB | 806.75ms / 942.1MB | 1619.10ms / 1858.1MB | 6451.66ms / 3690.1MB | SPILL | ? | ? |

**decode** (latency ms/step / peak mem)
| variant | seq 256 | seq 512 | seq 1024 | seq 2048 | seq 4096 | seq 8192 | seq 16384 | seq 32768 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mha_manual | 0.61ms / 34.8MB | 0.72ms / 54.9MB | 1.19ms / 95.2MB | 2.14ms / 175.7MB | 4.04ms / 336.7MB | OOM | OOM | OOM |
| mha_sdpa | 0.65ms / 34.7MB | 1.11ms / 54.7MB | 1.64ms / 94.7MB | 3.07ms / 174.7MB | 5.69ms / 334.7MB | 11.10ms / 654.7MB | 21.76ms / 1294.8MB | 680.36ms / 2574.9MB |
| mamba1 | 1.21ms / 22.2MB | 1.23ms / 26.2MB | 1.25ms / 34.2MB | 1.26ms / 50.2MB | 1.29ms / 82.2MB | 1.22ms / 146.2MB | 1.20ms / 274.2MB | 1.71ms / 530.2MB |
| mamba2 | 1.62ms / 31.4MB | 1.76ms / 35.4MB | 1.56ms / 43.4MB | 1.45ms / 59.4MB | 1.85ms / 91.4MB | OOM | OOM | OOM |
| mamba3_siso | 1.99ms / 36.7MB | 1.99ms / 40.7MB | 1.90ms / 48.7MB | 1.96ms / 64.7MB | 1.94ms / 96.7MB | 1.92ms / 160.7MB | 2.24ms / 288.7MB | 2.29ms / 544.7MB |
| mamba3_mimo | 2.71ms / 71.8MB | 3.68ms / 75.8MB | 3.44ms / 83.8MB | 2.56ms / 99.8MB | 2.35ms / 131.8MB | 2.45ms / 195.8MB | 16.43ms / 323.8MB | OOM |
<!-- bench:end -->
