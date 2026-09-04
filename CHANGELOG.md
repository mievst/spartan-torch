# Changelog

## Unreleased

- `MultiHeadAttention` / `TransformerBlock` / `CrossAttentionBlock`:
  `qkv_bias` / `out_bias` (загрузка biased QKV-весов timm/HF), `norm_mode="post"`
  в `TransformerBlock` (для ablation; дефолт `pre`).
- Новый слой `ALiBiBias` (+ `alibi_slopes`) — экстраполяция за длину обучения.
- `spartan_torch.compat`: key-mapping timm ViT / torchvision ResNet-18 →
  примитивы (полных сетей в `src` нет, сборки — в `tests/test_weight_parity.py`).
- Parity-сьют: `tests/test_weight_parity.py` (веса, gate `1e-5`/`0.99999`),
  `tests/test_sdpa_parity.py` (forward+backward vs `F.sdpa`); `RESULTS.md`
  генерируется `scripts/verify_pretrained.py`.
- Бенч-сьют: `scripts/bench_attention.py` (warmup+median, `synchronize`,
  SPILL-детект WDDM-пейджинга), `bench/` CSV + Pareto-график.
- Эксперименты: `experiments/positional/` (sin/learned/RoPE/ALiBi +
  экстраполяция), `experiments/norms/` (pre/post × LN/RMSNorm).
- Docstrings: `References` с arXiv-id у всех слоёв `src/`.
- CI: unit на PR, pretrained-parity по расписанию/лейблу, bench по dispatch
  (GPU-раннер). Маркеры pytest: `parity`, `pretrained`, `bench`.
