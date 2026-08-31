# linformer — сравнительный бенчмарк субквадратичных вниманий

Единый стенд для трёх семейств линейного/сублинейного внимания +
квадратичного baseline. Слои — из `spartan_torch`:

- `LinformerAttention` — *Linformer* (arXiv:2006.04768): low-rank проекция
  K/V с `n` на `k` токенов → O(nk);
- `LinearTransformerAttention` — *Linear Transformer / Transformers are RNNs*
  (arXiv:2006.16236): feature-map ядро `φ(Q)·φ(K)`, KV-продукт O(d²)
  не зависит от `n` → O(n);
- `ReformerAttention` — *Reformer* (arXiv:2001.04451): LSH-хэширование в
  бакеты, каждый запрос смотрит в окно `2·bucket_size` → O(n log n)
  (сортировка доминирует);
- `MultiHeadAttention` — baseline: классический (`use_sdpa=False`,
  материализует n×n скоры) и `use_sdpa=True` (SDPA/FlashAttention).

Все три эффективных слоя в bidirectional-режиме (`is_causal=False`) —
замеряем сложность, а не маскировку. Параметры Reformer — дефолты бумаги
(`n_hashes=8`, `bucket_size=64`, `n_buckets=128`), Linformer — `k=128`.

## Запуск

```bash
uv sync --extra dev --extra experiments
uv run jupyter lab experiments/linformer/benchmark.ipynb
```

Нужен CUDA-GPU: на больших `seq_len` ручной MHA упирается в память (n×n
скоры) и честно показывает квадратичный рост времени и памяти.

## Измеренные результаты (GPU, torch 2.13.0+cu132, d=512, H=8, k=128, batch=1)

Время (fwd+bwd) / пиковая память CUDA:

| модель | n=512 | n=2048 | n=8192 | n=16384 |
|---|---|---|---|---|
| Linformer (k=128) | 3.4 ms / 77 MiB | 4.3 ms / 105 MiB | 13.2 ms / 249 MiB | 24.3 ms / 441 MiB |
| Linear (feature map) | 2.7 ms / 74 MiB | 3.3 ms / 113 MiB | 11.9 ms / 269 MiB | 26.2 ms / 477 MiB |
| Reformer (LSH) | 11.4 ms / 236 MiB | 44.0 ms / 754 MiB | 215 ms / 2824 MiB | 6683 ms / 5585 MiB* |
| MHA (SDPA) | 2.0 ms / 77 MiB | 17.4 ms / 108 MiB | 258 ms / 228 MiB | 1007 ms / 389 MiB |
| MHA (manual n²) | 2.3 ms / 107 MiB | 16.2 ms / 599 MiB | 6718 ms / 8327 MiB | OOM |

\* n=16384 у Reformer превышает 4 GB VRAM: точка частично считалась из
shared-memory свопа, время (6683 ms) — не показатель чистой GPU-производительности.

Наклоны log-log «seq_len vs время / память» (все точки, для Reformer —
до 8192):

| модель | time-slope | mem-slope |
|---|---|---|
| Linformer (k=128) | 0.60 | 0.52 |
| Linear (feature map) | 0.69 | 0.54 |
| Reformer (LSH) | ~1.5 | 0.92 |
| MHA (SDPA) | 1.83 | 0.46 |
| MHA (manual n²) | 3.16 | 1.59 |

## Выводы

- **Квадратичный baseline**: время и память ручного MHA растут квадратично
  (наклон ~2–3, свыше 1.6 в обоих — завязан на малый batch/память); при
  n=16384 — OOM. SDPA/FlashAttention убирает квадрат по памяти (наклон 0.46
  — кэши и аллокатор), но время всё ещё ~1.8.
- **Linear Transformer** — самый дешёвый на длинных последовательностях:
  почти плоская память (KV-продукт d×d не зависит от n), наклон времени ~0.7.
  При n=8192 быстрее SDPA в ~22×, ручного MHA в ~560×.
- **Linformer** — того же порядка: при n=8192 быстрее SDPA в ~19×, ручного
  MHA в ~500×. Наклон < 1 — на малых n доминируют линейные проекции.
- **Reformer** — честная цена LSH: наклон времени ~1.5 (сортировка
  O(n log n)) и большой константный множитель из-за `n_hashes` раундов.
  До 8192 проигрывает обоим линейным слоям и даже SDPA по времени; окно
  `2·bucket_size` держит память линейной, но при 8 раундах она в разы выше,
  чем у Linear/Linformer. Выгода Reformer — не асимптотика на этом стенде,
  а каузальность без материализации полных скоров (см. тесты
  `tests/test_lsh_attention.py`).
- **Память эффективных слоёв**: Linformer/Linear растут как ~√n (проекции,
  аллокатор), Reformer — почти линейно (фиксированное окно на запрос).

Опорные цифры бумаг (V100): Linformer при n=65536, k=128 — ~20× быстрее и
~60× меньше памяти, чем стандартный transformer; Reformer — 12 слоёв на
1M-токенах на одном чипе 64 GB TPU.
