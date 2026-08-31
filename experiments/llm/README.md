# llm

Эксперименты по тренировке LLM, разбитые на **явные стадии пайплайна**.
Каждая стадия — отдельный ран со своим чекпойнтом; стадии связаны цепочкой
передачи весов.

Логика передачи: каждая следующая стадия стартует с лучшего чекпойнта
предыдущей через `CausalLM.from_pretrained`. Архитектура **обязана совпадать
1-в-1** между стадиями, иначе загрузка упадёт.

## Структура

```
llm/
  tinyllama/
    model.py        # Llama-стиль CausalLM: RoPE + RMSNorm + SwiGLU + GQA 8:1 (общий)
    data.py         # дата-рецепт 7:3 NL:code (+math), взвешенные миксы по стадиям
    train.py        # хелперы Trainer: SampleTextCallback, generate, evaluate_ppl
    data/           # общие кэши стадий: токенизатор + блоки по источникам
    1.pretrain/pretrain.ipynb
    2.continue_pretrain/continue_pretrain.ipynb
    3.cooldown/cooldown.ipynb
```

## TinyLlama (воспроизведение arXiv:2401.02385)

Llama-2 рецепт из блоков `spartan_torch`, 206M параметров, полный дата-рецепт
и трёхстадийный пайплайн v1.1.

### Архитектура (206M params)

Зеркалит пропорции TinyLlama: GQA 8:1 (16:2 головы), `ff_hidden_size = 2.75 * d_model`
(2816/1024), pre-norm, `RMSNorm`, `SwiGLUFeedForward`, RoPE base 10000, без dropout,
tied эмбеддинги. Residual-branch init `1/sqrt(2*n_layer)`.

| Параметр | Значение |
|---|---|
| d_model | 1024 |
| n_layer | 18 |
| n_head | 16 |
| num_kv_heads | 2 |
| ff_hidden_size | 2816 |
| vocab_size | 8192 |
| block_size | 512 |
| **Итого** | **~206M** |

### Данные (~500M токенов)

| Источник | В статье | У нас | Примечание |
|---|---|---|---|
| NL | SlimPajama | wikitext-103 | другой корпус |
| Code | StarCoder (gated) | codeparrot-clean | тот же экосистема |
| Math | Proof Pile 2 | open-web-math | proof-pile-2 не грузится v3 |

Миксы по стадиям (ветки v1.1):
- `1.pretrain` — 70% NL : 30% code, 975K блоков (~500M токенов)
- `2.continue_pretrain` — 75:15:10 NL:code:math, 250K блоков
- `3.cooldown` — тот же микс, `grad_accum` ×4, 125K блоков

### Трейнинг-оптимизации

| Оптимизация | Значение | Эффект |
|---|---|---|
| `bf16` | True | стабильнее чем fp16 |
| `gradient_checkpointing` | True | −60% activations → batch=32 |
| `max_grad_norm` | 1.0 | стабильность |
| `torch_compile` | True | +20-40% скорость (Triton) |
| `optim` | adamw_torch_fused | +10-15% скорость |

VRAM: ~2.75GB из 4GB (RTX 3050 Ti Laptop).

### Отклонения от статьи

| Параметр | У нас | Статья |
|---|---|---|
| vocab_size | 8192 | 32000 |
| d_model | 1024 | 2048 |
| n_layer | 18 | 22 |
| n_head | 16 | 32 |
| num_kv_heads | 2 | 4 |
| ff_hidden_size | 2816 | 5632 |
| block_size | 512 | 2048 |
| NL данные | wikitext-103 | SlimPajama |
| Code данные | codeparrot-clean | StarCoder |
| Токенов | ~500M | ~3T |
| Warmup | 500 | 2000 |
| min_lr | 1e-5 | ~0 |

### Стадии (цепочка чекпойнтов)

- `1.pretrain` — NL:code ≈ 70:30, с нуля
- `2.continue_pretrain` — 75:15:10 NL:code:math (ветка Math&Code)
- `3.cooldown` — тот же микс, `grad_accum` ×4, LR на минимуме

Сравниваем только ppl на wikitext-103 validation (фикс. eval-сет для всех
стадий, кривая как Рис. 3) и сэмплы текста/кода.

## Запуск

```bash
uv sync --extra dev --extra experiments
uv run jupyter lab
```

Открыть ноутбук из его папки (cwd ноутбука = папка стадии). Стадии идут
строго по цепочке `1.pretrain` → `2.continue_pretrain` → `3.cooldown`
(каждая init из `checkpoints/best` предыдущей). Первый запуск `1.pretrain`
качает wikitext-103 и стримит срезы codeparrot-clean / open-web-math и строит
кэши блоков в `tinyllama/data/`; стадии 2–3 их переиспользуют.

Без GPU всё тоже работает (структура пайплайна видна), но время трейна
неприемлемое — расчёт на ~4GB GPU (bf16).

## Нюансы

- Токенизатор тренируется на смешанном корпусе (NL + code + math) и живёт в
  общем `tinyllama/data/` — стадии его не пересобирают.
- Блоки-кэши (`tinyllama/data/blocks/{nl,code,math}`) токенизируются один
  раз, стадии только перекомбинируют микс (sampling с возвращением по весам
  стадии). Оценочный сет — фикс. wikitext-103 validation для всех стадий,
  чтобы кривая ppl была сопоставима между ними (как Рис. 3 в статье).
- `DataCollatorForLanguageModeling(mlm=False)`. Сдвиг target-токенов — внутри
  модели (стандарт HF CausalLM), данные его не делают.
- Шедулер: warmup + `cosine_with_min_lr` (нативный HF), лосс/чекпойнт по
  `eval_loss`, bf16 на CUDA, `gradient_checkpointing`, `adamw_torch_fused`, `torch_compile`.
- MLflow подключается через `report_to="mlflow"`; если сервер на
  `localhost:5000` недоступен — `report_to="none"`, эксперимент идёт без логов.
- Сэмплы текста печатаются на конец эпохи через `SampleTextCallback`.
