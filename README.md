# spartan-torch

Библиотека кастомных слоёв и вспомогательных заготовок для PyTorch: быстро
дообвешивать/адаптировать модели для файнтюна и собирать кастомные архитектуры
из кирпичиков.

## Что внутри

- **Кастомные слои и примитивы** — то, чего нет в `torch.nn`: слой за слоем из
  `spartan_torch.transformer` (внимания, блоки, нормы, позиционные эмбеддинги),
  `spartan_torch.cnn` (свёрточные блоки), `spartan_torch.vit` (элементы ViT).
- **Ready-made примеры** — `experiments/` содержит воспроизводимые пайплайны
  (видео-классификация на pytorch-lightning, крошечный TinyLlama на HF
  Trainer, бенчмарк субквадратичных вниманий).

Стандартные модели (ResNet, VGG, transformers и т.п.) здесь не переизобретаются —
их берём из `torchvision`/`huggingface` с предобученными весами. Библиотека
свободна от тяжёлых трейн-фреймворков: ядро зависит только от Torch.

## Установка

Требуется Python **3.13+** (и CUDA-совместимый Torch — см. ниже).

Прямо из репозитория:

```bash
pip install "spartan-torch @ git+https://github.com/mievst/spartan-torch.git"
```

Ядро ставится с `torch`/`torchvision` (из PyPI). Для экспериментов — полный
набор зависимостей:

```bash
pip install -e ".[experiments,dev]"
```

> **CUDA wheels (uv):** в `pyproject.toml` (`[tool.uv.sources]` / `[[tool.uv.index]]`)
> тач привязан к индексу `cu132` (`download.pytorch.org/whl/cu132`). `pip` из
> PyPI даст стандартную сборку torch; если нужен конкретный CUDA-релиз — ставь
> с явным `--index-url https://download.pytorch.org/whl/<cuda-version>`.

## Разработка

Зависимости и тесты — через [just](https://github.com/casey/just) + uv:

```bash
just sync   # установить/синхронизировать окружение (uv sync --extra dev --extra experiments)
just test   # прогнать тесты (venv-Python -m pytest)
just clean  # удалить .venv
```

> Не запускай `just sync` / `just clean`, пока живые Jupyter-ядеры держат
> `.pyd`/`.so` — uv не сможет завершить синк.

Разработка идёт в девконтейнере: `.devcontainer/` (CUDA-образ, GPU-проброс,
кэши HuggingFace/Weights & Biases монтируются с хоста).

## Лицензия

MIT — см. [LICENSE](LICENSE).
