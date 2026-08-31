# image_classification

Репликации статей по классификации изображений.

## Конвенция структуры

Каждый эксперимент — самодостаточный ноутбук в своей папке:

```
image_classification/
  <model>/
    <model>_<dataset>.ipynb   # полный пайплайн: сборка + верификация + трейн
    data/                     # скачивается ноутбуком в свою папку (gitignored)
    checkpoints/              # ModelCheckpoint-ы (gitignored)
```

Правила:

- **Полная изоляция.** Данные и чекпойнты лежат внутри папки эксперимента,
  ничего общего между экспериментами нет.
- **Ноутбук запускается из своей папки.** Jupyter ставит cwd ноутбука = его
  директории, поэтому `Path.cwd() / "data"` — надёжный путь.
- **Ноутбук самодостаточный.** Сборка модели и весь трейн-луп внутри `.ipynb`.
  В библиотеку попадают только переиспользуемые примитивы (`ResidualBlock`,
  `WarmupScheduler` и т.д.) — не целиковые модели.
- **Трейн через pytorch-lightning**, логгирование в MLflow
  (`http://localhost:5000`), графики — inline через коллбэк.
- В git уходит только ноутбук (+ этот README). `data/`, `checkpoints/` уже
  в `.gitignore`.

## Эксперименты

| Эксперимент | Датасет | Что проверяет |
|---|---|---|
| [resnet18](resnet18/resnet18_cifar10.ipynb) | CIFAR-10 | `ResidualBlock` против torchvision, эталонный пайплайн классификации |
| [mobilenetv2](mobilenetv2/mobilenetv2_cifar10.ipynb) | CIFAR-10 | `InvertedResidual` (инвертированный residual + линейное бутылочное горлышко) против torchvision; трейн SGD+warmup+cosine (RMSProp из статьи §6.1 на CIFAR-10 нестабилен — val_acc застревает на random) |
