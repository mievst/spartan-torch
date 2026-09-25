from .utils import WarmupScheduler

__all__ = ["WarmupScheduler", "OverallTQDMProgressBar", "OverallRichProgressBar"]

_LAZY_LIGHTNING = {
    "OverallTQDMProgressBar": ".lightning_progress",
    "OverallRichProgressBar": ".lightning_progress",
}


def __getattr__(name: str):
    # Lazy import: keeps `spartan_torch.utils` importable without the
    # `experiments` extra (`lightning`) installed.
    if name in _LAZY_LIGHTNING:
        import importlib

        module = importlib.import_module(_LAZY_LIGHTNING[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
