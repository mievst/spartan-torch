"""Shared MLflow logger helper for experiment notebooks.

Centralizes the ``_mlflow_reachable`` / ``make_logger`` snippet previously
copy-pasted across notebooks, and hardens experiment creation against the
classic ``RESOURCE_ALREADY_EXISTS`` crash: if the experiment name exists but
is soft-deleted on the tracking server (the usual cause — the lookup skips
deleted experiments while the unique constraint still holds the name), it is
restored instead of failing the training run.
"""

from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlparse


def mlflow_reachable(uri: str, timeout: float = 2.0) -> bool:
    """TCP-probe the tracking server (no HTTP cost, no auth needed)."""
    parsed = urlparse(uri)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_experiment(uri: str, name: str) -> str:
    """Get-or-create an experiment; restore it if soft-deleted. Returns the id."""
    from mlflow.exceptions import RestException
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(name)
    if exp is None:
        try:
            return client.create_experiment(name)
        except RestException as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(getattr(e, "error_code", e)):
                raise
            exp = client.get_experiment_by_name(name)
    if exp is None:  # pragma: no cover - server claims it exists but hides it
        raise RuntimeError(
            f"MLflow experiment {name!r} already exists on {uri} but is not "
            "visible (deleted?). Restore it in the MLflow UI or pick another name."
        )
    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        exp = client.get_experiment_by_name(name)
    return exp.experiment_id


def make_logger(uri: str, experiment_name: str, save_dir: str | Path, enabled: bool = True):
    """Build an ``MLFlowLogger`` or return ``None`` when tracking is off/unreachable.

    Parameters
    ----------
    uri : str
        Tracking URI, e.g. ``http://localhost:5000``.
    experiment_name : str
        Experiment to log into (created/restored as needed).
    save_dir : str | Path
        Local ``mlruns`` dir (offline fallback store).
    enabled : bool, default=True
        Master switch (notebooks pass ``False`` on Colab).
    """
    if not enabled:
        return None
    if not mlflow_reachable(uri):
        print(f"WARNING: MLflow unreachable ({uri}) — running without logger")
        return None
    ensure_experiment(uri, experiment_name)
    from lightning.pytorch.loggers import MLFlowLogger

    return MLFlowLogger(tracking_uri=uri, experiment_name=experiment_name, save_dir=str(save_dir))
