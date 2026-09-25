"""Tests for spartan_torch.utils.lightning_progress (needs `lightning` extra)."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

pl = pytest.importorskip("lightning.pytorch", reason="needs lightning extra")

from spartan_torch.utils.lightning_progress import (  # noqa: E402
    OverallRichProgressBar,
    OverallTQDMProgressBar,
    _resolve_overall_total,
)


class _TinyLit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self.layer(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _loader():
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    return DataLoader(TensorDataset(x, y), batch_size=2)


def _trainer(bar, max_epochs=2, **kwargs):
    return pl.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=[bar],
        **kwargs,
    )


class TestResolveOverallTotal:
    def test_epochs(self):
        assert _resolve_overall_total(SimpleNamespace(max_epochs=100)) == 100

    @pytest.mark.parametrize("max_epochs", [None, -1, 0])
    def test_unknown_total(self, max_epochs):
        assert _resolve_overall_total(SimpleNamespace(max_epochs=max_epochs)) is None


class TestOverallTQDMProgressBar:
    def test_overall_advances_per_epoch(self):
        torch.manual_seed(0)
        bar = OverallTQDMProgressBar()
        trainer = _trainer(bar)
        trainer.fit(_TinyLit(), _loader())
        assert bar.overall_total == 2
        assert bar.overall_n == 2
        assert bar._overall_bar is not None
        assert bar._overall_bar.n == 2
        assert bar._overall_bar.total == 2

    def test_overall_carries_metrics(self):
        torch.manual_seed(0)
        bar = OverallTQDMProgressBar()
        trainer = _trainer(bar)
        model = _TinyLit()
        trainer.fit(model, _loader())
        # Same plumbing as the train bar: prog_bar logs must be present.
        assert "train_loss" in bar.get_metrics(trainer, model)
        assert "train_loss" in str(bar._overall_bar.postfix)

    def test_train_bar_shifted_below_overall(self):
        torch.manual_seed(0)
        bar = OverallTQDMProgressBar()
        trainer = _trainer(bar, max_epochs=1)
        trainer.fit(_TinyLit(), _loader())
        assert abs(bar._overall_bar.pos) == 0
        assert abs(bar.train_progress_bar.pos) == 1

    def test_disabled_bar_is_noop(self):
        torch.manual_seed(0)
        bar = OverallTQDMProgressBar(refresh_rate=0)
        trainer = _trainer(bar, max_epochs=1)
        trainer.fit(_TinyLit(), _loader())
        assert bar.overall_total is None
        assert bar._overall_bar is None


class TestOverallRichProgressBar:
    rich = pytest.importorskip("rich", reason="needs rich for RichProgressBar")

    def test_overall_task_completes(self):
        torch.manual_seed(0)
        bar = OverallRichProgressBar()
        trainer = _trainer(bar, max_epochs=2)
        trainer.fit(_TinyLit(), _loader())
        assert bar.overall_total == 2
        assert bar.overall_n == 2
        assert bar.overall_task_id is not None
        task = bar.progress.tasks[bar.overall_task_id]
        assert task.total == 2
        assert task.completed == 2
        assert "Overall" in task.description

    def test_overall_task_on_top(self):
        torch.manual_seed(0)
        bar = OverallRichProgressBar()
        trainer = _trainer(bar, max_epochs=1)
        trainer.fit(_TinyLit(), _loader())
        task_ids = [t.id for t in bar.progress.tasks]
        assert task_ids[0] == bar.overall_task_id


def test_core_importable_without_lightning():
    code = (
        "import importlib.abc, sys\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'lightning' or name.startswith('lightning.'):\n"
        "            raise ImportError('blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "assert 'lightning' not in sys.modules\n"
        "import spartan_torch\n"
        "import spartan_torch.utils\n"
        "from spartan_torch.utils import WarmupScheduler\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
