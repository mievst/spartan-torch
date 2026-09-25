"""Overall-training progress bars for PyTorch Lightning.

The default Lightning bars show batch progress *within* the current epoch
(plus a separate validation bar). The callbacks in this module add one extra
bar on top that tracks whole-training progress in epochs — how many epochs
are done and how many remain — with ETA and the same ``prog_bar=True``
metrics as the built-in bars. The built-in bars themselves are untouched.

Usage:
    from lightning.pytorch import Trainer
    from spartan_torch.utils.lightning_progress import OverallTQDMProgressBar

    trainer = Trainer(max_epochs=100, callbacks=[OverallTQDMProgressBar()])

    For a ``rich``-based display use ``OverallRichProgressBar`` instead.

This module requires the ``experiments`` extra (``lightning``). It is imported
on demand — ``spartan_torch`` and ``spartan_torch.utils`` stay importable
without Lightning installed.

References
----------
PyTorch Lightning progress-bar docs
(https://lightning.ai/docs/pytorch/stable/common/progress_bar.html):
subclassing ``TQDMProgressBar``/``RichProgressBar`` is the documented
extension point. No arXiv source: this is a training-UI utility with no
underlying paper (exception to the ``arXiv``-docstring rule, see AGENTS.md).
"""

import sys
from typing import Any, Optional

from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar

try:
    from lightning.pytorch.callbacks.progress.rich_progress import MetricsTextColumn
except ImportError:  # `rich` not installed; OverallRichProgressBar unusable then
    MetricsTextColumn = None  # type: ignore[assignment, misc]

if MetricsTextColumn is not None:
    from rich.text import Text

    class _OverallMetricsTextColumn(MetricsTextColumn):
        """Metrics column that also renders on the overall-epoch task.

        The stock column only renders metrics for the per-epoch train task;
        this subclass additionally renders the same metrics on the overall
        task so the top bar carries live metrics too.
        """

        def render(self, task: Any) -> Any:
            callback = self._trainer.progress_bar_callback
            overall_id = getattr(callback, "overall_task_id", None)
            if overall_id is not None and task.id == overall_id:
                if self._trainer.state.fn != "fit" or self._trainer.sanity_checking:
                    return Text()
                metrics = self._text_delimiter.join(self._generate_metrics_texts())
                return Text(metrics, justify="left", style=self._style)
            return super().render(task)


def _resolve_overall_total(trainer: Any) -> Optional[int]:
    """Epoch count for the overall bar, or None when it cannot be known."""
    max_epochs = trainer.max_epochs
    if max_epochs is None or max_epochs == -1 or max_epochs <= 0:
        return None
    return int(max_epochs)


class OverallTQDMProgressBar(TQDMProgressBar):
    """``TQDMProgressBar`` with an extra overall-epoch bar on top.

    The extra bar has ``total=max_epochs`` and advances once per finished
    epoch, so ``elapsed``/``remaining`` show epoch-level ETA and ``n/total``
    shows done/remaining epochs. Its postfix carries the same metrics as the
    per-epoch train bar (``prog_bar=True`` logs). All built-in bars keep
    their default format and metrics; only their terminal line positions are
    shifted down by one while the overall bar is displayed.

    Parameters
    ----------
    refresh_rate : int, default=1
        Same as in ``TQDMProgressBar``.
    process_position : int, default=0
        Same as in ``TQDMProgressBar``; the overall bar takes line
        ``2 * process_position``, the train bar moves one line down.
    leave : bool, default=False
        Same as in ``TQDMProgressBar`` (per-epoch bars). The overall bar
        always stays visible after training.
    overall_description : str, default="Overall"
        Description of the extra bar.
    """

    def __init__(
        self,
        refresh_rate: int = 1,
        process_position: int = 0,
        leave: bool = False,
        overall_description: str = "Overall",
    ) -> None:
        super().__init__(refresh_rate=refresh_rate, process_position=process_position, leave=leave)
        self._overall_description = overall_description
        self._overall_bar: Optional[Tqdm] = None
        self.overall_total: Optional[int] = None
        self.overall_n: int = 0

    def __getstate__(self) -> dict:
        # tqdm objects hold locks and cannot be pickled (mirrors base).
        state = super().__getstate__()
        state["_overall_bar"] = None
        return state

    @property
    def _overall_active(self) -> bool:
        bar = self._overall_bar
        return bar is not None and not bar.disable

    def _shift_down(self, bar: Tqdm) -> Tqdm:
        # tqdm renders on terminal line abs(bar.pos); base bars seat at
        # 2 * process_position, so move them one line down while the overall
        # bar occupies line 0 (pos is stored negated, hence -= 1).
        if self._overall_active and isinstance(getattr(bar, "pos", None), int):
            bar.pos -= 1
        return bar

    # -- bar constructors (mirror the base ones, shifted down one line) --

    def init_overall_tqdm(self, total: int, initial: int) -> Tqdm:
        """Create the overall-epoch bar. Override to customize it."""
        return Tqdm(
            desc=self._overall_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
            total=total,
            initial=initial,
        )

    def init_train_tqdm(self) -> Tqdm:
        return self._shift_down(super().init_train_tqdm())

    def init_validation_tqdm(self) -> Tqdm:
        return self._shift_down(super().init_validation_tqdm())

    def init_sanity_tqdm(self) -> Tqdm:
        return self._shift_down(super().init_sanity_tqdm())

    def init_test_tqdm(self) -> Tqdm:
        return self._shift_down(super().init_test_tqdm())

    def init_predict_tqdm(self) -> Tqdm:
        return self._shift_down(super().init_predict_tqdm())

    # -- lifecycle --

    def _ensure_overall_bar(self, trainer: Any) -> Optional[Tqdm]:
        if self.is_disabled:
            return None
        total = _resolve_overall_total(trainer)
        if total is None:
            return None
        if self._overall_bar is None:
            initial = max(0, min(trainer.current_epoch, total))
            self.overall_total = total
            self.overall_n = initial
            self._overall_bar = self.init_overall_tqdm(total, initial)
        return self._overall_bar

    def _refresh_overall_postfix(self, trainer: Any, pl_module: Any) -> None:
        bar = self._overall_bar
        if bar is None or bar.disable:
            return
        bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        # Overall first so the train bar created by super() seats below it.
        self._ensure_overall_bar(trainer)
        super().on_train_start(trainer, pl_module)

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        # Keep metrics on the overall bar live without advancing it.
        train_bar = self._train_progress_bar
        if (
            self._overall_bar is not None
            and train_bar is not None
            and self._should_update(batch_idx + 1, train_bar.total)
        ):
            self._refresh_overall_postfix(trainer, pl_module)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        bar = self._overall_bar
        if bar is None or bar.disable or self.overall_total is None:
            return
        bar.update(1)
        self.overall_n = min(self.overall_n + 1, self.overall_total)
        self._refresh_overall_postfix(trainer, pl_module)

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        super().on_validation_end(trainer, pl_module)
        self._refresh_overall_postfix(trainer, pl_module)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        super().on_train_end(trainer, pl_module)
        bar = self._overall_bar
        if bar is None or self.overall_total is None:
            return
        # Stopped mid-epoch (e.g. max_steps): settle on started epochs.
        done = min(max(trainer.current_epoch, self.overall_n), self.overall_total)
        if done > bar.n:
            bar.update(done - bar.n)
        self.overall_n = done
        self._refresh_overall_postfix(trainer, pl_module)
        bar.close()


class OverallRichProgressBar(RichProgressBar):
    """``RichProgressBar`` with an extra overall-epoch task on top.

    The extra task has ``total=max_epochs`` and advances once per finished
    epoch. Elapsed/remaining time columns give epoch-level ETA, the count
    column shows done/remaining epochs, and the metrics column shows the same
    live metrics as the per-epoch train bar. Built-in columns and metrics are
    otherwise unchanged.

    Parameters
    ----------
    refresh_rate : int, default=100
        Same as in ``RichProgressBar``.
    leave : bool, default=False
        Same as in ``RichProgressBar``.
    theme : RichProgressBarTheme
        Same as in ``RichProgressBar``.
    console_kwargs : dict, optional
        Same as in ``RichProgressBar``.
    overall_description : str, default="Overall"
        Description of the extra task.
    """

    def __init__(
        self,
        refresh_rate: int = 100,
        leave: bool = False,
        theme: Any = None,
        console_kwargs: Optional[dict] = None,
        overall_description: str = "Overall",
    ) -> None:
        from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

        super().__init__(
            refresh_rate=refresh_rate,
            leave=leave,
            theme=theme or RichProgressBarTheme(),
            console_kwargs=console_kwargs,
        )
        self._overall_description = overall_description
        self.overall_task_id: Optional[Any] = None
        self._overall_progress: Optional[Any] = None
        self.overall_total: Optional[int] = None
        self.overall_n: int = 0

    def _init_progress(self, trainer: Any) -> None:
        super()._init_progress(trainer)
        if (
            MetricsTextColumn is None
            or self.progress is None
            or self._metric_component is None
            or self.is_disabled
        ):
            return
        if isinstance(self._metric_component, _OverallMetricsTextColumn):
            return
        # Swap in a metrics column that also renders on the overall task;
        # metrics then stay live there via the regular _update_metrics path.
        fresh = _OverallMetricsTextColumn(
            trainer,
            self.theme.metrics,
            self.theme.metrics_text_delimiter,
            self.theme.metrics_format,
        )
        self._metric_component = fresh
        self.progress.columns = (*self.progress.columns[:-1], fresh)

    def _overall_task_valid(self) -> bool:
        if self.progress is None or self.overall_task_id is None:
            return False
        if self.progress is not self._overall_progress:
            return False
        try:
            self.progress.tasks[self.overall_task_id]
        except (IndexError, KeyError):
            return False
        return True

    def _ensure_overall_task(self, trainer: Any) -> None:
        if self.is_disabled or self.progress is None:
            return
        if self._overall_task_valid():
            return
        total = _resolve_overall_total(trainer)
        if total is None:
            return
        initial = max(0, min(trainer.current_epoch, total))
        self.overall_total = total
        self.overall_n = initial
        self.overall_task_id = self._add_task(total, self._overall_description, visible=True)
        self._overall_progress = self.progress
        self._sync_overall_task(trainer, initial)

    def _sync_overall_task(self, trainer: Any, done: int) -> None:
        if not self._overall_task_valid() or self.overall_total is None:
            return
        done = max(0, min(done, self.overall_total))
        self.overall_n = done
        assert self.progress is not None and self.overall_task_id is not None
        self.progress.update(self.overall_task_id, completed=done)
        self.refresh()

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        super().on_train_start(trainer, pl_module)
        # Progress exists now; the overall task is added first so it renders
        # above the per-epoch train task created in on_train_epoch_start.
        self._ensure_overall_task(trainer)

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        if self.is_disabled:
            return
        if self.train_progress_bar_id is not None and self._leave:
            # Base re-creates the whole Progress here; re-seat the overall
            # task first so it stays on top.
            self._stop_progress()
            self._init_progress(trainer)
            self._ensure_overall_task(trainer)
        super().on_train_epoch_start(trainer, pl_module)
        self._ensure_overall_task(trainer)
        self._sync_overall_task(trainer, trainer.current_epoch)

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self._ensure_overall_task(trainer)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self.overall_total is None:
            return
        self._sync_overall_task(trainer, trainer.current_epoch + 1)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        if self.overall_total is not None and self._overall_task_valid():
            done = min(max(trainer.current_epoch, self.overall_n), self.overall_total)
            self._sync_overall_task(trainer, done)
        self.refresh()
