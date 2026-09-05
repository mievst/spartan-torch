"""Headless CLI driver for the tinyllama stage chain (pretrain -> continue_pretrain -> cooldown).

Wraps the existing ``data.py`` / ``model.py`` / ``train.py`` modules behind a
single script so the full recipe can run unattended on a big GPU (A100 40GB)
and resume from any HF ``checkpoint-XXX``.

Paths are computed from ``__file__`` (not cwd) so the script is location
independent. A stage's checkpoints live in ``<stage-dir>/checkpoints``; the
next stage loads ``<prev-stage>/checkpoints/best`` by default.

Usage::

    python train_stage.py --stage pretrain [--max-steps 50]
    python train_stage.py --stage continue_pretrain --resume-from auto
    python train_stage.py --stage cooldown

On success a ``DONE`` marker is written next to the output dir so an external
chain driver can tell the stage completed cleanly.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parent  # tinyllama/
sys.path.insert(0, str(EXPERIMENT_ROOT))

DATA_DIR = EXPERIMENT_ROOT / "data"
SAMPLES = ["The history of Rome", "Machine learning is", "def fibonacci(n):"]

STAGES = {
    "pretrain": {
        "dir": "1.pretrain",
        "mix": "pretrain",
        "total_blocks": 975000,
        "batch_size": 32,
        "grad_accum": 4,
        "epochs": 3,
        "lr": 3e-4,
        "min_lr": 1e-5,
        "warmup_steps": 500,
        "eval_steps": 1000,
        "save_steps": 500,
        "mlflow_experiment": "tinyllama-pretrain",
        "init": "from_scratch",
    },
    "continue_pretrain": {
        "dir": "2.continue_pretrain",
        "mix": "continue_pretrain",
        "total_blocks": 250000,
        "batch_size": 32,
        "grad_accum": 1,
        "epochs": 3,
        "lr": 2e-4,
        "min_lr": 1e-5,
        "warmup_steps": 200,
        "eval_steps": 500,
        "save_steps": 250,
        "mlflow_experiment": "tinyllama-continue-pretrain",
        "init": "continue",
    },
    "cooldown": {
        "dir": "3.cooldown",
        "mix": "cooldown",
        "total_blocks": 125000,
        "batch_size": 32,
        "grad_accum": 4,
        "epochs": 3,
        "lr": 1e-5,
        "min_lr": 1e-6,
        "warmup_steps": 50,
        "eval_steps": 250,
        "save_steps": 100,
        "mlflow_experiment": "tinyllama-cooldown",
        "init": "continue",
    },
}

PREV_STAGE = {
    "pretrain": None,
    "continue_pretrain": "pretrain",
    "cooldown": "continue_pretrain",
}

BLOCK_SIZE = 512
SEED = 0


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Highest-numbered ``checkpoint-XXX`` dir under ``ckpt_dir``."""
    best = None
    best_n = -1
    if ckpt_dir.is_dir():
        for d in ckpt_dir.iterdir():
            m = re.fullmatch(r"checkpoint-(\d+)", d.name)
            if m:
                n = int(m.group(1))
                if n > best_n:
                    best_n, best = n, d
    return best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGES),
        help="which pipeline stage to run",
    )
    parser.add_argument(
        "--resume-from",
        default="auto",
        metavar="DIR|auto|none",
        help="resume from a checkpoint dir, 'auto' (latest checkpoint-XXX in "
        "the output dir) or 'none' (fresh start). Default: auto.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="cap training at N global steps (sanity). None = full epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where this stage writes checkpoints. Default: <stage-dir>/checkpoints",
    )
    parser.add_argument(
        "--pretrain-dir",
        type=Path,
        default=None,
        help="source checkpoint to load at stage start (continue stages). "
        "Default: <prev-stage>/checkpoints/best.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://host.docker.internal:5000",
        help="MLflow tracking URI. Default: http://host.docker.internal:5000",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = STAGES[args.stage]

    stage_root = EXPERIMENT_ROOT / cfg["dir"]
    ckpt_dir = args.output_dir or (stage_root / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    done_marker = stage_root / "DONE"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("medium")
    print(f"[train_stage] stage={args.stage} device={DEVICE} ckpt_dir={ckpt_dir}")

    import mlflow
    from data import (
        build_eval_dataset,
        load_tokenizer,
        make_mixed_dataset,
    )
    from model import CausalLM, TinyLlamaConfig
    from train import SampleTextCallback, report_to_value
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # --- tokenizer + datasets ---
    tokenizer = load_tokenizer(DATA_DIR / "tokenizer")
    train_ds = make_mixed_dataset(
        cfg["mix"],
        DATA_DIR / "blocks",
        tokenizer,
        BLOCK_SIZE,
        total_blocks=cfg["total_blocks"],
        seed=SEED,
    )
    val_ds = build_eval_dataset(tokenizer, BLOCK_SIZE, cache_dir=DATA_DIR / "blocks")
    print(
        f"[train_stage] vocab={tokenizer.vocab_size} | "
        f"train blocks={len(train_ds):,} | eval blocks={len(val_ds):,}"
    )

    # --- model init ---
    if cfg["init"] == "continue":
        pretrain_dir = args.pretrain_dir or (
            EXPERIMENT_ROOT / STAGES[PREV_STAGE[args.stage]]["dir"] / "checkpoints" / "best"
        )
        assert pretrain_dir.is_dir(), f"source checkpoint not found: {pretrain_dir}"
        model = CausalLM.from_pretrained(pretrain_dir)
        print(f"[train_stage] loaded source ckpt: {pretrain_dir}")
    else:
        cfg_kwargs = dict(
            vocab_size=tokenizer.vocab_size,
            d_model=1024,
            n_layer=18,
            n_head=16,
            num_kv_heads=2,
            ff_hidden_size=2816,
            block_size=BLOCK_SIZE,
            dropout_p=0.0,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = CausalLM(TinyLlamaConfig(**cfg_kwargs))

    # --- resume resolution ---
    resume = None
    if args.resume_from == "auto":
        resume = find_latest_checkpoint(ckpt_dir)
    elif args.resume_from != "none":
        resume = Path(args.resume_from)
    if resume is not None:
        if not resume.is_dir():
            print(f"[train_stage] ERROR: resume dir not found: {resume}")
            return 2
        print(f"[train_stage] resuming from {resume}")

    # --- MLflow ---
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(cfg["mlflow_experiment"])

    report_to = report_to_value(args.mlflow_uri)
    train_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=cfg["epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        weight_decay=0.1,
        warmup_steps=cfg["warmup_steps"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": cfg["min_lr"]},
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        bf16=(DEVICE == "cuda"),
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        torch_compile=(DEVICE == "cuda"),
        seed=SEED,
        report_to=report_to,
        run_name=cfg["mlflow_experiment"],
        dataloader_num_workers=6,
        dataloader_pin_memory=(DEVICE == "cuda"),
    )

    sample_cb = SampleTextCallback(
        model,
        tokenizer,
        prompts=SAMPLES,
        max_new_tokens=64,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[sample_cb],
    )

    trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)

    # finalize: best model + done marker
    trainer.save_model(str(ckpt_dir / "best"))
    done_marker.touch()
    print(f"[train_stage] stage {args.stage} complete. best -> {ckpt_dir / 'best'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
