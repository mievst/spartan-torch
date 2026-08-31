"""HF Trainer helpers shared by the pretrain/continue_pretrain/cooldown stages.

Same helpers as the gpt2_rope experiment (``report_to`` detection, generate,
sample callback, perplexity eval), copied so this experiment stays
self-contained; the stage notebooks differ only in config.
"""

import math
import socket
from urllib.parse import urlparse

import mlflow
import torch
from model import CausalLM
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    GenerationConfig,
    TrainerCallback,
)


def report_to_value(tracking_uri: str, timeout: float = 2.0) -> str:
    """``report_to`` value for TrainingArguments: ``"mlflow"`` or ``"none"``.

    For remote URIs (http/https), probes the server with a short socket timeout.
    For local URIs (file paths, sqlite:///...), always returns ``"mlflow"``.
    """
    parsed = urlparse(tracking_uri)
    if parsed.scheme in ("http", "https"):
        try:
            with socket.create_connection(
                (parsed.hostname, parsed.port), timeout=timeout
            ):
                return "mlflow"
        except OSError:
            print(
                f"WARNING: MLflow недоступен ({tracking_uri}) — работаем без логгера"
            )
            return "none"
    return "mlflow"


@torch.no_grad()
def generate(
    model: CausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Sample ``max_new_tokens`` continuations after ``prompt`` (KV-cached)."""
    cfg = model.config
    ids = tokenizer(prompt, add_special_tokens=False).input_ids[-cfg.block_size :]
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    gen = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        pad_token_id=cfg.pad_token_id,
    )
    out = model.generate(input_ids, generation_config=gen)
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=True)


class SampleTextCallback(TrainerCallback):
    """Generate samples at each epoch end (print + MLflow artifact).

    transformers v5 callbacks receive no model handle, so the training model is
    passed in explicitly.
    """

    def __init__(
        self,
        model: CausalLM,
        tokenizer,
        prompts,
        max_new_tokens: int = 64,
        every_n_epochs: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        torch.manual_seed(0)
        text = ""
        for prompt in self.prompts:
            out = generate(self.model, self.tokenizer, prompt, max_new_tokens=self.max_new_tokens)
            text += f"\n### prompt\n{prompt}\n### completion\n{out}\n"
        print(text)
        try:
            if mlflow.active_run() is not None:
                mlflow.log_text(text, f"samples_epoch{epoch}.txt")
        except Exception as exc:  # non-fatal: logging is best-effort
            print(f"sample logging skipped: {exc}")


@torch.no_grad()
def evaluate_ppl(
    model: CausalLM, dataset, tokenizer, device: str = "cpu", batch_size: int = 8
) -> float:
    """Accurate perplexity: exp(mean NLL over all tokens of the dataset)."""
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    model.eval()
    total_nll, count = 0.0, 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        loss = model(input_ids=x, labels=y).loss
        n = (y != -100).sum().item()
        total_nll += loss.item() * n
        count += n
    return math.exp(total_nll / count) if count else float("inf")
