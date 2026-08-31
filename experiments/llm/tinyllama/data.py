"""Data pipeline for the tinyllama experiment.

Reproduces TinyLlama's data recipe at small scale: a weighted mix of natural
language, code and math, streamed from HF ``datasets``:

* **NL** — ``Salesforce/wikitext`` (wikitext-103-raw-v1), the corpus the
  gpt2_rope experiment already trains on.
* **Code** — ``codeparrot/codeparrot-clean``, BigCode's deduplicated GitHub
  corpus. The paper uses StarCoder (``bigcode/starcoderdata``), which is now
  gated; codeparrot-clean is the same ecosystem and stays ungated.
* **Math** — ``open-web-math/open-web-math``, the source Proof Pile 2's
  ``openwebmath`` subset is built from (``EleutherAI/proof-pile-2`` itself no
  longer loads on ``datasets`` v3 — it ships a dataset script).

Per-stage mixes mirror the paper's v1.1 branches:

* ``pretrain`` — 70% NL : 30% code (paper: SlimPajama:StarCoder ≈ 7:3)
* ``continue_pretrain`` — 75% NL : 15% code : 10% math (paper's Math&Code
  branch: 75% SlimPajama, 15% StarCoder, 10% Proof Pile 2)
* ``cooldown`` — same mix as continue_pretrain; the stage only changes the
  batch size (paper: 1.8M -> 7.2M tokens).

Block caches (tokenized + packed, per source) and the tokenizer live in a
shared ``data/`` dir at the experiment root so all three stages reuse them —
only the weighted mix differs per stage.
"""

from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# source_key -> (dataset, config, text column, streaming)
SOURCES = {
    "nl": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text", False),
    "code": ("codeparrot/codeparrot-clean", None, "content", True),
    "math": ("open-web-math/open-web-math", None, "text", True),
}

# stage -> source weights (TinyLlama v1.1 branches)
MIXES = {
    "pretrain": {"nl": 0.70, "code": 0.30, "math": 0.00},
    "continue_pretrain": {"nl": 0.75, "code": 0.15, "math": 0.10},
    "cooldown": {"nl": 0.75, "code": 0.15, "math": 0.10},
}


def group_texts(examples: dict, block_size: int) -> dict:
    """Concatenate token batches and cut contiguous ``block_size`` blocks."""
    tokens = [i for row in examples["input_ids"] for i in row]
    n = len(tokens) // block_size * block_size
    return {"input_ids": [tokens[i : i + block_size] for i in range(0, n, block_size)]}


def iter_texts(
    source_key: str, n_examples: int, offset: int = 0
) -> list[str]:
    """Raw text strings from a source: ``offset`` skips, then ``take``."""
    name, config, col, stream = SOURCES[source_key]
    if config is not None:
        ds = load_dataset(name, config, split="train", streaming=stream)
    else:
        ds = load_dataset(name, split="train", streaming=stream)
    return [row[col] for row in ds.skip(offset).take(n_examples)]


def build_tokenizer(
    seed_texts: list[str], path: Path, vocab_size: int = 8192, min_frequency: int = 2
) -> PreTrainedTokenizerFast:
    """Train and persist a byte-level BPE on the mixed corpus (gpt2-style).

    The tokenizer is fit on a sample from all sources so code/math syntax is
    covered — an NL-only BPE breaks on code punctuation.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(seed_texts, trainer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast.save_pretrained(str(path))
    return fast


def load_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    """Load a ``PreTrainedTokenizerFast`` saved by :func:`build_tokenizer`."""
    return PreTrainedTokenizerFast.from_pretrained(str(path))

def tokenize_and_pack(
    texts: list[str], tokenizer: PreTrainedTokenizerFast, block_size: int
) -> Dataset:
    """Raw texts -> tokenized -> packed ``block_size`` blocks."""
    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(lambda b: tokenizer(b["text"]), batched=True, remove_columns=["text"])
    return ds.map(
        group_texts,
        batched=True,
        remove_columns=["input_ids", "attention_mask"],
        fn_kwargs={"block_size": block_size},
    )


def build_block_cache(
    source_key: str,
    cache_dir: Path,
    tokenizer: PreTrainedTokenizerFast,
    block_size: int,
    n_examples: int = 200000,
    offset: int = 0,
    max_blocks: int | None = None,
) -> Dataset:
    """Tokenized+packed blocks of one source, cached to ``cache_dir``.

    Idempotent: skips network/tokenization when the cache exists. ``offset``
    picks a different slice of a streaming source; ``max_blocks`` trims the
    cache so a huge corpus does not blow up disk.
    """
    path = Path(cache_dir) / source_key
    if path.exists():
        return Dataset.load_from_disk(str(path))
    print(f"[data] {source_key}: fetching {n_examples} examples (offset={offset})")
    texts = iter_texts(source_key, n_examples, offset)
    ds = tokenize_and_pack(texts, tokenizer, block_size)
    if max_blocks is not None:
        ds = ds.select(range(min(len(ds), max_blocks)))
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    print(f"[data] {source_key}: {len(ds):,} blocks cached -> {path}")
    return ds


def build_eval_dataset(
    tokenizer: PreTrainedTokenizerFast,
    block_size: int,
    cache_dir: Path | None = None,
) -> Dataset:
    """wikitext-103 validation split as the held-out eval set for ALL stages.

    A fixed eval set gives the cross-stage perplexity curve (paper's Fig. 3).
    Cached idempotently when ``cache_dir`` is given.
    """
    path = Path(cache_dir) / "eval" if cache_dir is not None else None
    if path is not None and path.exists():
        return Dataset.load_from_disk(str(path))
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    ds = ds.map(lambda b: tokenizer(b["text"]), batched=True, remove_columns=["text"])
    ds = ds.map(
        group_texts,
        batched=True,
        remove_columns=["input_ids", "attention_mask"],
        fn_kwargs={"block_size": block_size},
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(path))
    return ds


def make_mixed_dataset(
    mix_name: str,
    cache_dir: Path,
    tokenizer: PreTrainedTokenizerFast,
    block_size: int,
    total_blocks: int,
    seed: int = 0,
    n_examples: dict[str, int] | None = None,
    offsets: dict[str, int] | None = None,
    max_blocks: dict[str, int] | None = None,
) -> Dataset:
    """Concatenate source block caches into a weighted ``total_blocks`` mix.

    Per source ``round(weight * total_blocks)`` blocks are sampled *with
    replacement*, so a short cache (or a small ``n_examples`` cap) never
    breaks the ratio — it just repeats blocks. The concatenated blocks are
    then shuffled once.
    """
    n_examples = n_examples or {}
    offsets = offsets or {}
    max_blocks = max_blocks or {}
    weights = MIXES[mix_name]

    caches = {}
    for key, weight in weights.items():
        if weight == 0.0:
            continue
        caches[key] = build_block_cache(
            key,
            cache_dir,
            tokenizer,
            block_size,
            n_examples=n_examples.get(key, 200000),
            offset=offsets.get(key, 0),
            max_blocks=max_blocks.get(key),
        )

    rng = np.random.default_rng(seed)
    pieces = []
    for key, weight in weights.items():
        if weight == 0.0:
            continue
        n = round(weight * total_blocks)
        ds = caches[key]
        idx = rng.integers(0, len(ds), size=n)
        pieces.append(ds.select(idx.tolist()))
        print(f"[data] {mix_name}: {key} {n:,} blocks (weight {weight:.2f})")

    # concatenate_datasets + shuffle keep indices tables — no full copy of the
    # token data into Python lists, so large mixes stay light in RAM.
    mixed = concatenate_datasets(pieces).shuffle(seed=seed)
    print(f"[data] {mix_name}: {len(mixed):,} blocks total")
    return mixed
