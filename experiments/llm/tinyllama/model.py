"""TinyLlama-style causal LM built from spartan_torch blocks.

Reproduces the TinyLlama recipe (Zhang et al., 2024, arXiv:2401.02385) at a
scale that fits ~4GB VRAM: pre-norm Llama-2 architecture with RoPE, RMSNorm,
SwiGLU and grouped-query attention in the paper's 8:1 query:kv-head ratio.

Default config: 206M params (d_model=1024, 18 layers, GQA 8:1, ff=2816).

Shared by all three pipeline stages (pretrain -> continue_pretrain ->
cooldown) — stages MUST build this exact architecture so checkpoints
transfer 1-to-1.

A full ``transformers`` model (:class:`PreTrainedModel` + :class:`GenerationMixin`):
``save_pretrained``/``from_pretrained`` give standard checkpoints
(``config.json`` + ``model.safetensors``) and ``model.generate`` replaces a
hand-rolled sampler.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

from spartan_torch import RMSNorm, RotaryPositionalEmbedding, SwiGLUFeedForward, TransformerBlock


class CheckpointedTransformerBlock(GradientCheckpointingLayer, TransformerBlock):
    """TransformerBlock with HF v5 gradient-checkpointing wired in.

    ``GradientCheckpointingLayer.__call__`` wraps the layer with
    ``torch.utils.checkpoint`` when ``model.gradient_checkpointing_enable()``
    was called and the model is training; otherwise it delegates to the
    regular ``nn.Module.__call__``. Hidden state must stay a positional arg.
    """


class TinyLlamaConfig(PretrainedConfig):
    """Hyperparameters of the tiny Llama-style LM (fits in ~4GB VRAM).

    Default 154M config mirrors TinyLlama's relative proportions: GQA with
    8 query heads per KV head (16:2), ``ff_hidden_size = 2.75 * d_model``
    (2816/1024), RoPE base 10000, pre-norm with RMSNorm, no dropout,
    tied embeddings.  Residual-branch init scaled by ``1/sqrt(2*n_layer)``.
    """

    model_type = "spartan_tinyllama"

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 1024,
        n_layer: int = 18,
        n_head: int = 16,
        num_kv_heads: int = 2,
        ff_hidden_size: int = 2816,
        block_size: int = 512,
        rms_norm_eps: float = 1e-5,
        rope_base: float = 10000.0,
        dropout_p: float = 0.0,  # Llama pretraining runs without dropout
        tie_word_embeddings: bool = True,
        use_cache: bool = True,  # KV cache on: RoPE rotates only the new token
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_kv_heads = num_kv_heads
        self.ff_hidden_size = ff_hidden_size
        self.block_size = block_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.dropout_p = dropout_p
        # transformers v5 PretrainedConfig drops unknown kwargs: use_cache is
        # no longer a base field, so set it explicitly or config.use_cache
        # raises AttributeError on the first forward call.
        self.use_cache = use_cache
        # from_pretrained reload passes the serialized num_hidden_layers back
        # through **kwargs; pass it through instead of hard-coding n_layer or
        # the config gets "multiple values for keyword argument" on reload.
        num_hidden_layers = kwargs.pop("num_hidden_layers", n_layer)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            # transformers generate() sizes the DynamicCache from
            # num_hidden_layers (default 5) — must mirror n_layer.
            num_hidden_layers=num_hidden_layers,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    @property
    def head_size(self) -> int:
        return self.d_model // self.n_head


class CausalLM(PreTrainedModel, GenerationMixin):
    """Llama-style decoder-only LM from library blocks.

    ``token embedding -> N TransformerBlock(is_causal=True, qk_mod=RoPE,
    norm_layer=RMSNorm, ff_layer=SwiGLUFeedForward) -> RMSNorm -> tied LM
    head``. Same composition contract as the gpt2_rope experiment, with the
    Llama-family norm/FFN swapped in.
    """

    config_class = TinyLlamaConfig
    _tied_weights_keys = {"lm_head.weight": "tok_emb.weight"}
    supports_gradient_checkpointing = True

    def __init__(self, config: TinyLlamaConfig):
        super().__init__(config)
        cfg = config

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryPositionalEmbedding(
            cfg.head_size, base=cfg.rope_base, max_seq_len=cfg.block_size
        )
        norm = lambda dim: RMSNorm(dim, eps=cfg.rms_norm_eps)
        self.blocks = nn.ModuleList(
            CheckpointedTransformerBlock(
                cfg.d_model,
                cfg.head_size,
                cfg.n_head,
                cfg.d_model,
                cfg.ff_hidden_size,
                num_kv_heads=cfg.num_kv_heads,
                is_causal=True,
                norm_layer=norm,
                attn_p=cfg.dropout_p,
                dropout_p=cfg.dropout_p,
                use_sdpa=True,
                qk_mod=self.rope,
                ff_layer=SwiGLUFeedForward(cfg.d_model, cfg.ff_hidden_size),
            )
            for _ in range(cfg.n_layer)
        )
        self.ln_f = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie manually: post_init's auto-tie runs inside super().__init__ before
        # these modules exist. `_tied_weights_keys` + config.tie_word_embeddings
        # let save_pretrained/from_pretrained keep the sharing 1-to-1.
        self.lm_head.weight = self.tok_emb.weight
        # Llama-style init: N(0, 0.02) everywhere; the residual branches are
        # scaled by 1/sqrt(2*n_layer) so the deep residual stream keeps unit
        # variance. post_init() is NOT called by PreTrainedModel.__init__ in
        # transformers v5, so we must call it (builds all_tied_weights_keys).
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        resid_scale = 1.0 / math.sqrt(2.0 * cfg.n_layer)
        for block in self.blocks:
            block.attn.out.weight.data.mul_(resid_scale)
            block.ff.down_proj.weight.data.mul_(resid_scale)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.tok_emb = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                past_key_values: Cache | tuple | None = None,
                use_cache: bool | None = None,
                cache_position: torch.Tensor | None = None,
                return_dict: bool | None = None):
        """Next-token predictions with optional KV cache.

        Parameters
        ----------
        input_ids : torch.Tensor
            ``(batch, seq_len)`` token ids. With ``past_key_values`` given this
            must be only the newly appended tokens (decode step); the cache
            supplies the prefix.
        labels : torch.Tensor | None, default=None
            ``(batch, seq_len)`` next-token ids; ``-100`` entries are ignored
            in the loss (SFT masks the prompt with them).
        past_key_values : Cache | tuple | None, default=None
            KV cache. A ``transformers`` ``Cache`` (as built by ``generate``)
            is updated in place per layer; a plain tuple of ``(k, v)`` pairs
            (one per block) is also accepted. ``None`` = full-context prefill.
        use_cache : bool | None, default=None
            Whether to return/update the KV cache. ``None`` falls back to
            ``config.use_cache``.
        return_dict : bool | None, default=None
            Return :class:`CausalLMOutput` (default) or a plain tuple.

        Returns
        -------
        ``CausalLMOutput(loss, logits, past_key_values)`` (or a plain tuple):
        ``loss`` is ``None`` when ``labels`` is ``None``; ``past_key_values``
        is ``None`` when ``use_cache`` is off.
        """
        use_cache = self.config.use_cache if use_cache is None else use_cache
        # KV cache + gradient checkpointing don't mix (the backward replay would
        # re-run the block and re-append to the cache) — same rule as HF models.
        if getattr(self, "gradient_checkpointing", False) and self.training:
            use_cache = False
        h = self.tok_emb(input_ids)
        caches: list = []
        for i, block in enumerate(self.blocks):
            past_kv = None
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    layer = past_key_values.layers[i]
                    past_kv = (layer.keys, layer.values) if layer.keys is not None else None
                else:
                    past_kv = past_key_values[i]
            h, cache = block(h, past_key_value=past_kv)
            if use_cache:
                if isinstance(past_key_values, Cache):
                    past_key_values.update(cache[0], cache[1], i)
                else:
                    caches.append(cache)
        h = self.ln_f(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # Standard HF CausalLM shift: position t predicts token t+1.
            # Labels come aligned with input_ids (from DataCollatorForLanguageModeling),
            # so the shift happens inside the model.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        if not use_cache:
            past = None
        elif isinstance(past_key_values, Cache):
            past = past_key_values
        else:
            past = tuple(caches) if caches else None

        if return_dict is None:
            return_dict = True

        if not return_dict:
            return tuple(v for v in (loss, logits, past) if v is not None)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
