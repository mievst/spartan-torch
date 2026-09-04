# Norm placement × norm type ablation

Grid: `pre | post` × `layernorm | rmsnorm`, RoPE held constant, same
motif-LM task/seeds/steps (harness: `experiments/_tiny_lm.py`).

```bash
uv run python experiments/norms/norm_ablation.py [--steps 2000] [--n-layers 8]
```

## Results (`results_norm.csv`, 2000 steps, lr 3e-4, 8L d128, RoPE)

| layers | mode | norm | train loss | max grad norm | eval 64 |
| --- | --- | --- | --- | --- | --- |
| 8 | pre | layernorm | 0.3266 | 1.70 | 0.3269 |
| 8 | pre | rmsnorm | 0.3258 | 1.75 | 0.3272 |
| 8 | post | layernorm | 0.3265 | 3.86 | 0.3268 |
| 8 | post | rmsnorm | 0.3265 | 3.65 | 0.3268 |

Reading:

- Final loss is identical everywhere — placement/type don't change *what* this
  shallow setup converges to.
- Stability differs: `post` max grad norm is ~2.2x `pre` (3.7–3.9 vs 1.7)
  without warmup — the known post-norm spike regime, reproduced.
- `rmsnorm` ≈ `layernorm` on both loss and stability here.
- At 2 layers the grid is a null result (all cells ~0.33 loss, gnorm ~1.3–1.7):
  depth is required for the placement gap to open. Default stays `pre`.
