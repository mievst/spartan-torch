# Positional schemes: one task + length extrapolation

Task: repeating-motif next-token prediction (harness: `experiments/_tiny_lm.py`).
Random motif (`motif=7`) tiled to `seq_len`; model must copy from 6 positions
back — needs position info. Train at 64, eval at 64/128/256.

```bash
uv run python experiments/positional/pos_compare.py [--steps 2000]
```

## Results (`results_pos.csv`, 2000 steps, lr 3e-4, 2L d128)

| pos | train loss | eval 64 | eval 128 | eval 256 |
| --- | --- | --- | --- | --- |
| none | 1.1840 | 1.2249 | 1.1972 | 1.2739 |
| learned | 0.3268 | 0.3270 | n/a | n/a |
| sinusoidal | 0.3269 | 0.3269 | n/a | n/a |
| rope | 0.3281 | 0.3277 | 0.1736 | 0.1356 |
| alibi | 0.4402 | 0.4594 | 0.2510 | 0.1457 |

Reading:

- `none` never solves the task — control proving position info is required.
- `learned`/`sinusoidal` solve it at train length but raise past it
  (`LearnablePositionEmbedding`/`PositionalEncoding` are capped at `max_len`) —
  `n/a` IS the extrapolation result.
- `rope`/`alibi` extrapolate (no tables to overflow); loss even *drops* past
  train length — longer context holds more motif repeats, i.e. more evidence.
- `alibi` fits slower at train length (distance penalty starts dominant) but
  converges to the same extrapolated regime.
