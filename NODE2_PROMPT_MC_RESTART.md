# NODE2 gpu3 (2026-08-05): restart the multi-chunk NVS sequence after `git pull`

`git pull` first (`961425a`), then restart `mc_*` from scratch.

---

## Why restart

The four multi-chunk runs must share identical code, and yours do not: `mc_base`
started before a set of rotary changes landed, and the remaining three would start
after. Discard `mc_base` and rerun all four.

Three changes landed while it was running. Two are harmless, one is not:

| change | bit-exact? |
|---|---|
| phase cache (24 blocks were rebuilding identical cos/sin) | yes |
| duplicate `silu(gate_before_act)` hoisted, 14 sites incl. the stock kernels | yes |
| **rotary fused with `torch.compile`** | **no, 1 ULP in bf16** |

The fusion is why: it reorders the rounding, so a run before it and a run after it are
not comparable at the precision our paired comparisons assume. It is worth keeping,
since it took the rotary from 50x its memory-bandwidth floor to 4.2x (1.892 ms to
0.160 ms at the tttLRM hidden shape), which is what made `Both` measurably slower than
`base` in the first place.

Note `mc_base` is the NoPE arm and has no rotary, but the silu hoist touches the stock
kernel too, so it is not exempt.

## What to run

Unchanged from the earlier prompt, except that all four now start from the same commit:

- 4 runs, one per arm (NoPE / input / hidden / Both), **sequential on gpu3**
- `ttt_num_chunks: [1, 2, 4, 8]` — one model per arm, drawing one n per forward
- **32 input views** fixed, so chunk sizes are 8192/4096/2048/1024, all above Muon's
  ~427-token amortisation point
- otherwise the standard protocol: RE10K 256x256, 30k iters, bs16, lr 1e-4, LPIPS from
  step 5k, seed 95
- then evaluate each finished model at n = 1, 2, 4, 8 and report
  **Delta(rotary - NoPE) at each n**, paired per scene with a t-statistic

Do not compare absolute PSNR across n: it falls as chunks shrink, for reasons unrelated
to the rotary.

## Verify before leaving it

1. `ttt_num_chunks` actually took effect. It is a plain config key, so a typo silently
   leaves you at n=1 and the run looks healthy. Confirm the op order has the number of
   update operators you expect.
2. A checkpoint appears at the usual interval.

## Context, so the result is read correctly

Node1 already has the evaluation-only half (F41): checkpoints trained with a single
update, evaluated with the update split n ways. Both's paired delta over NoPE went
+1.43 -> +3.06 -> +5.03 -> +4.76 for n = 1,2,4,8, because chunking collapses the NoPE
baseline (21.93 -> 14.30) while the rotary arms hold up. Your runs answer the other
half: whether the method also TRAINS well in that regime.
