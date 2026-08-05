# NODE2 gpu3 (2026-08-05): multi-chunk training, 4 NVS runs, sequential

`git pull` first: the code this needs landed in `3e07ad9`.

---

## What and why

TTT-RoPE is derived for a **single** update step: the phases cancel inside one inner
product, one gradient step. In practice, once the token count grows, the update is split
into n sequential chunks, each updating the fast weights on top of the previous chunk's
result. The derivation does not cover that, so the paper needs it shown experimentally.

Node1 already did the cheap half of this, **evaluation only** (F41 in
`RESULTS_DOSSIER.md`): the seed-95 checkpoints, all trained with one update, evaluated
with the update split n ways at 32 input views. Paired delta over NoPE:

| arm | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| input | +0.795 | +2.404 | +4.457 | +4.054 |
| hidden | +1.126 | +1.937 | +1.375 | +0.948 |
| **Both** | +1.433 | +3.055 | **+5.025** | +4.763 |

The rotary's value more than triples, because chunking collapses the NoPE baseline
(21.93 -> 14.30): each chunk writes over the previous fast weights and per-chunk
weight-norm decays the earlier ones, so without an address space the earlier chunks are
lost. Sequential updating is the regime where addressing matters most.

**What is still missing** is the training side. F41 shows learned phases *survive*
chunked application; it does not show the method *trains well* in the n-step regime.
That is this job.

## The runs

Four runs, one per arm, **sequential on gpu3**. One model per arm, not one per n:
`ttt_num_chunks` now accepts a list, and the model draws one n per forward, so a single
model learns to update under every chunk count.

```yaml
ttt_num_chunks: [1, 2, 4, 8]     # multi-chunk training
```

Everything else is the standard protocol except the view count:

- **32 input views** (fixed), 4 target views at eval
- RE10K 256x256, 30k iterations, bs16, lr 1e-4, LPIPS loss from step 5k, seed 95
- arms: NoPE / input RoPE / hidden RoPE / Both, i.e. the same four configs as
  `base_s95`, `pra_hi_s95`, `h_pra_hi_s95`, `pra_h_hi_s95`

32 views = 8192 update tokens, so chunk sizes are 8192 / 4096 / 2048 / 1024. All stay
above Muon's ~427-token amortisation point. Do **not** add n=16 or n=32: 256-token
chunks are F8's setting and lose 0.23 dB for a reason unrelated to addressing.

32 input views is 4x the tokens of the usual 8-view protocol, so budget roughly 6-8 h
per run rather than 1.6 h. Four sequential runs is a long job; check in rather than
assuming it finished.

## Evaluate each finished model at n = 1, 2, 4, 8

```bash
python eval.py --load outputs/<run>/model_0030000.pth --config <cfg> \
  --num_scenes 256 --num_input_views 32 --num_target_views 4 \
  --ttt_num_chunks <n> --out <run>_n<n>.json
```

Report **Delta(rotary - NoPE) at each n**, paired per scene with a t-statistic. Do not
compare absolute PSNR across n: absolute quality is expected to fall as chunks shrink,
for reasons that have nothing to do with the rotary.

The question is whether the F41 pattern survives training in this regime, and whether
the two sites still respond oppositely (input's delta grew with n, hidden's shrank).

## Two things to verify before leaving it running

1. `ttt_num_chunks` actually took effect. It is a plain config key, so a typo silently
   leaves you at n=1 and the run looks healthy. Confirm from the log that the op order
   has the number of update operators you expect.
2. A checkpoint appears at the usual interval.

## Note on the CCV cells on gpu0-2

If those are still the runs launched at `94b4fd6`, they have `ttt_hrope_frac` at its 0.5
default and need restarting at 1.0. See `NODE2_PROMPT_RESTART_CCV.md`. If you already
restarted them, ignore this.

## House rules

- Report paired, never unpaired; seed-matched; win-rate alongside the mean.
- PAPER FREEZE: results go to `RESULTS_DOSSIER.md`, not to any `.tex`.
- Keep the `node2_` prefix on GPU locks; they are shared across nodes on lustre.
- Launch as your own background task so the harness notifies you.
