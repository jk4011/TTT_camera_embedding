# NODE2: CCV sampled-metrics eval (queued 2026-08-12, user-approved)

## What and why

The paper's CCV table currently reports held-out validation loss only. The user wants
generation-quality metrics next to it: for each F54 site-ablation cell, SAMPLE the
same 64 held-out pairs with fixed seeds and score the generated videos against the
MultiCamVideo ground truth with PSNR / SSIM / LPIPS, paired per pair. The sampled
eval machinery already exists (`minVid/eval_ccv_generate.py`, previously run at
n_pairs=8 on the older grid, dossier F30c); it has just been extended with `--start`
sharding and per-pair flush/crash-resume, and `run_ccv_gen_eval.sh` stripes the
4 cells x 2 shards over your GPUs.

Cells (all EMA weights at the common step 013999, same as the F54 val-loss table):
ccv_base, ccv_site_in, ccv_site_h, ccv_site_both. Checkpoints live in this repo's
`lact_ar_video/outputs/` (`_keep_step13999/` archive preferred; the script resolves
this itself). Dataset is read straight from lustre; NO /tmp resharding is needed.

## Steps

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope && git pull
cd lact_ar_video
# claim your GPUs in the lock dir first (see CLAUDE.md convention), then e.g.
./run_ccv_gen_eval.sh 0 1 2 3 4 5 6 7     # 8 GPUs: ~10 h wall clock
# fewer GPUs works too (jobs queue per lane): ./run_ccv_gen_eval.sh 0 1 2 3  # ~19 h
```

Launch as your own background Bash task (`setsid nohup ... < /dev/null &`) so it
survives session restarts; rerunning the script after any interruption RESUMES
(per-pair flush + skip of completed pairs; indices and seeds are global).

Runtime sanity: ~18 min per pair per GPU at the default 40 Euler steps. Each job
processes 32 pairs (~9.5 h). Logs: `outputs/eval_site/gen_ccv_*/gen_start*.log`
(one `[generate] pair N ...` block per pair with running PSNR/SSIM/LPIPS).

## Reporting

When all 8 jobs finish, compute per-cell means AND paired per-pair deltas vs
ccv_base from the union of `metrics_partial_*.json` (64 records per cell, keyed by
global `index`; pair identity matches across cells). Report:
- per cell: PSNR / SSIM / LPIPS means over the 64 pairs
- per rotary cell: paired delta vs ccv_base with t-stat and win count (n=64)
- flag any pair whose generation obviously failed (PSNR < 10) rather than silently
  averaging it in.

Append the table to RESULTS_DOSSIER.md (numbers only; interpretation is node1's),
commit, push. The paper is under FREEZE: do not edit paper_overleaf.

## Pitfalls

- /tmp is noexec on these nodes: the runner already exports repo-local triton /
  inductor caches; do not strip those exports.
- Do not switch the sampler to the `ar_*_inference` kernels: they support neither
  the SRC prefix nor the camera rotary. The full-sequence teacher-forcing sampler
  in eval_ccv_generate.py is the correct (training-identical) path.
- If a shard dies with CUDA OOM alongside other work, rerun it alone; resume makes
  this cheap.
