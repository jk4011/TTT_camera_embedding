# NODE2 instruction (2026-08-05): stop music, switch to paper-table runs

Paste this whole file into the node2 Claude session.

---

## 1. STOP the music WAVE 3 runs

The user has cancelled music. WAVE 3 (T9-T12: `music_nope` / `music_rope` /
`music_honly_g1` / `music_hpra_g1`, launched 02:33 on gpu0-3) should not finish.

Reason, so this is not re-queued later: the music REMI 4-cell grid is **already
complete and is a total null** (nope 1.7085 / rope 1.7082 / honly 1.7151 /
hpra 1.7104, spread 0.0069 ppl = 0.4%), with the cause identified: REMI emits explicit
`Bar` and `Position` tokens, so metric position is readable as content and positional
addressing has nothing left to contribute. The user reviewed the TSD follow-up and
declined it twice ("음악은 하지 말아줘", and again on the TSD contrast). WAVE 3 fired
on its "4 GPUs free" gate without that decision having reached the node2 queue.

**Kill in this order. It matters.**

1. **Retry/self-heal parents FIRST.** If you kill the trainers first, the parents
   relaunch them and you will think the kill failed. Find them by walking the process
   tree, not by name matching.
2. Then the trainer processes.
3. Then remove the locks `lact_nvs/outputs/.gpu_locks/node2_gpu{0,1,2,3}`.
4. Then remove the music entries from whatever queue file drives the node (so a
   resubmission does not restart them).

**Do not use `pkill -f`.** It matches your own command line and has twice killed the
compound command issuing it, and sibling runs with it. Kill by PID or PGID only,
after listing exactly what you are about to kill.

Keep the partial checkpoints; do not delete `outputs/music_*`.

## 2. Then run these three, in order

All three fill blank cells in `paper_overleaf/experiment.md`. See
`EXPERIMENT_QUEUE_PAPER.md` at the repo root for the full list and rationale.

### P1. NVS NoPE baseline, seed 95 [1 GPU, ~1.6 h] : DO THIS FIRST

The single highest-value run available. It is both the fourth curve of Figure 1 and
the row every Table-1 delta is measured against, and **no SwiGLU NoPE checkpoint
survives anywhere** (node1's `lact_nvs/outputs/` has `mlp2_base` and `fw4l_base`, both
different architectures). Figure 1 is running on node1 right now and cannot be
completed without this.

Standard protocol, no deviations: `lact_l6_d256_p16`, no `cam.mode`, RE10K 256x256,
8 input + 8 target views, 30k iters, bs16, lr 1e-4, LPIPS loss from step 5k, seed 95.
Then evaluate: 256 held-out scenes, 8 uniform inputs / 4 midpoint targets.

RE10K is node-local and wiped on reset, so reshard first if `/tmp/re10k` is missing
(`data_preprocess/reshard_re10k.py`; test split alone is 54 GB and takes ~45 s).

After the checkpoint exists, also run the Figure-1 view sweep for this arm so all four
curves come from the same script:
`ARMS="base" VIEWS="2 4 8 16 24 32" bash lact_nvs/run_fig1_viewsweep.sh <gpu>`
(add a `base` entry to that script's `ARM_RUN` map pointing at the new run dir).

### P3. Video: input-only and Both [2 GPUs, 2 runs]

Table 6 has neither cell. **Naming trap:** in the plain-video runs `full` means
*hidden + learnable frequencies*, NOT input+hidden, so the existing `full` cell is not
the Both arm. That is why this is 2 runs and not 1.

Protocol must match F21/F22 exactly (Wan1.3B attention-only finetune, MultiCamVideo,
deterministic noise, 20k steps) or the new cells will not be comparable to the
existing base and hidden cells.

### P5. CCV site ablation [3 runs, ~46 h each, run them in PARALLEL]

**You were right to ask before starting. The arm labels in F30 were wrong.** Checked
against the config each run actually SAVED (`outputs/ccv_*/seed_1/config.yaml`), not
just the repo configs:

| run | cam_encoder | rotary sites | ladder |
|---|---|---|---|
| ccv_base | **ON** | none | . |
| ccv_pra | off | **input AND hidden** | learnable |
| ccv_pra_fixed | off | **input AND hidden** | fixed |
| ccv_both | **ON** | **input AND hidden** | learnable |

So `both` means **cam_encoder + rotary**, not input+hidden, and `pra` was never
input-only. Every rotary cell already has both sites. The consequence: F30's
"THE HIDDEN ROTARY EARNS IN VIDEO" does not follow from that grid, and its t=-9.0 is
the **cam_encoder** increment. Corrected in `RESULTS_DOSSIER.md` F30 and
`paper_overleaf/CLAUDE.md`.

**What to run (user decision 2026-08-05: FIXED ladder).** The headline CCV comparison is
`ccv_base` vs `ccv_both`, which are matched except for the rotary, both with the
cam_encoder ON. The site ablation must therefore live in the **cam_encoder ON** family,
not the OFF family, so that it mirrors NVS Table 1 (where NoPE also still receives the
camera, via ray maps):

| cell | cam_encoder | ttt_input_rope | ttt_hidden_rope | ttt_learnable_freqs | status |
|---|---|---|---|---|---|
| none | true | false | false | . | = `ccv_base`, exists |
| input | true | **true** | false | **false** | RUN |
| hidden | true | false | **true** | **false** | RUN |
| both | true | **true** | **true** | **false** | RUN |

Three runs, not two: `ccv_both` cannot serve as the "both" cell because it used
`ttt_learnable_freqs: true`, and the user has chosen the fixed ladder for headlines
(fixed also beats learnable in ccv: 0.04633 vs 0.04742).

Start from `abl_ccv_both.yaml` (cam_encoder already ON), set
`ttt_learnable_freqs: false`, and switch the two site flags per row. Keep everything
else identical: `cam_phase_mode: plucker`, same data, same seed, same step budget, and
evaluate at the same common checkpoint step F30 used so the numbers line up.

## 3. House rules

- **Report paired**, never unpaired: per-scene/per-pair deltas with a t-statistic,
  seed-matched, win-rate alongside the mean. Single-seed PSNR gaps of ~0.1-0.3 dB are
  init noise (F18).
- **PAPER FREEZE**: results go to `RESULTS_DOSSIER.md` / `lact_llm/ga_honly/LEDGER.md`
  only. Do not edit any `.tex`.
- **Launch as your own background Bash task** (`run_in_background: true`) so the
  harness notifies you; externally-started nohup daemons finish silently.
- GPU locks live on lustre and are shared between nodes: keep the `node2_` prefix.
- If a run finishes and the queue is empty, say so explicitly rather than leaving GPUs
  idle.

## 4. Slurm can stop this node at any time, and P3/P5 are 40-110 h runs

Checkpointing was NOT adequate for runs this long. Fixed on node1 before you start
(`git pull` to get it):

| config | save_every | exposure at ~10.3 s/step |
|---|---|---|
| `abl_video_*.yaml` | 4100 -> **250** | 11.7 h -> **43 min** |
| `abl_ccv_*.yaml` | 2000 -> **250** | 5.7 h -> **43 min** |

The video configs even carried the comment "only save near the end of the run", which
is the opposite of what a 110 h job needs.

`keep_last_iter` was also `1000000`, i.e. keep everything. One DCP checkpoint is about
7.3 GB (ccv_base holds 51 GB from 7 saves), so at the new interval a 20k-step run
would have written ~580 GB, and three runs 1.7 TB. Set to **1000**, which keeps the
4 most recent. Lustre has 153 TB free, so this is comfort not necessity, but 1.7 TB of
redundant checkpoints is still waste.

**Consequence you must handle:** pruning removes a step once the run moves past it. If
you want a specific step for an eval ladder (F30 used a common step 13999), **copy
that checkpoint aside** when it appears. Do not rely on it surviving.

Resume is automatic: `train.py` calls `find_latest_checkpoint` then `resume_job_dcp`
on the optimizer, scheduler and EMA params. Verify on the first restart that the step
number continues rather than restarting at 0, and say so in your report.

**Before launching, confirm each cell prints its resume line, and after the first
checkpoint interval confirm a checkpoint directory actually appeared.** A run that
silently never checkpoints looks identical to a healthy one until Slurm kills it.

## 5. Three failures node1 hit today, so you do not repeat them

1. **Port collisions.** Launchers that derive `--master_port` from a loop index give
   every per-cell invocation the same port, because the index is always 0. Two runs on
   one node then die with `EADDRINUSE`. Derive the port from something actually unique
   (GPU id, and the sweep variable if there is one).
2. **kmeans view selection has a floor.** `kmeans_input True` clusters the available
   frames into `num_input_views` groups, so it needs n_samples >= n_clusters. Asking
   for 4 input views dies with `n_samples=4 should be >= n_clusters=8`. Any view sweep
   has to start at the trained view count and go up.
3. **Log redirects that drop the sweep variable.** If every point of a sweep appends to
   one log file, the per-point checks (like grepping for `rotary VERIFIED ACTIVE`) look
   at a file that does not exist and silently pass. Put the sweep variable in the log
   filename too, not only in the output directory.

And the standing one: a cam checkpoint strict-loads into a stock model with zero
missing keys, so an eval that forgets to convert the model first reports a plausible
NULL with nothing in any log to indicate a problem. Always require a positive
`rotary VERIFIED ACTIVE` line.
