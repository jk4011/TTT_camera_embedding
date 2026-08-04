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

### P5. CCV hidden-only [1 GPU, 1 run]

F30 ran base / pra(learnable) / pra_fixed / both, so hidden-only is the one hole in an
otherwise complete table. It matters because CCV is the only video-domain setting
where the hidden site earns (+3.9-4.6% over input), and the paper should show
hidden-alone there rather than inferring it. Same protocol as F30 (held-out val loss,
64 fixed pairs, EMA, common checkpoint step).

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
