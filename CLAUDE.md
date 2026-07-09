# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research goal

This is a research repo. The objective (see `instruction.md`, written in Korean) is to develop a way to
**inject camera-pose information into LaCT / TTT layers** for novel view synthesis (NVS), analogous to how
PRoPE and RayRoPE inject camera info into *attention*. TTT (test-time training) replaces attention with a
linear-time fast-weight update, so relative positional-encoding tricks that rely on the attention Q·Kᵀ
matrix do not transfer directly — finding an equivalent for TTT is the open problem.

The intended workflow: brainstorm ~10 hypotheses for camera embedding in TTT, then implement and benchmark
them against the **LaCT LVSM baseline** in `lact_nvs/`, comparing PSNR/LPIPS. Hardware is **8× B200**
(since the 2026-07-09 batch-node move; was 4), so up to 8 experiments run concurrently.

**This research is well underway** — do not re-brainstorm from scratch. Dozens of variants have been
implemented (`lact_nvs/lact_ttt_cam.py`) and evaluated. The current state lives in the tracking docs at
the repo root (read these first before proposing or running anything):
- **`RESULTS_DOSSIER.md`** — every evaluated run with PSNR/LPIPS and paired stats; the source of truth
  for "what works". The headline finding is TTT-RoPE (Plücker rotary applied to fast-weight Q/K), currently
  ~+1.7 dB over baseline.
- **`EXPERIMENT_QUEUE.md`** — pending/running/done experiments in priority order, with GPU assignments.
- **`IDEAS.md`** — the synthesized hypothesis list. **`WRITING_BRIEF.md`** — the paper's central theory
  (inner-product addressing lemma) and terminology rules. **`CAMCTRL_DESIGN.md`** / **`IMPL_SPEC_CCV.md`** —
  the camera-controlled video-gen cross-task validation. **`paper_draft/`** — the LaTeX method draft.

## Repository layout

- **`lact_nvs/`** — the working codebase. LaCT (Large-Chunk TTT) applied to LVSM for NVS. **This is what you
  modify and experiment on.** Everything below under "Architecture" describes this directory.
- **`prope/`** — reference implementation of PRoPE ("Cameras as Relative Positional Encoding"), an
  *attention-based* camera conditioning method. Read-only reference for ideas; do not build the experiments here.
- **`RayRoPE/`** — reference implementation of RayRoPE (depth/3D-point-based ray positional encoding for
  multi-view attention). Also read-only reference. Its core is just `pos_enc/rayrope.py` + `pos_enc/utils/rayrope_mha.py`.
- **`minimal_implementations/`** — standalone, pedagogical LaCT layers (`bidirectional_lact_layer.py`,
  `causal_lact_with_sliding_window_attn.py`). Best starting point for understanding the TTT update math.
- **`papers/`** — PDFs *and* LaTeX source (so equations are readable) for LaCT, LVSM, PRoPE, RayRoPE, TTT,
  SwiGLU, DPPE.
- **`lact_llm/`, `lact_ar_video/`** — the same TTT-RoPE idea transplanted to language modeling and
  camera-controlled AR video generation, to show the method is not NVS-specific (cross-task evidence for the
  paper). Imported from the upstream LaCT repo; read-only unless running those specific ablations.

## Architecture (`lact_nvs/`)

Data flow: multi-view posed images → ray-map + image tokens → patchify → stack of `Block`s → decode target
view tokens to RGB. **Camera information currently enters the model only through the input ray maps** — this is
the key fact for the research task, since the TTT layer itself has no notion of pose.

- **`model.py`**
  - `LaCTLVSM` — top-level model. `compute_rays()` turns `fxfycxcy` (intrinsics) + `c2w` (extrinsics) into
    per-pixel `ray_o`, `ray_d`, and `o_cross_d` (the Plücker cross term). These 3 pose maps (+ normalized RGB
    for input views) are concatenated channel-wise, patchified, and linearly projected to `dim`. Target views
    contribute pose-only tokens (RGB zeroed).
  - Three entry points share the same blocks: `forward()` (training: update fast weights on input tokens, then
    apply to input+target), `reconstruct()` + `rendering()` (inference: save per-block fast-weight states from
    input views, then apply them to each target view separately).
  - `Block` — a residual stack of sub-modules whose *types and order come from the YAML* `block_config`
    (constructed via `get_class_by_name`, so config strings like `"lact_ttt.FastWeightGluMLPMultihead"` are
    import paths). Each sub-module has a `length_dim`:
    - `"l"` → reshaped to operate **within a single image** (e.g. `SelfAttention` over that view's patches).
    - `"vl"` → operates **across all views' tokens at once** (the TTT layer and MLP). Cross-view / camera
      reasoning happens here.
  - `SelfAttention`, `MLP` are standard; the TTT layer lives in `lact_ttt.py`.

- **`lact_ttt.py`** — the LaCT test-time-training layer.
  - `FastWeightGluMLPMultihead` — fast weights `w0, w1, w2` form a per-head **SwiGLU MLP** that is updated at
    test time. Q/K/V come from `to_qkv` (silu-gated); Q,K are L2-normalized per head. Per-token learning
    rates `lr0/lr1/lr2` are predicted by `lr_fc` (softplus around `base_lr`).
  - `fast_weight_swish_glu_weight_norm_mini_batch_apply` — the core update. For each `TTTOperator` it (a) if
    `update`: computes the manual SwiGLU backward, forms gradients, orthogonalizes them with
    `zeropower_via_newtonschulz5` (Muon, `muon_update_steps` iterations), applies them, then re-normalizes each
    fast weight to its original per-column norm (weight-norm); (b) if `apply`: runs the (possibly updated)
    SwiGLU on the query tokens.
  - `TTTOperator(start, end, update, apply)` (a namedtuple) is the control primitive: it selects a token range
    and says whether to update fast weights and/or emit output on it. The `ttt_op_order` list built in
    `model.py` is what distinguishes train / reconstruct / render.
  - Fast-weight state is threaded between blocks via the `info` dict (`{"w0","w1","w2"}` in the return value).

- **`lact_ttt_cam.py`** — **the camera-embedding research layer** (the main deliverable; `lact_ttt.py` is the
  untouched baseline). `CamFastWeightGluMLPMultihead` subclasses `FastWeightGluMLPMultihead` and adds camera
  conditioning selected by a single **`cam_mode` string** in the YAML. `cam_mode` is a `+`-joined set of
  feature flags (e.g. `"qk_rope_cam"`, `"pra_sinc"`, `"h_pra+ms2"`, `"fw3l_rot3"`); the `known` set in
  `__init__` is the authoritative list of implemented variants, with assertions enforcing which combos are
  legal (one rotary family at a time, one hidden rotary at a time, `fw3l*` standalone, etc.). The flagship is
  `qk_rope_cam` = **TTT-RoPE**: rotate the fast-weight Q/K by Plücker-coordinate phases (a log-spaced `omega`
  ladder × learnable `freq_gain`) so relative camera pose enters through the inner-product addressing the
  fast weights already use. Other families extend this to the hidden address space (`h_pra`), to ray
  segments (`pra_sinc`, sinc-integrated), to deeper 3-layer fast weights (`fw3l`), etc. Adding a variant =
  add a flag to `known`, its params in `__init__`, and its rotary/lr logic in the apply helpers +
  `forward`.
- **`eval.py`** — computes PSNR/LPIPS on held-out RE10K scenes for a trained checkpoint (the number that goes
  into `RESULTS_DOSSIER.md`). `data_re10k.py` is the RE10K-specific dataset loader used for these experiments.

- **`data.py`** — `NVSDataset` reads a JSON that lists per-datapoint JSONs; each holds per-image
  `fx,fy,cx,cy`, `w2c`, `file_path`. `resize_and_crop` adjusts intrinsics to match the resize+center-crop.
  `scene_pose_normalize` (scene data) re-centers/scales cameras via `normalize_with_mean_pose`.

## Commands (run from inside `lact_nvs/`)

```bash
pip install -r requirements.txt      # torch>=2.4, einops, lpips, omegaconf, transformers, huggingface-hub

# Training (multi-GPU). --actckpt (activation checkpointing on Block) is effectively required.
torchrun --nproc_per_node=8 --standalone train.py --config config/lact_l14_d768_ttt2x.yaml --actckpt

# Object-level inference
python inference.py --load weight/obj_res512.pt --image_size 512 512 \
  --data_path data_example/gso_sample_data_path.json

# Scene-level inference (adds --scene_inference, which sorts views and normalizes poses)
python inference.py --load weight/scene_res256x256.pt --config config/lact_l24_d768_ttt2x.yaml \
  --image_size 256 256 --scene_inference --num_all_views 136 --num_input_views 128 \
  --data_path data_example/dl3dv_sample_data_path.json
```

There is no test suite in `lact_nvs/`. (The reference `prope/` dir has `pytest tests/`.)

Configs (`config/*.yaml`) set `patch_size`, `dim`, `layers`, and the per-block `block_config`. `ttt2x`/`ttt4x`
refer to the TTT `inter_multi` / training scale; `l14`/`l24` is the layer count.

### Camera-embedding experiment workflow

- **Config naming:** `config/cam_*.yaml` are the camera-embedding experiments — each pins a `cam_mode`
  string (see `lact_ttt_cam.py`) onto the small model. The rest (`lact_l*`, `lact_l6_d256_p16`) are the
  stock LaCT baselines.
- **Standard ablation protocol** (must match, or a result isn't comparable — see `RESULTS_DOSSIER.md`
  header): small config **L6 / d256 / patch 16**, RE10K 256×256, 8 input + 8 target views, 30k iters,
  bs16, lr1e-4, LPIPS loss from step 5k. Eval: 256 held-out scenes, 8 uniform inputs / 4 midpoint targets,
  PSNR/LPIPS with **paired per-scene** stats. Baseline is PSNR 21.970 / LPIPS 0.2883. One small run ≈ 1.6 h
  on one B200. **Always compare seed-matched and report paired deltas** — single-seed PSNR gaps of ~0.1 dB
  are within seed noise (F18).
- **Launching:** the `run_*.sh` / `chain*.sh` / `launch_exp.sh` scripts in `lact_nvs/` chain train→eval and
  place runs on specific GPUs; `EXPERIMENT_QUEUE.md` tracks which GPU is free. Read the relevant script
  before launching rather than reconstructing the torchrun invocation.

### Terminology (user instruction, follow LaCT paper)

Use **"update"** (not "write") for the fast-weight gradient step, and **"apply"** (not "read") for using the
fast weights on queries — in code comments, docs, and the paper draft.

## Environment (rebuilt 2026-07-09 after node reset — read before launching anything)

The compute node is an ephemeral Slurm container: **home (`~`) and `/tmp` are node-local and vanish on
reset**; only lustre (`/NHNHOME/WORKSPACE/26msit001_A/...`) is durable. Sessions are interactive (2-day
limit) or batch via a web GUI (no `sbatch`/`squeue` CLI here — the user submits batch jobs manually).

**After any node reset, run first**:
`bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/setup_node.sh` — restores `~/.claude` →
lustre symlinks (Claude account, conversations, memory all live on lustre; `claude --continue` resumes
the previous conversation), puts the portable claude binary on PATH, and restores git identity +
credential helper. Nothing else about Claude needs recovering.

- **Python env**: venv at `/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm` (torch 2.11+cu128 for
  B200/sm_100; the old conda env path in git history is dead). All `lact_nvs/*.sh` already point here.
- **RE10K data**: source chunks survive at `/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/re10k`
  (train+test). Working copy must be reshared into node-local `/tmp/re10k` via
  `data_preprocess/reshard_re10k.py` (~2 min, 536 G tmpfs) — gone after every reset; reshard first.
- **Compile caches**: default `/tmp/torchinductor_*` is noexec → triton crashes mid-train ("failed to
  map segment"). `launch_exp.sh` now exports NFS caches (`.cache_triton_nvs`/`.cache_inductor_nvs`)
  + `TORCHINDUCTOR_COMPILE_THREADS=1`; keep that in any new launcher.
- **Batch workflow**: `lact_nvs/batch_entry.sh` is a self-contained batch entry point (reshards data,
  runs `BATCH_QUEUE.txt` jobs striped over all visible GPUs, **skips runs whose `eval.json` exists** —
  resubmission = resume). `batch_with_claude.sh` additionally runs headless Claude to update the dossier;
  it uses the portable install at `/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/` (binary +
  `CLAUDE_CONFIG_DIR` config), since `~/.claude` doesn't exist on batch nodes.
- **Interactive batch (`batch_remote.sh`)**: preferred long-running mode. Starts a
  **remote-control Claude session in tmux** the user chats with from claude.ai/code (named
  "ttt-batch"; URL also in `outputs/REMOTE_SESSION_URL.txt`), resuming the pinned conversation
  (`claude_portable/RESUME_SESSION`).
  **Claude is the sole executor** (policy 2026-07-09, user decision): no auto-started daemon. On
  every (re)start the script sends the session a kickoff message; Claude then assesses durable
  state (BATCH_QUEUE.txt incompletes, interrupted ccv checkpoints) and relaunches work itself.
  **Launch runs as your own background Bash tasks** (`run_in_background`) so the harness notifies
  you on completion — externally-started processes (nohup daemons) finish silently.
  `queue_daemon.sh` still exists as an optional helper Claude may start explicitly; when it (or
  any manual work) uses a GPU, claim it via `outputs/.gpu_locks/gpu<i>` (`echo <purpose> > …`,
  remove when done); `outputs/QUEUE_STATUS.txt` shows daemon claims. Git push works: credentials
  at `/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/.git-credentials` (wired via repo-local
  `credential.helper`).
- Checkpoints/evals land in `lact_nvs/outputs/` (lustre, durable). All pre-reset checkpoints were lost;
  results live only in `RESULTS_DOSSIER.md`.

## Gotchas

- **`train.py` reads `args.data_path` but never defines the `--data_path` argparse flag** (only `inference.py`
  does). Training will `AttributeError` until you add it. Training data is not released — the train script is
  reference code per the README.
- `--compile` gives ~1.4–1.5× speedup but the first step takes ~30 s–2 min to compile; **omit it while
  debugging**. It is separate from `--actckpt`, which must wrap the `Block` module (already wired in `train.py`)
  or backward compilation is hurt.
- All ray-map / pose preprocessing runs under `torch.autocast(enabled=False)` + `torch.no_grad()` (fp32) even
  though training is otherwise bf16 autocast — keep new pose math in fp32 to match.
- Checkpoints are saved with wrapper prefixes; `remove_module_prefix` strips `module.`, `_orig_mod.`,
  `_checkpoint_wrapped_module.` before saving the released `.pt`.
