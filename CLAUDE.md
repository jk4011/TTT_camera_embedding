# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research goal

This is a research repo. The objective (see `instruction.md`, written in Korean) is to develop a way to
**inject camera-pose information into LaCT / TTT layers** for novel view synthesis (NVS), analogous to how
PRoPE and RayRoPE inject camera info into *attention*. TTT (test-time training) replaces attention with a
linear-time fast-weight update, so relative positional-encoding tricks that rely on the attention Q·Kᵀ
matrix do not transfer directly — finding an equivalent for TTT is the open problem.

The intended workflow: brainstorm ~10 hypotheses for camera embedding in TTT, then implement and benchmark
them against the **LaCT LVSM baseline** in `lact_nvs/`, comparing PSNR/LPIPS. Hardware available is **4× B200**,
so up to 4 experiments run concurrently.

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
