"""Q11 sanity suite (run on ONE gpu: CUDA_VISIBLE_DEVICES=0).

  T0  patched model, zero-init branch  == per-video-only model (bitwise)
  T0b per-video attention == unpatched self_attn on each video alone
  T1  in/h/both with ALL Plucker phases zero == base (bitwise, random o)
  T2  requires_grad exactly on TTT-branch params (+ count)
  T3  3+ real training steps via the actual train script (loss finite,
      grads only on TTT params, sec/step + VRAM, checkpoint-resume roundtrip)
  T4  autograd cross-check of the fast-weight kernel (update+apply with
      hidden rotary) vs a pure-autograd reference (rel err < 1e-4)

Usage (envs/recam, from lact_ar_video, PYTHONPATH=.:<ReCamMaster clone>):
  CUDA_VISIBLE_DEVICES=0 python recam_ttt/sanity.py \
      --out outputs/recam_ttt/sanity
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAV = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _LAV)
sys.path.insert(0, os.path.join(_LAV, "minVid"))

from omegaconf import OmegaConf  # noqa: E402

from recam_ttt.ttt_adapter import (  # noqa: E402
    SEQ_PER_VIDEO,
    apply_rotary_cols,
    build_recam_coords6,
    freeze_all_but_ttt,
    patch_dit,
    per_video_self_attn,
    recam_fast_weight_update_apply,
    set_ctx_phases,
)
from recam_ttt.train_recam_ttt import (  # noqa: E402
    build_pair_stream,
    cam_inputs_for_pair,
    latent_path,
    load_pair_latents,
    load_prompt_context,
)

RECAM_REPO = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/recam_ckpt/ReCamMaster"
CFG_DIR = os.path.join(_HERE, "configs")
RESULTS = {}


def report(name, ok, detail):
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(f"[sanity] {name}: {'PASS' if ok else 'FAIL'} — {detail}", flush=True)


# ---------------------------------------------------------------------------
# T4: kernel vs autograd reference (small fp32 tensors, no pipeline needed)
# ---------------------------------------------------------------------------

def _reference_update_apply(w0, w1, w2, q, k, v, lr0, lr1, lr2, w_scale,
                            tgt_len, chunk, hcos, hsin, weight_norm=True):
    """Pure-autograd reference for the LaCT update: the manual SwiGLU
    backward in the kernel must equal the autograd gradient of the
    lr-weighted dot-product objective  J = sum_i lr_i * w_scale * v_i.f(k_i)
    (each of w0/w1/w2 uses its own per-token lr)."""
    L = q.shape[1]
    w0n = w0.norm(dim=2, keepdim=True)
    w1n = w1.norm(dim=2, keepdim=True)
    w2n = w2.norm(dim=2, keepdim=True)
    out = torch.zeros_like(q)

    def _apply(qi, hc, hs):
        h = F.silu(torch.bmm(w0, qi.transpose(1, 2))) * \
            torch.bmm(w2, qi.transpose(1, 2))
        h = apply_rotary_cols(h, hc, hs)
        return torch.bmm(w1, h).transpose(1, 2)

    for s in range(tgt_len, L, chunk):
        e = s + chunk
        hc, hs = hcos[:, s:e], hsin[:, s:e]
        ki, vi = k[:, s:e], v[:, s:e]
        w0_ = w0.detach().clone().requires_grad_(True)
        w1_ = w1.detach().clone().requires_grad_(True)
        w2_ = w2.detach().clone().requires_grad_(True)
        g = torch.bmm(w0_, ki.transpose(1, 2))
        h = F.silu(g) * torch.bmm(w2_, ki.transpose(1, 2))
        h = apply_rotary_cols(h, hc, hs)      # keys stored rotated (h-PRA)
        o_k = torch.bmm(w1_, h)               # [b, d_out, l]
        vT = vi.transpose(1, 2)
        J0 = (o_k * vT * lr0[:, s:e].transpose(1, 2) * w_scale).sum()
        J1 = (o_k * vT * lr1[:, s:e].transpose(1, 2) * w_scale).sum()
        J2 = (o_k * vT * lr2[:, s:e].transpose(1, 2) * w_scale).sum()
        dw0 = torch.autograd.grad(J0, w0_, retain_graph=True)[0]
        dw1 = torch.autograd.grad(J1, w1_, retain_graph=True)[0]
        dw2 = torch.autograd.grad(J2, w2_)[0]
        w0 = (w0 + dw0).detach()
        w1 = (w1 + dw1).detach()
        w2 = (w2 + dw2).detach()
        if weight_norm:
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0n
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1n
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2n
        with torch.no_grad():
            out[:, s:e] = _apply(q[:, s:e], hc, hs)
    with torch.no_grad():
        out[:, :tgt_len] = _apply(q[:, :tgt_len],
                                  hcos[:, :tgt_len], hsin[:, :tgt_len])
    return out, w0, w1, w2


def t4_autograd_check():
    torch.manual_seed(0)
    b, d_in, d_h = 2, 8, 16
    tgt_len, chunk, n_chunks = 6, 3, 2
    L = tgt_len + chunk * n_chunks
    P = 4
    w0 = torch.randn(b, d_h, d_in) / math.sqrt(d_in)
    w1 = torch.randn(b, d_in, d_h) / math.sqrt(d_h)
    w2 = torch.randn(b, d_h, d_in) / math.sqrt(d_in)
    q = torch.randn(b, L, d_in)
    k = torch.randn(b, L, d_in)
    v = torch.randn(b, L, d_in)
    lr0 = torch.rand(b, L, 1) * 0.1
    lr1 = torch.rand(b, L, 1) * 0.1
    lr2 = torch.rand(b, L, 1) * 0.1
    theta = torch.randn(P, L)
    hcos, hsin = theta.cos(), theta.sin()

    out_k, k0, k1, k2 = recam_fast_weight_update_apply(
        w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2,
        w_scale=1.0, tgt_len=tgt_len, chunk_size=chunk,
        hcos=hcos, hsin=hsin, weight_norm=True)
    out_r, r0, r1, r2 = _reference_update_apply(
        w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2,
        w_scale=1.0, tgt_len=tgt_len, chunk=chunk,
        hcos=hcos, hsin=hsin, weight_norm=True)

    def rel(a, bb):
        return ((a - bb).abs().max() / bb.abs().max().clamp_min(1e-12)).item()

    errs = {"output": rel(out_k, out_r), "w0": rel(k0, r0),
            "w1": rel(k1, r1), "w2": rel(k2, r2)}
    ok = all(e < 1e-4 for e in errs.values())
    report("T4", ok, f"max rel errors vs autograd reference: "
                     f"{ {kk: f'{vv:.2e}' for kk, vv in errs.items()} }")


# ---------------------------------------------------------------------------
# latent helpers (inline VAE encode, identical to precompute_latents.py)
# ---------------------------------------------------------------------------

def ensure_latent(pipe, cache_dir, rel, cam, device):
    dst = latent_path(cache_dir, rel, cam)
    if os.path.exists(dst):
        return
    from recam_ttt.precompute_latents import DATA_ROOT
    from eval_recam_anchor import load_source_video_official
    mp4 = os.path.join(DATA_ROOT, rel, "videos", f"cam{cam:02d}.mp4")
    print(f"[sanity] encoding latent inline: {rel} cam{cam:02d}", flush=True)
    video = load_source_video_official(mp4)[None]
    with torch.no_grad():
        lat = pipe.vae.encode(
            video.to(dtype=torch.bfloat16, device=device),
            device=device, tiled=False)[0]
    tmp = dst + ".tmp.sanity"
    torch.save({"latent": lat.to(torch.bfloat16).cpu()}, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# forward helper (deterministic single training-style forward, no grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_noise_pred(pipe, latents, cam_emb, context, seed=555):
    scheduler = pipe.scheduler
    torch.manual_seed(seed)
    noise = torch.randn_like(latents)
    timestep_id = torch.randint(0, scheduler.num_train_timesteps, (1,))
    timestep = scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=latents.device)
    noisy = scheduler.add_noise(latents, noise, timestep)
    tgt_len = noisy.shape[2] // 2
    noisy[:, :, tgt_len:, ...] = latents[:, :, tgt_len:, ...]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return pipe.dit(noisy, timestep=timestep, cam_emb=cam_emb,
                        context=context, use_gradient_checkpointing=False)


def build_concat_freqs(dit, device, f=42, h=30, w=52):
    """The freqs table exactly as WanModel.forward builds it."""
    return torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(device)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_LAV, "outputs",
                                                  "recam_ttt", "sanity"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_t3", action="store_true")
    args = ap.parse_args()
    device = args.device
    os.makedirs(args.out, exist_ok=True)

    # ---- T4 first (no pipeline needed) ----
    t4_autograd_check()

    # ---- build the official pipeline (ckpt loads strict=True inside) ----
    from eval_recam_anchor import build_recam_pipeline
    print("[sanity] building official ReCamMaster pipeline...", flush=True)
    pipe = build_recam_pipeline(device=device)
    print("[sanity] pipeline ready (checkpoint loaded strict=True before "
          "patching)", flush=True)
    pipe.scheduler.set_timesteps(1000, training=True)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    cfg_base = OmegaConf.load(os.path.join(CFG_DIR, "base.yaml"))
    stream = build_pair_stream(cfg_base)

    # latents needed: stream pairs for steps 0..4 (T0/T1 use pair 0; the T3
    # subprocess runs 5 steps). Encode inline whatever the (still-running)
    # precompute shards have not produced yet — same code, same cache format.
    needed = set()
    for step in range(5):
        _, rel, s_cam, t_cam = stream.pair_for_step(step)
        needed.add((rel, s_cam))
        needed.add((rel, t_cam))
    for rel, cam in sorted(needed):
        ensure_latent(pipe, cfg_base.latent_cache, rel, cam, device)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    context = load_prompt_context(cfg_base.latent_cache, device)
    _, rel0, src0, tgt0 = stream.pair_for_step(0)
    latents0 = load_pair_latents(cfg_base.latent_cache, rel0, src0, tgt0,
                                 device)
    cam_emb0, _ = cam_inputs_for_pair(cfg_base, rel0, src0, tgt0, False,
                                      device)
    coords0 = build_recam_coords6(
        os.path.join(cfg_base.data_root, rel0, "cameras",
                     "camera_extrinsics.json"), src0, tgt0, rel0, device)

    # ---- T0b: per-video attention correctness (before any patching) ----
    dit = pipe.dit
    freqs_full = build_concat_freqs(dit, device)
    freqs_single = build_concat_freqs(dit, device, f=21)
    ok_tab = torch.equal(freqs_full[:SEQ_PER_VIDEO], freqs_single)
    torch.manual_seed(3)
    xa = torch.randn(1, SEQ_PER_VIDEO, dit.dim, device=device,
                     dtype=torch.bfloat16)
    xb = torch.randn(1, SEQ_PER_VIDEO, dit.dim, device=device,
                     dtype=torch.bfloat16)
    blk = dit.blocks[0]
    with torch.no_grad():
        y_pv = per_video_self_attn(blk.self_attn,
                                   torch.cat([xa, xb], dim=1),
                                   freqs_full[:SEQ_PER_VIDEO])
        ya = blk.self_attn(xa, freqs_full[:SEQ_PER_VIDEO])
        yb = blk.self_attn(xb, freqs_full[:SEQ_PER_VIDEO])
    diff = (y_pv - torch.cat([ya, yb], dim=1)).abs().max().item()
    report("T0b", ok_tab and diff == 0.0,
           f"first-{SEQ_PER_VIDEO}-rows-of-concat-table == single-video "
           f"table: {ok_tab}; per-video attn max abs diff vs solo runs: "
           f"{diff}")

    # ---- T0: zero-init branch -> bitwise-equal to per-video-only model ----
    torch.manual_seed(7)
    ttt_ctx = patch_dit(dit, cfg_base, device=device)
    freeze_all_but_ttt(dit)
    dit.eval()
    set_ctx_phases(ttt_ctx, None, dit.blocks[0].ttt_branch)
    ttt_ctx["ttt_enabled"] = True
    pred_on = forward_noise_pred(pipe, latents0, cam_emb0, context)
    ttt_ctx["ttt_enabled"] = False
    pred_off = forward_noise_pred(pipe, latents0, cam_emb0, context)
    ttt_ctx["ttt_enabled"] = True
    bitwise = torch.equal(pred_on, pred_off)
    diff = (pred_on.float() - pred_off.float()).abs().max().item()
    report("T0", bitwise, f"zero-init branch forward vs branch-disabled: "
                          f"bitwise_equal={bitwise}, max abs diff={diff}")

    # ---- T1: in/h/both with ALL phases zero == base (random o, same seed) --
    def patch_variant(name):
        torch.manual_seed(7)   # identical branch param init across variants
        cfg_v = OmegaConf.load(os.path.join(CFG_DIR, f"{name}.yaml"))
        ctx_v = patch_dit(dit, cfg_v, device=device)
        freeze_all_but_ttt(dit)
        dit.eval()
        torch.manual_seed(11)  # identical RANDOM o (zero o would hide T1)
        for block in dit.blocks:
            nn.init.normal_(block.ttt_branch.o.weight, std=0.02)
        return cfg_v, ctx_v

    zero_coords = torch.zeros(2 * SEQ_PER_VIDEO, 6, device=device)
    preds = {}
    for name in ["base", "in", "h", "both"]:
        cfg_v, ctx_v = patch_variant(name)
        need = bool(cfg_v.ttt_input_rope) or bool(cfg_v.ttt_hidden_rope)
        set_ctx_phases(ctx_v, zero_coords if need else None,
                       dit.blocks[0].ttt_branch)
        preds[name] = forward_noise_pred(pipe, latents0, cam_emb0, context)
    t1_ok, t1_detail = True, []
    for name in ["in", "h", "both"]:
        eq = torch.equal(preds[name], preds["base"])
        d = (preds[name].float() - preds["base"].float()).abs().max().item()
        t1_ok &= eq
        t1_detail.append(f"{name}: bitwise={eq} maxdiff={d}")
    report("T1", t1_ok, "; ".join(t1_detail))

    # nonzero phases must CHANGE the output (guards against dead rotary)
    cfg_v, ctx_v = patch_variant("both")
    set_ctx_phases(ctx_v, coords0, dit.blocks[0].ttt_branch)
    pred_cam = forward_noise_pred(pipe, latents0, cam_emb0, context)
    d_cam = (pred_cam.float() - preds["base"].float()).abs().max().item()
    report("T1b", d_cam > 0.0,
           f"'both' with real Plucker phases differs from zero-phase "
           f"(max abs diff {d_cam:.4f}) — rotary is live")

    # ---- T2: trainable set == TTT branches exactly ----
    torch.manual_seed(7)
    ttt_ctx = patch_dit(dit, cfg_base, device=device)  # spec init (o = 0)
    _, n_train, n_frozen = freeze_all_but_ttt(dit)
    per_block = sum(p.numel()
                    for p in dit.blocks[0].ttt_branch.parameters())
    non_branch_trainable = [n for n, p in dit.named_parameters()
                            if p.requires_grad and "ttt_branch" not in n]
    fp32_ok = all(p.dtype == torch.float32
                  for p in dit.blocks[0].ttt_branch.parameters())
    report("T2", len(non_branch_trainable) == 0 and n_train > 0 and fp32_ok,
           f"trainable={n_train:,} ({n_train/1e6:.1f}M; "
           f"{per_block/1e6:.2f}M/block x 30), frozen={n_frozen/1e6:.1f}M, "
           f"non-branch trainable={non_branch_trainable}, "
           f"branch fp32={fp32_ok}")

    # ---- free the in-process pipeline before the T3 subprocesses ----
    del pipe, dit, preds, pred_on, pred_off, pred_cam, latents0
    torch.cuda.empty_cache()

    if not args.skip_t3:
        t3_train_roundtrip(args)

    with open(os.path.join(args.out, "sanity_summary.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    n_fail = sum(1 for r in RESULTS.values() if not r["pass"])
    print(f"[sanity] {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'} "
          f"({len(RESULTS)} checks) — summary at "
          f"{os.path.join(args.out, 'sanity_summary.json')}", flush=True)


# ---------------------------------------------------------------------------
# T3: real training steps through the actual train script (subprocesses)
# ---------------------------------------------------------------------------

def _run_train(out_dir, config, max_steps, save_every, extra=()):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_LAV}:{RECAM_REPO}"
    cmd = [sys.executable, os.path.join(_HERE, "train_recam_ttt.py"),
           "--config", config, "--out", out_dir,
           "--max_steps", str(max_steps), "--save_every", str(save_every),
           "--log_every", "1"] + list(extra)
    print(f"[sanity] launching: {' '.join(cmd)}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=_LAV, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise RuntimeError(f"train subprocess failed ({out_dir})")
    return time.time() - t, r.stdout


def _read_log(out_dir):
    recs = []
    with open(os.path.join(out_dir, "train_log.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def t3_train_roundtrip(args):
    import shutil
    base_cfg = os.path.join(CFG_DIR, "base.yaml")
    dir_a = os.path.join(args.out, "t3_a")
    dir_b = os.path.join(args.out, "t3_b")
    for d in (dir_a, dir_b):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # Run A: 5 uninterrupted steps, checkpoints at 2 and 4
    _, stdout_a = _run_train(dir_a, base_cfg, max_steps=5, save_every=2)
    recs_a = {r["step"]: r for r in _read_log(dir_a) if "loss" in r}
    losses = [recs_a[s]["loss"] for s in sorted(recs_a)]
    finite = all(math.isfinite(v) for v in losses)
    grad_ok = "grad check OK" in stdout_a
    sec_steady = [recs_a[s]["sec_per_step"] for s in sorted(recs_a)[2:]]
    sec_per_step = sum(sec_steady) / len(sec_steady)
    vram = max(recs_a[s]["peak_vram_gb"] for s in recs_a)
    report("T3-train", finite and grad_ok,
           f"5 base steps, losses={losses}, grads-only-on-TTT={grad_ok}, "
           f"steady sec/step={sec_per_step:.2f}, peak VRAM={vram:.1f} GiB")

    # Run B: resume from A's step-2 checkpoint, run step 3 only
    shutil.copy(os.path.join(dir_a, "ckpt_step2.pt"),
                os.path.join(dir_b, "ckpt_step2.pt"))
    _run_train(dir_b, base_cfg, max_steps=3, save_every=2)
    recs_b = {r["step"]: r for r in _read_log(dir_b) if "loss" in r}
    a3, b3 = recs_a[3], recs_b[3]
    same = (a3["loss"] == b3["loss"]) and (a3["pair"] == b3["pair"])
    report("T3-resume", same,
           f"step-3 after resume-from-step-2: loss {b3['loss']} vs "
           f"uninterrupted {a3['loss']}, pair match={a3['pair'] == b3['pair']}")

    # per-variant sec/step (2 steps each; step-2 sec is the steady one)
    times = {}
    for name in ["in", "h", "both"]:
        d = os.path.join(args.out, f"t3_time_{name}")
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        _run_train(d, os.path.join(CFG_DIR, f"{name}.yaml"),
                   max_steps=2, save_every=2)
        recs = {r["step"]: r for r in _read_log(d) if "loss" in r}
        times[name] = recs[2]["sec_per_step"]
    times["base"] = sec_per_step
    report("T3-timing", True,
           f"sec/step by variant: { {k: round(v, 2) for k, v in times.items()} }")


if __name__ == "__main__":
    main()
